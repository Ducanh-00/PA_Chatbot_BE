import os
import sys
import json
import uuid
import hashlib
import glob
import time
import io
from PIL import Image

# Fix Unicode issue for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGED_IMAGES_DIR = os.path.join(BASE_DIR, "data", "staged_images")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pa_procedures"

def generate_chunk_id(chunk: Document) -> str:
    content_to_hash = f"{chunk.metadata.get('source_file', '')}_{chunk.page_content}"
    hash_hex = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex))

def resize_image_if_needed(img_path, max_size=768):
    """Tối ưu kích thước ảnh để tăng tốc độ xử lý GPU."""
    with Image.open(img_path) as img:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            
            temp_path = img_path.replace(".jpg", "_temp.jpg")
            img.save(temp_path, "JPEG", quality=80)
            return temp_path
    return img_path

def process_image_qwen(model, processor, img_path):
    processed_img_path = resize_image_if_needed(img_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": processed_img_path,
                },
                {"type": "text", "text": "Extract all process steps and logical flow from this image into Markdown. Be concise. Respond ONLY with Markdown."},
            ],
        }
    ]
    
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    finally:
        if processed_img_path != img_path and os.path.exists(processed_img_path):
            os.remove(processed_img_path)

def main():
    print("=======================================", flush=True)
    print("STEP 2: SUPER FAST GPU VISION EXTRACTION", flush=True)
    print("=======================================", flush=True)
    
    json_files = glob.glob(os.path.join(STAGED_IMAGES_DIR, "*.json"))
    if not json_files:
        print(f"No staged images found. Everything is done!", flush=True)
        return
        
    print(f"Found {len(json_files)} images waiting for GPU processing.", flush=True)
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("1. Loading Qwen2-VL-2B (4-bit Mode) on GPU...", flush=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", 
            quantization_config=bnb_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        print("   Model & Processor loaded (Optimized).", flush=True)
    except Exception as e:
        print(f"   CRITICAL ERROR: {e}", flush=True)
        return

    print("2. Init VectorStore & Embeddings...", flush=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    start_time = time.time()
    
    for i, json_path in enumerate(json_files):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        print(f"\n[{i+1}/{len(json_files)}] Processing... (Avg: {avg_time:.1f}s/img)", flush=True)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            img_path = json_path.replace(".json", ".jpg")
            if not os.path.exists(img_path):
                continue
            
            try:
                content = process_image_qwen(model, processor, img_path)
                extracted_by = "qwen2-vl"
                print(f"     ✅ AI Done.", flush=True)
            except Exception as e:
                print(f"     ❌ Error: {e}", flush=True)
                content = metadata.get("fallback_text", "")
                extracted_by = "pymupdf4llm_fallback"
                
            if content and content.strip():
                md_splits = markdown_splitter.split_text(content)
                chunks = text_splitter.split_documents(md_splits)
                for chunk in chunks:
                    chunk.metadata.update({
                        "category": metadata.get("category", ""),
                        "source_file": metadata.get("source_file", ""),
                        "page": metadata.get("page", ""),
                        "extracted_by": extracted_by
                    })
                
                if chunks:
                    ids = [generate_chunk_id(c) for c in chunks]
                    vector_store.add_documents(documents=chunks, ids=ids)
            
            # Cleanup
            os.remove(json_path)
            if os.path.exists(img_path): os.remove(img_path)
                
        except Exception as e:
            print(f"  ❌ Unhandled: {e}", flush=True)

    print(f"\n✅ All Done! Total time: {time.time() - start_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
