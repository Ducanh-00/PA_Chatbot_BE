"""
Script convert tất cả file .doc (Word 97-2003) sang .docx
Dùng textutil (có sẵn trên macOS, KHÔNG cần cài thêm gì)
Chạy: python3 -m scripts.convert_doc
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

print(f"Scanning for .doc files in: {DATA_DIR}\n")

converted = 0
skipped = 0
failed = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        # Chỉ xử lý file .doc (KHÔNG phải .docx)
        if not file.lower().endswith(".doc"):
            continue
        if file.lower().endswith(".docx"):
            continue
        if file.startswith("."):
            continue

        doc_path = os.path.join(root, file)
        docx_name = os.path.splitext(file)[0] + ".docx"
        docx_path = os.path.join(root, docx_name)

        # Nếu bản .docx đã tồn tại thì bỏ qua
        if os.path.exists(docx_path):
            print(f"Already exist: {docx_name}")
            skipped += 1
            continue

        print(f"Converting: {file}")

        try:
            result = subprocess.run(
                ["textutil", "-convert", "docx", doc_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(docx_path):
                print(f"✅ -> {docx_name}")
                converted += 1
            else:
                print(f"Error: {result.stderr.strip()}")
                failed += 1
        except Exception as e:
            print(f"Error: {e}")
            failed += 1

print(f"\n{'=' * 50}")
print(f"Kết quả: {converted} converted, {skipped} skipped, {failed} failed")
print(f"{'=' * 50}")
