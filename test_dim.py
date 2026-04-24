import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

model_name = "models/gemini-embedding-2"
print(f"Checking dimension for {model_name}...")

try:
    res = client.models.embed_content(model=model_name, contents="test")
    dim = len(res.embeddings[0].values)
    print(f"Dimension: {dim}")
except Exception as e:
    print(f"Error: {e}")
