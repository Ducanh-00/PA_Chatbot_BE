import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print(f"Listing models...")

try:
    for m in client.models.list():
        # In tên model ra trước
        print(f"Model Name: {m.name}")
except Exception as e:
    print(f"Error: {e}")
