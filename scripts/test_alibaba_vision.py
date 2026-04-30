import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_qwen_vision():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found in .env")
        return

    # Khởi tạo client OpenAI trỏ tới Alibaba Cloud DashScope
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print("--- TESTING QWEN-VL-MAX ---")
    
    try:
        # Thử nghiệm với một hình ảnh mẫu (Logo Petrolimex hoặc bất kỳ ảnh nào trên internet)
        # Ở đây tôi dùng một ảnh ví dụ về sơ đồ để test khả năng đọc hiểu
        image_url = "https://img.alicdn.com/tfs/TB1_uT8nSrI8KJjSsphXXbf3FXa-117-117.png" # Ảnh logo mặc định của Alibaba để test
        
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hình ảnh này có nội dung gì? Hãy mô tả bằng tiếng Việt."},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )

        print("\nKết quả từ Qwen-VL-Max:")
        print(completion.choices[0].message.content)
        print("\n✅ API KEY HOẠT ĐỘNG TỐT!")

    except Exception as e:
        print(f"\n❌ LỖI KHI GỌI API: {str(e)}")

if __name__ == "__main__":
    test_qwen_vision()
