import requests
import json


def test_alibaba_api():
    url = "http://localhost:8000/api/chat/alibaba"
    payload = {
        "question": "Trong quy trình kiểm soát thông tin dạng văn bản, tại Bước 2 nếu kết quả xem xét và phê duyệt là 'Không phù hợp' thì bước tiếp theo phải làm gì?",
        "session_id": "test_alibaba",
    }
    headers = {"Content-Type": "application/json"}

    print(f"🚀 Sending request to Alibaba API: {url}")
    print(f"❓ Question: {payload['question']}")
    print("-" * 50)

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            result = response.json()
            print("\n🤖 ALIBABA ANSWER:")
            print(result["answer"])
            print("\n📚 SOURCES:")
            for source in result["sources"]:
                print(
                    f"- {source['file']} (Trang: {source['page']}) [Score: {source['score']}]"
                )
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    test_alibaba_api()
