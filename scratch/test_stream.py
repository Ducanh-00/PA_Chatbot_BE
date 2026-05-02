import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get("ALIBABA_CHAT_MODEL", "qwen3.6-plus"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    streaming=True,
    model_kwargs={"extra_body": {"enable_thinking": True}}
)

for chunk in llm.stream("Thủ đô của Việt Nam là gì?"):
    print(f"Content: {repr(chunk.content)} | Additional: {chunk.additional_kwargs}")
