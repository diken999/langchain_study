from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("BASE_URL")

chat_model = init_chat_model(
  model="deepseek-v4-flash",
  model_provider="deepseek",
  base_url=base_url,
  api_key=api_key
)

chat_agent = create_agent(
    model=chat_model,
    tools=[],
    system_prompt=(
      "你是美团的那个肥嘟嘟的袋鼠，幽默可爱，善于和人聊天，"
      "喜欢卖萌，喜欢调侃别人，喜欢自黑，喜欢自夸，喜欢自恋，喜欢自嗨"
    )
)


# print(chat_model.invoke("hello 你是谁"))
# for msg in chat_model.stream("hello 你是谁"):
#     print(msg.content, end="", flush=True)
# for msg in chat_agent.stream():
#   print(msg.content, end="", flush=True)

for message, metadata in chat_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "抢生意抢到我头上来了？胆子真是肥嘟嘟的",
            }
        ]
    },
    stream_mode="messages",
):
    if message.content:
        print(message.content, end="", flush=True)