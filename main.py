from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
# api_key = os.getenv("DEEPSEEK_API_KEY")
# base_url = os.getenv("BASE_URL")

chat_model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    # base_url=base_url,
    # api_key=api_key
)

agent = create_agent(chat_model)

# response = chat_model.invoke("Hello, how are you?")
# response = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "你好👋"}
#     ]
# })
response_stream = agent.stream(
    {
        "messages": [{"role": "user", "content": "你好👋，介绍一下你自己"}],
    },
    stream_mode="messages"
)
# print(response)
# for msg in response["messages"]:
#     print(msg.model_dump_json(indent=2))
print(type(response_stream))
for message, metadata in response_stream:
    if message.content:
        print(message.content, end="", flush=True)
print()