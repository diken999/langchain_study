from langchain.chat_models import init_chat_model
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

response = chat_model.invoke("Hello, how are you?")
print(response)