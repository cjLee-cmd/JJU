import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
    }
    if not env_vars["HUGGINGFACE_API_KEY"]:
        print("HUGGINGFACE_API_KEY 환경 변수가 설정되지 않았습니다. HuggingFace API가 필요할 수 있습니다.")
    return env_vars
