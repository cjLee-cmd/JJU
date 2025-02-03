import os
from dotenv import load_dotenv
import mimetypes
import openai

# 1. .env 파일 로드
load_dotenv()

# 2. 파일 형식 판별 함수
def detect_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type

# 3. 로더 선택 함수
def select_loader(file_type):
    if 'pdf' in file_type:
        return ["PDFLoader1", "PDFLoader2", "PDFLoader3"]
    elif 'text' in file_type:
        return ["txtLoader"]
    elif 'json' in file_type:
        return ["JsonLoader"]
    else:
        return []

# 4. 스플리터 정의
def get_splitters():
    return [
        "CharacterTextSplitter",
        "TokenTextSplitter",
        "RecursiveCharacterTextSplitter"
    ]

# 5. OpenAI API 활용 함수 (최신 API 사용)
def evaluate_splits(api_key, loader, splitter, data):
    openai.api_key = api_key
    prompt = f"Loader: {loader}\nSplitter: {splitter}\nData: {data}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 최신 GPT-4 모델
        messages=[
            {"role": "system", "content": "You are an assistant evaluating document splits."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response

# 6. 메인 함수
def main():
    file_path = input("파일 경로를 입력하세요: ")
    file_type = detect_file_type(file_path)
    loaders = select_loader(file_type)
    splitters = get_splitters()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: 환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    results = []
    for loader in loaders:
        for splitter in splitters:
            data = f"Simulated data using {loader}"
            result = evaluate_splits(api_key, loader, splitter, data)
            results.append({
                "loader": loader,
                "splitter": splitter,
                "result": result
            })
    
    best_combination = max(results, key=lambda x: x['result']['choices'][0]['message']['content'])
    print(f"최적 조합: {best_combination}")

if __name__ == "__main__":
    main()
