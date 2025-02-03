# main.py
from pdf_loader import load_pdf, save_pdf_results
from open_ai import load_openai_api_key, generate_response

FILE_PATH = './pdf_lecture/module/2024년도_전략기술_테마별_프로젝트(DCP)_제2차_기술수요조사_시행계획_공고.pdf'
OUTPUT_FILE = "./pdf_lecture/module/PDF_Loader_result.txt"

# PDF 파일 로드
pdf_data = load_pdf(FILE_PATH)

# 로드한 데이터를 파일에 저장
save_pdf_results(OUTPUT_FILE, pdf_data)

# 저장된 결과를 파일에서 읽기
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    result_load = f.read()

# OpenAI API 키 로드
OPENAI_API_KEY = load_openai_api_key()

# 모델에 데이터를 입력하고 결과 출력
response = generate_response(result_load, OPENAI_API_KEY)
print(response)

SCORE_FILE = "./pdf_lecture/module/score.md"
# response를 score.md 파일로 저장
with open(SCORE_FILE, "w", encoding="utf-8") as f:
    f.write(response)