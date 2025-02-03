import re
import shutil
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter

def print_filled_line():
    terminal_width = shutil.get_terminal_size().columns
    print('=' * terminal_width)

# GPT-2 토크나이저 로드
hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 데이터 파일 읽기
with open("과제/data/appendix-keywords.txt", 'r', encoding='utf-8') as f:
    file = f.read()

# 정규화: 제목과 내용을 합치기 전에 줄바꿈 문제 수정
normalized_file = re.sub(
    r"(\n연관키워드: [^\n]+)\n([A-Z][^\n]+)",  # "연관키워드:" 뒤에 제목이 이어지는 경우
    r"\1\n\n\2",  # 제목 앞에 줄바꿈 추가
    file
)

# CharacterTextSplitter 설정
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    hf_tokenizer,
    chunk_size=100,      # 충분히 큰 청크 크기
    chunk_overlap=0       # 중첩 없음
)

# 텍스트 분할
text = text_splitter.split_text(normalized_file)

# 0과 1, 2와 3, ... 이런 식으로 두 개씩 합쳐 출력
for i in range(0, len(text), 2):
    if i + 1 < len(text):  # 두 개씩 묶을 수 있는 경우
        combined_text = f"{text[i].strip()}\n\n{text[i+1].strip()}"
    else:  # 홀수 개라 마지막 하나만 남은 경우
        combined_text = text[i].strip()
    print_filled_line()
    print(combined_text)
    print_filled_line()