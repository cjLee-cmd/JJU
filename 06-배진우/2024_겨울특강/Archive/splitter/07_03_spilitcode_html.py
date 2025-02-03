import requests
from langchain.text_splitter import HTMLHeaderTextSplitter
from bs4 import BeautifulSoup
import logging
import sys
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("advanced_html_splitter.log")
    ]
)

def fetch_html(url):
    """
    주어진 URL로부터 HTML 콘텐츠를 가져옵니다.
    """
    try:
        logging.info(f"URL에 접속 중: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logging.info("HTML 콘텐츠를 성공적으로 가져왔습니다.")
        return response.text
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP 오류 발생: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"연결 오류 발생: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"타임아웃 오류 발생: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"요청 오류 발생: {req_err}")
    return None

def clean_html(html_content):
    """
    BeautifulSoup을 사용하여 HTML을 정리하고 텍스트를 추출합니다.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 스크립트와 스타일 태그 제거
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    # 불필요한 공백 제거
    cleaned_html = ' '.join(soup.stripped_strings)
    logging.info("HTML 콘텐츠를 정리했습니다.")
    return cleaned_html

def split_html(html_content, headers_to_split_on):
    """
    HTMLHeaderTextSplitter를 사용하여 HTML 콘텐츠를 지정된 헤더 기준으로 분할합니다.
    """
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = splitter.split_text(html_content)
    logging.info(f"HTML 콘텐츠를 {len(splits)}개의 섹션으로 분할했습니다.")
    return splits

def structure_splits(splits):
    """
    분할된 섹션을 구조화된 데이터로 변환합니다.
    """
    structured_data = []
    for split in splits:
        section = {
            "content": split.page_content,
            "metadata": split.metadata
        }
        structured_data.append(section)
    logging.info("분할된 섹션을 구조화된 데이터로 변환했습니다.")
    return structured_data

def save_splits_to_json(splits, filename):
    """
    분할된 섹션을 JSON 파일에 저장합니다.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(splits, f, ensure_ascii=False, indent=4)
        logging.info(f"분할된 내용을 '{filename}' 파일에 JSON 형식으로 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생: {e}")

def main():
    # 전주대학교 공식 웹사이트 URL
    url = 'https://www.jj.ac.kr/jj/main.jsp'
    
    # HTML 가져오기
    html_string = fetch_html(url)
    if not html_string:
        logging.error("HTML 콘텐츠를 가져오지 못했습니다. 프로그램을 종료합니다.")
        return
    
    # HTML 정리
    cleaned_html = clean_html(html_string)
    
    # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
        ("h5", "Header 5"),
        ("h6", "Header 6"),
    ]
    
    # HTML 분할
    html_header_splits = split_html(cleaned_html, headers_to_split_on)
    
    # 분할된 섹션을 구조화된 데이터로 변환
    structured_splits = structure_splits(html_header_splits)
    
    # 필요 시 콘솔에 출력 (옵션)
    for idx, section in enumerate(structured_splits, start=1):
        print(f"===== 섹션 {idx} =====")
        print(f"내용:\n{section['content']}")
        print(f"메타데이터: {section['metadata']}", end="\n=====================================================\n")

if __name__ == "__main__":
    main()