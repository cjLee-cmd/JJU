from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
import xml.etree.ElementTree as ET
import pandas as pd

# (1) CSV 로드 함수
def UCL(file_path):
    """
    UnstructuredCSVLoader를 이용해 CSV 파일을 로드한 뒤
    Document 객체 리스트를 반환합니다.
    """
    loader = UnstructuredCSVLoader(file_path=file_path, mode="elements")
    docs = loader.load()
    return docs

# (2) 상위 N개의 Document를 XML 문자열로 변환하는 함수
def str_to_xml(docs, N=10):
    """
    주어진 docs(List[Document])에서 상위 N개를 XML 형식으로 변환한 뒤
    XML 문자열을 반환합니다.
    """
    root = ET.Element("root")

    # 상위 N개만 변환
    for idx, doc in enumerate(docs[:N], start=1):
        item = ET.SubElement(root, "item")

        # page_content
        pc_elem = ET.SubElement(item, "page_content")
        pc_elem.text = doc.page_content if doc.page_content else ""

        # metadata
        metadata_elem = ET.SubElement(item, "metadata")
        for k, v in doc.metadata.items():
            meta_child = ET.SubElement(metadata_elem, k)
            meta_child.text = str(v)

    # XML 트리를 문자열로 변환
    xml_string = ET.tostring(root, encoding="utf-8").decode("utf-8")
    return xml_string

# (3) XML 형식으로 상위 10개 문서를 출력(또는 반환)하는 함수
def head(docs, N=10):
    """
    상위 N개 문서를 XML 문자열로 변환하여 출력합니다.
    필요에 따라 return xml_data 를 사용하여 반환할 수도 있습니다.
    """
    xml_data = str_to_xml(docs, N=N)
    print(xml_data)

# (4) boston.csv 내 'MEDV' 컬럼(주택 가격)에 가장 크게 영향을 미치는
#     상위 3가지 요소를 찾는 함수 (단순 상관계수 기반 예시)
def top_3(file_path):
    """
    CSV를 pandas로 불러오고, 'MEDV'와의 상관계수를 기준으로
    가장 큰 요소 상위 3가지를 찾아 리스트 형태로 반환합니다.
    """
    df = pd.read_csv(file_path)
    # numeric_only=True는 pandas 버전에 따라 필요할 수도, 아닐 수도 있습니다.
    corr_series = df.corr(numeric_only=True)['MEDV'].abs().sort_values(ascending=False)
    # 첫 번째가 'MEDV'이므로, 인덱스 1~3까지
    top_3 = corr_series.index[1:4].tolist()
    return top_3

if __name__ == "__main__":
    # 윈도우 환경에서의 실제 CSV 경로 (백슬래시 -> \\ 이스케이프)
    file_path = "C:\\Users\\koll2\\OneDrive\\program_A\\2024_비트컴퓨터\\과제\\data\\boston.csv"
    
    # 1) CSV 파일 로드
    docs = UCL(file_path)

    # 2) 상위 10개의 Document를 XML로 변환해 출력
    head(docs, N=10)
    
    # 3) 'MEDV'와의 상관계수가 높은 상위 3개 요소를 찾기
    factors = top_3(file_path)
    print("주택 가격(MEDV)에 영향을 주는 상위 3가지 요소:", factors)


# 시간 없어서 결국 굴복하고 gpt사용.......................................................................................................................................