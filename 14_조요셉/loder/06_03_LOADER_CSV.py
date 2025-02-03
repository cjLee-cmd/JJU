import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

# 비구조화 CSV 로더 인스턴스 생성
loader = UnstructuredCSVLoader(file_path="loder\data\BostonHousing.csv", mode="elements")

# 문서 로드
docs = loader.load()
print(len(docs))
# 첫 번째 문서의 HTML 텍스트 메타데이터 출력
print(docs[0].metadata["text_as_html"])

with open("loder\SUB\RESULT\CSVLOADER_BOSTONHOUSING_RESULT.txt", "w", encoding="utf-8") as file:
    for doc in docs:
        row = doc.metadata["text_as_html"].split('<tr>')
        print(len(row))
        row_str = '<row>'

        for element in row[:12]: # 12개의 엘리먼트만 추출
            splitted_element = element.split(':')
            value = splitted_element[-1]
            col = ".".join(splitted_element[:-1])
            row_str += f"<{col}>{value.strip()}</{col}>"

        row_str += "</row>\n"
        file.write(row_str)

print("파일 저장 완료: /SUB/RESULT/CSVLOADER_BOSTONHOUSING_RESULT.txt")

# pandas DataFrame으로 변환
df = pd.read_csv("loder\data\BostonHousing.csv")

# 데이터의 숫자형 컬럼만 선택
numeric_df = df.apply(pd.to_numeric, errors='coerce')

# 상관 행렬 계산
correlation_matrix = numeric_df.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# 히트맵을 파일로 저장
folder_path = "loder/SUB/RESULT"
plt.savefig(os.path.join(folder_path, "heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# 상위 3개 상관관계 추출 (절댓값 기준으로)
corr_pairs = correlation_matrix.unstack().sort_values(ascending=False, key=abs)

# 자기 자신과의 상관 관계를 제외하고 상위 3개 상관 관계 출력
top_3_corr = corr_pairs[corr_pairs < 1].head(3)
print("상위 3개 상관 관계:")
print(top_3_corr)