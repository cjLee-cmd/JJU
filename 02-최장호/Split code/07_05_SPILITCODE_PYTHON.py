%pip install -qU langchain-text-splitters

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)



PYTHON_CODE = """
def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("A의 열의 수와 B의 행의 수가 일치해야 합니다.")
    

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

A = [[1, 2, 3],
     [4, 5, 6]]
B = [[1, 4],
     [2, 5],
     [3, 6]]

result = matrix_multiply(A, B)
for row in result:
    print(row)

"""

python_docs = python_splitter.create_documents([PYTHON_CODE])

for doc in python_docs:
    print(doc.page_content, end="\n==================\n")

