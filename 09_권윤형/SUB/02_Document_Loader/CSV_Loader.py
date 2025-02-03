from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./data/boston_housing.csv')
loader = UnstructuredCSVLoader(file_path='./data/boston_housing.csv', mode='elements')

docs = loader.load()

# print(docs[0].metadata["text_as_html"][:1000])
# print(docs[0].page_content)

lines = docs[0].page_content.strip().split('\n')
header = lines[0].split()
rows = [line.split() for line in lines[1:]]

xml_data = '<data>\n'
for row in rows:
    xml_data += '  <row>\n'
    for i, value in enumerate(row):
        xml_data += f'    <{header[i]}>{value}</{header[i]}>\n'
    xml_data += '  </row>\n'
xml_data += '</data>'

print(xml_data)