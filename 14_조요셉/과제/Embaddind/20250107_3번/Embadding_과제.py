from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# OpenAI의 "text-embedding-3-large" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

load_dotenv()

sentence1 = "안녕하세요? 반갑습니다."
sentence2 = "안녕하세요? 반갑습니다!"
sentence3 = "안녕? 만나서 반가워."
sentence4 = "Hi, nice to meet you."
sentence5 = "I like to eat apples."
sentence6 = "Hi? Freut mich, Sie kennenzulernen"

# 원하는 데이터 구조를 정의합니다.
class Topic(BaseModel):
    translation: str = Field(translation="번역문, 한글을 입력 받으면 번역하지 마.")
    availability: str = Field(availability="번역 유무(True/False) 한글을 입력 받으면 무조건 False를 반환해.")

# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
parser = JsonOutputParser(pydantic_object=Topic)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 프롬프트 템플릿을 생성합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 통역일을 20년간 했으며 세계 모든 언어를 할 줄 아는 통역사야. 질문을 받으면 한글로 통역해줘.번역만 답으로 해."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,

)

chain = prompt | llm | parser

response = chain.invoke({"question": "Hi? Freut mich, Sie kennenzulernen"})
print(response)

from sklearn.metrics.pairwise import cosine_similarity

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6]
trans_sentences = sentences.copy()
for i, sentence in enumerate(sentences):
    print(sentence)
    response = chain.invoke({"question": sentence})
    print(response)
    if response["availability"] == "True":
      trans_sentences[i] = response['translation']
      print(f"response : {response}")

scaled_embeddings=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

print(f"sentences : {sentences}")
embedded_sentences=scaled_embeddings.embed_documents(trans_sentences)

for i, sentence in enumerate(embedded_sentences):
    for j, other_sentence in enumerate(embedded_sentences):
        if i < j:
            print(
                f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
            )