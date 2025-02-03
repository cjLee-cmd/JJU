# 필요한 라이브러리 임포트
import os
from datetime import datetime
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# API 키 설정
api_key = 'your api key'
os.environ['OPENAI_API_KEY'] = api_key

def get_today():
    return datetime.now().strftime('%B %d')

output = '''
    Day 1: 한국-베이징
        - 아침: 한국에서 비행기 타고 출발하여 베이징으로 이동
        - 점심: 호텔 체크인 후, 중국 전통 요리를 맛볼 수 있는 식당에서 점심 식사
        - 오후: 자금성 방문하여 중국의 역사와 문화를 체험
        - 저녁: 왕푸징 거리에서 쇼핑 및 거리 음식 체험
        - 밤: 호텔로 돌아와 휴식

    Day 2: 베이징
        - 아침: 호텔에서 조식 후, 천안문 광장 방문
        - 점심: 베이징 오리 요리 맛보기
        - 오후: 이화원 방문하여 아름다운 정원 산책
        - 저녁: 후퉁 지역에서 전통 차 체험
        - 밤: 베이징 야경 투어

    Day 3: 베이징
        - 아침: 호텔에서 조식 후, 베이징 동물원 방문하여 팬더 관람
        - 점심: 현지 식당에서 베이징 특선 요리 맛보기
        - 오후: 798 예술구 방문하여 현대 미술 감상
        - 저녁: 베이징 전통 공연 관람
        - 밤: 호텔로 돌아와 휴식

    Day 4: 베이징-한국
        - 아침: 호텔에서 조식 후, 체크아웃
        - 오전: 베이징 시내 자유 시간
        - 점심: 현지 식당에서 점심 식사
        - 오후: 공항으로 이동하여 한국행 비행기 탑승
        - 저녁: 한국 도착
'''

travle_promp = PromptTemplate(
    template='''
    오늘의 날짜는 {today}입니다.
    {today}부터 {n}일간 {country}의 여행하기 좋은 여행지를 추천해 주세요.
    그리고 {n}일간의 여행계획을 자세하게 만들어 주세요.
    출력 결과는 다음과 같아야 합니다:
    예시 출력: {output}
    위 출력 결과를 참고하여 {country}의 {n}일간의 여행계획을 자세하게 만들어 주세요.`
    ''',
    input_variables=['country', 'n'],
    partial_variables={'today': get_today(), 'output' : output}
)

model = ChatOpenAI(
    model="gpt-4",
    max_tokens=2048,
    temperature=0.1
)

chain = travle_promp | model

country='일본'
n=2

input_prompt = {'country':country, 'n':n}
content = chain.invoke(input_prompt).content

prompt = travle_promp.format(country=country, n=n)
print(prompt+'\n\n')

print(content)
