import requests
import json
import xml.etree.ElementTree as ET


ENDPOINT = "https://apis.data.go.kr/1160100/service/GetKrxListedInfoService/getItemInfo?serviceKey="


params = {
    'numOfRows': 1000,  # 데이터 수
    'pageNo': 1,      # 페이지 번호
}

result_list = [] #결과 저장할 리스트

try:
    response = requests.get(ENDPOINT, params=params) 
    print("응답 상태 코드:", response.status_code)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    items = root.findall(".//item")

    for item in items:
        KoreaCompany = {
            'itmsNm': item.findtext('itmsNm'),  # 주식 종목 이름
            'corpNm': item.findtext('corpNm')   # 회사 이름
        }
        result_list.append(KoreaCompany)

    # JSON save 함
    with open('KoreaCompany.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_list, json_file, ensure_ascii=False, indent=4)

    print("JSON 파일로 저장 완료: KoreaCompany.json")

except requests.exceptions.RequestException as e:
    print(f"API 요청 오류 발생: {e}")
except ET.ParseError as e:
    print(f"XML 파싱 오류 발생: {e}")
except Exception as e:
    print(f"기타 오류 발생: {e}")
