import requests
import cv2
import streamlit as st
import json
import time

# 기본 API 설정
API_KEY = "your api key"
BASE_URL = "https://openapi.its.go.kr:9443/cctvInfo"

def fetch_cctv_info(min_x, max_x, min_y, max_y):
    """
    API에서 CCTV 정보를 가져옵니다.
    """
    params = {
        "apiKey": API_KEY,
        "type": "국도",
        "cctvType": 1,
        "minX": min_x,
        "maxX": max_x,
        "minY": min_y,
        "maxY": max_y,
        "getType": "json"
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("CCTV 정보를 가져오지 못했습니다. API 요청을 확인하세요.")
        return None

def parse_cctv_info(json_data):
    """
    JSON 데이터를 파싱하여 CCTV 목록을 반환합니다.
    """
    cctv_list = []
    data = json_data.get("response", {}).get("data", [])

    for item in data:
        cctv_name = item.get("cctvname")
        coord_x = item.get("coordx")
        coord_y = item.get("coordy")
        cctv_url = item.get("cctvurl")

        cctv_list.append({
            "name": cctv_name,
            "latitude": coord_y,
            "longitude": coord_x,
            "url": cctv_url
        })

    return cctv_list

def run_cctv_stream(url):
    # 영상 스트림 열기
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        st.error("CCTV 스트림을 열 수 없습니다. URL을 확인하세요.")
        return

    # Streamlit용 비디오 렌더링
    placeholder = st.empty()

    # 프레임 속도 설정 (60 FPS = 1초에 60 프레임, 한 프레임당 약 0.016초 지연)
    frame_delay = 1 / 30

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            st.warning("스트림을 읽을 수 없습니다. 연결 상태를 확인하세요.")
            break

        # OpenCV BGR 이미지를 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Streamlit 이미지 업데이트
        placeholder.image(frame, channels="RGB", use_container_width=True)

        # 60 FPS를 위한 지연 설정
        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed_time))

    cap.release()
    st.write("CCTV 스트림이 종료되었습니다.")

def main():
    st.title("ITS 국도 CCTV 정보 뷰어")
    st.sidebar.title("검색 영역 설정")

    # 사용자가 검색할 영역 설정
    min_x = st.sidebar.number_input("최소 X 좌표 (minX)", value=127.0)
    max_x = st.sidebar.number_input("최대 X 좌표 (maxX)", value=127.3)
    min_y = st.sidebar.number_input("최소 Y 좌표 (minY)", value=35.7)
    max_y = st.sidebar.number_input("최대 Y 좌표 (maxY)", value=36.0)

    # session_state 초기화 (CCTV 정보 및 선택된 CCTV 상태 유지)
    if "cctv_list" not in st.session_state:
        st.session_state.cctv_list = []
    if "selected_cctv" not in st.session_state:
        st.session_state.selected_cctv = None
    if "show_url" not in st.session_state:
        st.session_state.show_url = False

    # 검색 버튼
    if st.sidebar.button("CCTV 검색"):
        json_data = fetch_cctv_info(min_x, max_x, min_y, max_y)
        if json_data:
            st.session_state.cctv_list = parse_cctv_info(json_data)
            if not st.session_state.cctv_list:
                st.warning("해당 영역에서 CCTV 정보를 찾을 수 없습니다.")
            else:
                # 지도에 CCTV 위치 표시
                location_data = [{"lat": cctv["latitude"], "lon": cctv["longitude"]} for cctv in st.session_state.cctv_list]
                st.map(location_data)

                # CCTV 목록 표시
                st.session_state.selected_cctv = st.selectbox("CCTV를 선택하세요:", st.session_state.cctv_list, format_func=lambda x: x["name"])
                st.write(f"선택된 CCTV: {st.session_state.selected_cctv['name']}")

   # CCTV 스트림 URL 보기 버튼
    if st.button("CCTV URL 보기"):
        if st.session_state.selected_cctv:
            # 선택된 CCTV의 이름과 URL을 보여주고 스트리밍 시작
            st.write(f"선택된 CCTV: {st.session_state.selected_cctv['name']}")
            run_cctv_stream(st.session_state.selected_cctv['url'])
        else:
            st.warning("CCTV를 선택해 주세요.")

if __name__ == "__main__":
    main()
