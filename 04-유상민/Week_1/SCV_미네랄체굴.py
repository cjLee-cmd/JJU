initial_minerals = 100000  # 초기 미네랄
command_center_minerals = 0  # 커맨드 센터 초기 미네랄
scv_count = 1  # 초기 SCV 수
scv_production_time = 10  # SCV 생산 시간 (초)
scv_cost = 50  # SCV 생산 비용
mineral_to_reach = 50000  # 목표 미네랄
mineral_rate_per_scv = 5  # SCV 당 초당 채굴량

time = 0  # 경과 시간 (초)

while command_center_minerals < mineral_to_reach:
    # 현재 SCV 수로 채굴한 미네랄 계산
    command_center_minerals += scv_count * mineral_rate_per_scv

    # 시간 경과
    time += 1

    # 10초마다 SCV 생산 가능 여부 확인
    if time % scv_production_time == 0 and initial_minerals >= scv_cost:
        scv_count += 1  # SCV 추가
        initial_minerals -= scv_cost  # 미네랄 차감

print(f"커맨드 센터에 미네랄이 50000이 되는데 걸리는 시간: {time}초")



