import threading
import time
import matplotlib.pyplot as plt
from starcraft import SCV, CommandCenter, Mineral

def simulate_mining(mineral_field, command_center, mineral_quantity, mineral_distance, time_step=1):
    # total_time을 스레드 안전하게 관리하기 위한 Lock
    time_lock = threading.Lock()
    total_time = 0

    # 히스토리 기록용 리스트
    scv_count_history = []
    mineral_depletion_history = []
    total_minerals_history = []

    # 실제 생성한 작업 스레드 관리용 리스트 (join을 위해)
    worker_threads = []

    # SCV별 작업 함수
    def scv_work(scv):
        nonlocal total_time
        while command_center.total_minerals < mineral_quantity and mineral_field.amount > 0:
            # 광물까지 이동
            with time_lock:
                total_time += scv.travel_time(mineral_distance) * time_step
            
            scv.harvest(mineral_field)
            # 채집 대기 시간
            time.sleep(30 * time_step)

            # 커맨드 센터까지 귀환
            with time_lock:
                total_time += scv.travel_time(mineral_distance) * time_step
            
            scv.deliver(command_center)
            # 광물 배달 후 대기 시간
            time.sleep(30 * time_step)

    # 커맨드 센터 작업 함수 (SCV 추가 생산 및 기록)
    def command_center_work():
        while command_center.total_minerals < mineral_quantity and mineral_field.amount > 0:
            # SCV 생산 대기
            time.sleep(10 * time_step)
            command_center.produce_scv()
            
            # 기록
            scv_count_history.append(len(command_center.scv_list))
            mineral_depletion_history.append(mineral_field.amount)
            total_minerals_history.append(command_center.total_minerals)

    # 초기 SCV 4기 생성 (주석과 실제 코드 일치)
    for i in range(4):
        initial_scv = SCV(id=i+1)
        command_center.scv_list.append(initial_scv)
        t = threading.Thread(target=scv_work, args=(initial_scv,))
        t.start()
        worker_threads.append(t)

    # 커맨드 센터 SCV 생산 스레드 시작
    cc_thread = threading.Thread(target=command_center_work)
    cc_thread.start()
    worker_threads.append(cc_thread)

    # 주기적으로 새로 생산된 SCV를 작업 스레드에 할당
    while command_center.total_minerals < mineral_quantity and mineral_field.amount > 0:
        # 커맨드 센터에서 새로운 SCV가 4기 이상 추가되면 그때부터 추가 스레드 시작
        if len(command_center.scv_list) > len(worker_threads):  
            # 새로 추가된 SCV들 스레드 할당
            for idx in range(len(worker_threads), len(command_center.scv_list)):
                new_scv = command_center.scv_list[idx]
                t = threading.Thread(target=scv_work, args=(new_scv,))
                t.start()
                worker_threads.append(t)

        # 주기적 대기
        time.sleep(10 * time_step)

    # 모든 SCV 스레드 종료 대기
    for t in worker_threads:
        t.join()

    return total_time, scv_count_history, mineral_depletion_history, total_minerals_history


# 메인 실행부
if __name__ == "__main__":
    # 초기화
    mineral_field = Mineral(100000)
    command_center = CommandCenter()

    # 파라미터 설정
    mineral_distance = 30
    required_minerals = 50000
    time_step = 0.001

    # 시뮬레이션 실행
    total_time_taken, scv_count_history, mineral_depletion_history, total_minerals_history = simulate_mining(
        mineral_field, command_center, required_minerals, mineral_distance, time_step)

    # 결과 출력
    print(f"총 {required_minerals} 미네랄 채굴에 걸린 시간: {total_time_taken}초")

    # 그래프 그리기
    time_steps = range(len(scv_count_history))

    plt.figure(figsize=(15, 5))

    # SCV 수 변화 그래프
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, scv_count_history, color='b')
    plt.xlabel('Time Step')
    plt.ylabel('Number of SCVs')
    plt.title('SCV Count Over Time')

    # 광물 감소량 그래프
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, mineral_depletion_history, color='r')
    plt.xlabel('Time Step')
    plt.ylabel('Mineral Amount')
    plt.title('Mineral Depletion Over Time')

    # 커맨드 센터 내 총 미네랄 변화 그래프
    plt.subplot(1, 3, 3)
    plt.plot(time_steps, total_minerals_history, color='g')
    plt.xlabel('Time Step')
    plt.ylabel('Total Minerals in Command Center')
    plt.title('Total Minerals Stored in Command Center Over Time')

    plt.tight_layout()
    plt.show()
