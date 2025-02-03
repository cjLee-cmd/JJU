import threading
import time

class Mineral:
    def __init__(self, amount=100000):
        self.amount = amount
        self.lock = threading.Lock()  # 동기화를 위한 Lock

    def mine(self, quantity):
        """미네랄 채굴"""
        
        with self.lock:  # 동기화
            if self.amount >= quantity:
                self.amount -= quantity
                print(f"[미네랄 필드] {quantity} 미네랄 채굴. 남은 미네랄: {self.amount}")
                return quantity
            else:
                mined = self.amount
                self.amount = 0
                print(f"[미네랄 필드] {mined} 미네랄 채굴. 남은 미네랄: {self.amount}")
                return mined

    def __str__(self):
        return f"미네랄: {self.amount} 남음"


class SCV:
    def __init__(self, id, speed=1):
        self.id = id
        self.carrying = 0
        self.speed = speed  # m/s
        self.power = 1
        self.defend = 10
        self.stamina = 1000

    def harvest(self, mineral, quantity=10):
        """미네랄을 채굴"""
        self.carrying += mineral.mine(quantity)
        print(f"[SCV {self.id}] {self.carrying} 미네랄 채굴 완료.")

    def deliver(self, command_center):
        """커맨드 센터로 미네랄 전달"""
        command_center.store(self.carrying)
        print(f"[SCV {self.id}] {self.carrying} 미네랄 커맨드 센터에 전달 완료.")
        self.carrying = 0

    def travel_time(self, distance):
        """거리 이동 시간 계산"""
        travel_time = distance / self.speed
        print(f"[SCV {self.id}] {distance}m 이동 중. 소요 시간: {travel_time}초.")
        return travel_time


class CommandCenter:
    def __init__(self):
        self.total_minerals = 0
        self.lock = threading.Lock()  # 동기화를 위한 Lock
        self.scv_list = []

    def store(self, amount):
        """미네랄 저장"""
        with self.lock:  # 동기화
            self.total_minerals += amount
            print(f"[커맨드 센터] {amount} 미네랄 저장 완료. 총 미네랄: {self.total_minerals}")

    def produce_scv(self):
        """SCV 생산"""
        with self.lock:
            if self.total_minerals >= 50:
                self.total_minerals -= 50
                new_scv = SCV(id=len(self.scv_list) + 1)
                self.scv_list.append(new_scv)
                print(f"[커맨드 센터] SCV {new_scv.id} 생산 완료! 총 SCV 수: {len(self.scv_list)}")
            else:
                print("[커맨드 센터] 미네랄이 부족합니다. SCV 생산을 위한 미네랄이 부족합니다.")