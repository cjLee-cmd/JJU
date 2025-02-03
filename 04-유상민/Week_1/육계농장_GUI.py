import sys
import pandas as pd
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QComboBox, QWidget, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


def preprocess_data(df, jeouls='all'):
    """
    데이터프레임을 전처리하고, 선택한 jeouls 데이터를 필터링하여 반환
    """
    # 열 이름 확인 및 처리
    df.columns = df.columns.str.strip()  # 공백 제거
    df.columns = df.columns.str.lower()  # 소문자로 변환

    if "gettime" not in df.columns or "jeoulid" not in df.columns:
        raise ValueError("데이터셋에 'getTime' 또는 'jeoulID' 열이 없습니다.")

    # getTime 열 변환
    df['gettime'] = pd.to_datetime(df['gettime'], errors='coerce')
    df = df.dropna(subset=['gettime'])  # 유효하지 않은 날짜 제거

    # jeoulID 필터링
    if jeouls != 'all':
        jeoul_num = int(jeouls[1:])  # 'j01' -> 1
        filtered_df = df[df['jeoulid'] == jeoul_num]
    else:
        filtered_df = df  # 모든 데이터를 반환

    return filtered_df


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("육계농장 시뮬레이션 소프트웨어")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 파일 업로드 버튼
        self.upload_button = QPushButton("CSV 파일 업로드")
        self.upload_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.upload_button)

        # 분석 대상 선택
        self.label = QLabel("분석 대상 선택:")
        self.layout.addWidget(self.label)
        self.combo_box = QComboBox()
        self.combo_box.addItems(['j01', 'j02', 'j03', 'all'])
        self.layout.addWidget(self.combo_box)

        # 결과 출력 버튼
        self.process_button = QPushButton("데이터 처리 및 시각화")
        self.process_button.clicked.connect(self.process_data)
        self.layout.addWidget(self.process_button)

        # 테이블 위젯
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        # 그래프 캔버스
        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)

        self.df = None

    def load_csv(self):
        """
        CSV 파일을 읽고, 데이터를 테이블에 표시
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df.columns = self.df.columns.str.strip()  # 열 이름 공백 제거
                self.df.columns = self.df.columns.str.lower()  # 열 이름 소문자 변환
                self.show_table(self.df)
            except Exception as e:
                self.show_error(f"파일을 읽는 중 오류 발생: {e}")

    def show_table(self, df):
        """
        데이터프레임을 테이블 위젯에 표시
        """
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def process_data(self):
        """
        데이터 필터링 및 전처리, 시각화 실행
        """
        if self.df is not None:
            try:
                jeouls = self.combo_box.currentText()
                processed_df = preprocess_data(self.df, jeouls=jeouls)
                self.show_table(processed_df)
                self.plot_data(processed_df, jeouls)
            except Exception as e:
                self.show_error(f"데이터 처리 중 오류 발생: {e}")
        else:
            self.show_error("먼저 CSV 파일을 업로드하세요.")

    def plot_data(self, df, jeouls):
        """
        선택된 데이터의 산점도 시각화 (all 포함)
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        if jeouls == 'all':
            # 'w01'부터 'w60'까지 모든 열을 시각화
            weight_columns = [col for col in df.columns if col.startswith('w')]
            for col in weight_columns:
                ax.scatter(df['gettime'], df[col], s=5, alpha=0.5, label=col)
            ax.set_title('All Data Weight Progression')
        else:
            if 'w01' in df.columns:
                ax.scatter(df['gettime'], df['w01'], s=5, alpha=0.5)
                ax.set_title(f'{jeouls} Weight Progression')

        ax.set_xlabel('Time')
        ax.set_ylabel('Weight (g)')
        ax.legend()
        self.canvas.draw()

    def show_error(self, message):
        """
        오류 메시지 출력
        """
        error_dialog = QLabel(message, self)
        error_dialog.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(error_dialog)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
