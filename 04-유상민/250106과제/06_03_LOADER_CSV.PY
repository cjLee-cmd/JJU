# 상위 데이터 10개 파싱

from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

loader = UnstructuredCSVLoader(file_path = '/content/data/house_price.csv', mode = "elements")
docs = loader.load()

print(docs[0].metadata["text_as_html"][:2213])

# ------------------------------------------------------------

# 선형회귀분석

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로드
# 데이터 파일 경로를 지정하여 CSV 파일을 DataFrame으로 읽음
file_path = '/content/data/house_price.csv'
data = pd.read_csv(file_path)

# 2. 데이터 구조 확인
# 데이터의 기본 정보를 출력하여 열 이름, 데이터 유형, 결측값 유무를 확인
data.info()

# 3. 상관 분석
# 각 변수와 MEDV 간의 상관계수를 계산
correlation_matrix = data.corr()  # 데이터의 상관계수 행렬 계산
correlation_with_medv = correlation_matrix['MEDV'].sort_values(ascending=False)  # MEDV 기준으로 상관계수 정렬

# 상관 관계를 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_medv.index, y=correlation_with_medv.values)
plt.title('Correlation with MEDV (House Price)')  # 그래프 제목
plt.xticks(rotation=45)  # x축 레이블 45도 회전
plt.ylabel('Correlation Coefficient')  # y축 레이블
plt.show()

# CAT.MEDV와 MEDV 제외 후 상위 3개 변수 추출
# 'MEDV'와 'CAT.MEDV'를 제외한 변수들 중에서 상관계수 절댓값 기준 상위 3개 변수 선택
top_features = correlation_with_medv[~correlation_with_medv.index.isin(['MEDV', 'CAT.MEDV'])].abs().sort_values(ascending=False).head(3).index
print("\nTop 3 variables excluding CAT.MEDV and MEDV:\n", top_features)

# 4. 회귀 분석
# 주요 변수 선택
# 앞에서 선택한 상위 3개 변수(`top_features`)를 설명 변수로 설정
X = data[top_features]  # 입력 데이터 (설명 변수)
y = data['MEDV']  # 타겟 데이터 (목표 변수)

# 데이터 분할
# 데이터를 학습용(80%)과 테스트용(20%)으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 회귀 모델 훈련
# 선형 회귀 모델 생성 및 학습 데이터로 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 성능 평가
# 테스트 데이터를 사용해 주택 가격 예측
y_pred = model.predict(X_test)

# 모델 성능 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차 (MSE)
r2 = r2_score(y_test, y_pred)  # 결정 계수 (R²)

# 회귀 계수 확인
# 각 변수의 회귀 계수를 DataFrame 형태로 정리
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})

# 결과 출력
print("\nRegression Coefficients:\n", coefficients)  # 회귀 계수 출력
print("\nMean Squared Error (MSE):", mse)  # MSE 출력
print("R-squared (R2):", r2)  # R² 출력
