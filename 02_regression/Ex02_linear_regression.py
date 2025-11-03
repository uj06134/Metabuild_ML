import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# y = 3*x + 1
# -----------------------------
# 선형회귀(직선) 식의 일반형
# y = a*x + b
# a → 기울기(회귀계수, slope)
# b → 절편(intercept)
# -----------------------------
# a = 3  → 기울기 (x가 1 증가할 때 y는 3 증가)
# b = 1  → 절편  (x=0일 때 y값)

x = np.array([1,3])
y = np.array([4,10])

g = (y[1] - y[0]) / (x[1] - x[0])
print("기울기:", g)

x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

# (1,2) (2,4) = 4-2 / 2-1 = 2
slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
print(slopes)

# (-2 * -2) + (-1 * 0) + (0 * -1) + (1 * 0) + (2 * 1) = 6
# (-2)**2 + (-1)**2 + 0**2 + 1**2 + 2**2 = 10

# 평균
mean_x = np.mean(x)
mean_y = np.mean(y)
# 기울기 구하기
# 분자: Σ(x - 평균x)(y - 평균y)
# 분모: Σ(x - 평균x)²
numerator = sum((x - mean_x) * (y - mean_y))
denominator = sum((x - mean_x) ** 2)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

print("기울기(slope):", slope)
print("절편(intercept):", intercept)

# LinearRegression: 입력 변수(x) 와 출력 변수(y) 사이의 직선 관계(선형 관계) 를 찾아주는 모델
# x가 1차원 배열이라 sklearn이 feature 차원을 인식 못 함
x = x.reshape(-1, 1)   # (5,) → (5,1)
model = LinearRegression()
model.fit(x, y)
y_pred  = model.predict(x)

print("회귀계수(coefficient):", model.coef_[0])
print("절편(intercept):", model.intercept_)
print("예측값:", y_pred )

# 결정계수: 회귀모델이 데이터를 얼마나 잘 설명하는가(예측하는가)를 나타내는 지표
# R² = 1 - (RSS/TSS)
# RSS(잔차제곱합): 모델의 예측값과 실제값의 차이를 제곱해서 모두 더한 값
RSS = np.sum((y - y_pred) ** 2)
# TSS(총제곱합): 실제 데이터가 평균값으로부터 얼마나 떨어져 있는지를 나타내는 값
TSS = np.sum((y - mean_y) ** 2)
R2 = 1 - (RSS / TSS)
print("결정계수(R²):", R2)
R2 = model.score(x, y)
print("결정계수(R²):", R2)
R2 = r2_score(y, y_pred)
print("결정계수(R²):", R2)

r2_score(y, y_pred)


# 그래프 그리기
plt.scatter(x, y, color='b', label='실제 데이터(x, y)')
plt.plot(x, y_pred, label=f'회귀선 y = {model.coef_[0]}x + {model.intercept_}', color='r')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.title("선형 회귀: 데이터와 회귀선")
plt.show()