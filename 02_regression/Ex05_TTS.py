from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

x = np.array([50, 60, 70, 80, 90, 100, 110])        # 집 크기
y = np.array([150, 180, 200, 220, 240, 260, 300])   # 집 가격
x = x.reshape(-1, 1)

# 학습용, 테스트용 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
r2 = model.score(x_test, y_test)
print("결정계수(R²):", r2)

# 그래프
plt.scatter(x_train, y_train, label='학습 데이터', color='blue')
plt.scatter(x_test, y_test, label='테스트 데이터', color='green')
plt.plot(x_test, y_pred, color='red', label=f'회귀선 y = {model.coef_[0].round(2)}x + {model.intercept_.round(2)}')
plt.xlabel("집 크기")
plt.ylabel("집 가격")
plt.grid(True)
plt.legend()

plt.show()