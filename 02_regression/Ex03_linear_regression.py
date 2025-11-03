from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

x = np.array([50, 60, 70, 80, 90, 100, 110])        # 집 크기
y = np.array([150, 180, 200, 220, 240, 260, 300])   # 집 가격

x = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
y_pred  = model.predict(x)
r2 = model.score(x, y)

print("기울기:", model.coef_[0])
print("절편:", model.intercept_)
print("결정계수: ", r2)

plt.scatter(x, y, label='데이터(x, y)')
plt.plot(x, y_pred, color='red', label=f'회귀선 y = {model.coef_[0].round(2)}x + {model.intercept_.round(2)}')
plt.xlabel("집 크기")
plt.ylabel("집 가격")
plt.grid(True)
plt.legend()

plt.show()
