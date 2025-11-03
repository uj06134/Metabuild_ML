import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# 공부시간, 수면시간
x = np.array([
    [0, 1],
    [5, 1],
    [15, 2],
    [25, 5],
    [35, 11],
    [45, 15],
    [55, 34],
    [60, 35]
])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

print("회귀계수(coefficient):", model.coef_)
print("절편(intercept):", model.intercept_)
print("예측값:", y_pred)

a1, a2 = model.coef_
b = model.intercept_
print(f"방정식: y = {a1:.2f} * 공부시간 + {a2:.2f} * 수면시간 + {b:.2f}")

# 결정계수
RSS = np.sum((y - y_pred) ** 2)
TSS = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - RSS/TSS
print("결정계수(R²):", r2)
r2 = r2_score(y, y_pred)
print("결정계수(R²):", r2)
r2 = model.score(x, y)
print("결정계수(R²):", r2)

x_test = [[6,5],[8,6],[5,7]]
y_pred_test = model.predict(x_test)
print(y_pred_test)