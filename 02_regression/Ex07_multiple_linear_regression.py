import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'study_time': [2, 3, 4, 5, 6, 8, 10, 12],
    'sleep_time': [9, 8, 8, 7, 6, 6, 5, 5],
    'phone_time': [5, 5, 4, 4, 3, 2, 2, 1],
    'score': [50, 55, 60, 65, 70, 75, 85, 88]
}

df = pd.DataFrame(data)
x = df.drop(columns='score')
y = df['score']
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

print("회귀계수:", model.coef_)
print("절편:", model.intercept_)
print("예측점수:", y_pred)

df['predicted'] = y_pred
print(df)

r2 = model.score(x, y)
print("결정계수:", r2)

RSS = np.sum((y - y_pred) ** 2)
TSS = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - RSS/TSS
# print("결정계수:", r2)

# RMSE(평균 제곱근 오차)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)

x_test = pd.DataFrame(
    [[6,7,2],[4,8,4],[10,5,1]],
    columns=['study_time', 'sleep_time', 'phone_time']
)
y_pred_test = model.predict(x_test)
print(y_pred_test)
