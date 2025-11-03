import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../00_data/서울시 부동산 실거래가 정보.csv", encoding="UTF-8")
print(df.head(5))
print(df.columns)
df.dropna(inplace=True)
x = df[["건물면적(㎡)", "층", "건축년도"]]
y = df["물건금액(만원)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

a1, a2, a3 = model.coef_
b = model.intercept_
print("회귀계수:", model.coef_)
print("절편:", b)
print(f"방정식: y = {a1:.2f} * 건물면적 + {a2:.2f} * 층 + ({a3:.2f}) * 건축년도 + ({b:.2f})")

y_pred = model.predict(x_test)
print("예측값",y_pred)
r2 = model.score(x_test, y_test)
print("결정계수:", r2)

x2 = pd.DataFrame([[84, 5, 2010]], columns=["건물면적(㎡)", "층", "건축년도"])
y_pred2 = model.predict(x2)
print("예측금액(만원):", y_pred2[0])
