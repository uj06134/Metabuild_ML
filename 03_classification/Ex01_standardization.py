import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'height' : [160,165,170,175,180],
    'weight' : [55,60,65,70,75]
}

df = pd.DataFrame(data)
print(df)

# 표준화
scaler = StandardScaler()
# scaler.fit(df)
# x_scaled = scaler.transform(df)
x_scaled = scaler.fit_transform(df)
print("x_scaled:\n",x_scaled)
print("표준화 평균:", x_scaled.mean())
print("표준화 표준편차:", x_scaled.std())