# 변수명                    설명
# Channel                고객 유형 (1: Horeca → 호텔/레스토랑/카페, 2: Retail → 소매상)
# Region                지역 (1: Lisbon, 2: Oporto, 3: Other)
# Fresh                    신선 제품 지출 금액
# Milk                    유제품 지출 금액
# Grocery                식료품 지출 금액
# Frozen                냉동식품 지출 금액
# Detergents_Paper        세제 및 종이류 지출 금액
# Delicassen            조미료, 특별식품, 반조리·즉석 식품 지출 금액

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../00_dataIn/Wholesale customers data.csv')
df = df.iloc[:,2:]

# 표준화
scaler = StandardScaler()
x = scaler.fit_transform(df)

# n_clusters=5 → 5개의 그룹으로 나누기
# n_init=10 → 초기 중심점을 10번 바꿔 시도하여 최적 결과 선택
kmeans = KMeans(n_clusters=5, n_init=10, random_state=1234)
kmeans.fit(x)

print(kmeans.labels_)
print(kmeans.cluster_centers_) # 각 군집별 중심좌표

df['Cluster'] = kmeans.labels_
print(df.head(5))
print(df['Cluster'].value_counts())

colors = ['red','blue','yellow','pink','green']

plt.figure(figsize=(10,10))
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Milk'], cluster_data['Grocery'], color=colors[i], label = 'Cluster'+str(i))
plt.xlabel('Milk')
plt.ylabel('Grocery')
plt.xlim(0, 14000)
plt.legend()
plt.show()