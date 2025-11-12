import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'Action': [9, 8, 2, 1, 7, 6, 3, 2, 9, 8],
    'Romance': [1, 2, 8, 9, 3, 2, 7, 9, 1, 3],
    'Horror': [8, 7, 2, 1, 7, 6, 3, 2, 8, 7],
    'Comedy': [4, 5, 7, 8, 5, 4, 6, 7, 3, 4],
    'Drama': [2, 3, 9, 8, 4, 3, 8, 9, 2, 3]
})
# print(data)

# 표준화
scaler = StandardScaler()
x = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)
data['Cluster'] = kmeans.labels_
print(data)

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,10))
for i in range(10):
    plt.text(data['Action'][i]+0.1, data['Romance'][i]+0.1,  f"P{i+1}", color='black')
plt.scatter(data['Action'], data['Romance'], c=data['Cluster'], s=100)
plt.xlabel('Action')
plt.ylabel('Romance')
plt.title('K-Means 클러스터링 (Action vs Romance)')
plt.show()

    