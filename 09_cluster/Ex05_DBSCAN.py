import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import folium

df = pd.read_excel('../00_dataIn/middle_shcool_graduates_report.xlsx')
print(df.head())

dummy_list = ['지역', '코드', '유형', '주야']
for i in dummy_list:
    print(f'{i}:{df[i].unique()}')

# 범주형 데이터 원-핫 인코딩
df_encoded = pd.get_dummies(df, columns=dummy_list, dtype=int)
print(df_encoded.columns)

train_feature = ['과학고', '외고_국제고', '자사고', '자공고', '유형_공립', '유형_국립', '유형_사립']
x = df_encoded[train_feature]
print(x)

# 표준화
scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x)

# 모델
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(x)

cluster_label = dbscan.labels_
print('cluster_label:\n', cluster_label)

result = np.unique(cluster_label, return_counts=True)
print('result:\n', result)

unique_vals, counts = np.unique(cluster_label, return_counts=True)
print('unique_vals:\n', unique_vals)
print('counts:\n', counts)

# Cluster 컬럼 추가
df_encoded['Cluster'] = cluster_label
print(df_encoded.head())
print(df_encoded['Cluster'].value_counts())

# Cluster 그룹핑
concern = ['학교명', '과학고', '외고_국제고', '자사고']

grouped = df_encoded.groupby('Cluster')
print(len(grouped))

for group_no, group in grouped:
    print('군집번호:', group_no)
    print('군집별 데이터 갯수:', len(group))
    print(group.loc[:,concern])
    print()

# for cluster_num in df_encoded['Cluster'].unique():
#     df_cluster = df.loc[df_encoded['Cluster'] == cluster_num, concern]
#
#     print(f'군집번호: {cluster_num}')
#     print(f'군집별 데이터 갯수: {len(df_cluster)}개')
#     print(df_cluster)
#     print()

colors = {-1: 'gray', 0: 'red', 1: 'green', 2: 'pink', 3: 'lightblue'}
m = folium.Map(location=[37.487841, 127.039141], zoom_start=12)
folium.Marker(location=[37.487841, 127.039141],
              popup="home",
              icon=folium.Icon(color='black', icon='home')).add_to(m)

for cluster_num in df_encoded['Cluster'].unique():
    df_cluster = df.loc[df_encoded['Cluster'] == cluster_num, ['학교명', '위도', '경도']]
    for name, lat, lon in zip(df_cluster['학교명'], df_cluster['위도'], df_cluster['경도']):
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color=colors.get(cluster_num))
        ).add_to(m)
m.save('school.html')

