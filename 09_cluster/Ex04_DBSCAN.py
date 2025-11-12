import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

x = np.array([
    [1,2], [2,2], [2,3], # 군집1
    [8,7], [8,8], [7,8], # 군집2
    [20,30] # 노이즈(잡음)
])

# eps: 한 점을 중심으로 이웃으로 간주할 반경 크기
eps_values = [0.5, 1.5, 3.0] # 반경

plt.figure(figsize=(15,5))
for i, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=2)
    labels = dbscan.fit_predict(x)

    # 군집이 아닌 노이즈(-1) 분리
    mask_noise = labels == -1
    mask_cluster = labels != -1

    plt.subplot(1,3,i+1)
    # 군집 점들
    plt.scatter(x[mask_cluster,0], x[mask_cluster,1],
                c=labels[mask_cluster], cmap='rainbow', s=100, label='Clusters')
    # 노이즈
    plt.scatter(x[mask_noise,0], x[mask_noise,1],
                c='k', marker='x', s=100, label='Noise')

    plt.title(f'eps={eps}')
    plt.legend()

plt.show()