import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x = np.array([
    [1,2],
    [2,3],
    [2,2.5],
    [3,4],
    [4,5], # 여기까지 사과
    [7,5],
    [8,8],
    [9,7] # 바나나
])
y = np.array([0,0,0,0,0,1,1,1])
new_point = np.array([[6,5]])

k_values = [1,3,5]

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize = (15,5))
for i, kv in enumerate(k_values):
    # 모델 훈련 및 예측
    model = KNeighborsClassifier(n_neighbors=kv)
    model.fit(x, y)
    y_pred = model.predict(new_point)

    # X, Y 범위 설정
    x_min, x_max = x[:, 0].min(), x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min(), x[:, 1].max() + 1

    # X, Y를 0.1 단위로 나눈 격자 생성
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    xy = np.vstack((xx.ravel(), yy.ravel())).T # 격자 좌표쌍(xy) 생성 → shape: (N, 2)

    # 각 좌표쌍에 대해 클래스 예측 (0 or 1)
    z = model.predict(xy)
    z = z.reshape(xx.shape) # 격자 형태로 다시 변환

    # 시각화
    plt.subplot(1, 3, i + 1)

    # 결정 경계
    plt.contourf(xx, yy, z, colors=['r', 'y'], levels=[0, 0.5, 1], alpha=0.2)

    # 산점도
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='r', s=100, edgecolors='k', label='사과')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='y', s=100, edgecolors='k', label='바나나')
    color = 'g' if y_pred == 0 else 'y'
    plt.scatter(new_point[:,0], new_point[:,1], c=f'{color}', s=100, marker='*', edgecolors='k', label='새과일')

    plt.title(f'사과/바나나 예측 (K={kv})')
    plt.xlabel('색상')
    plt.ylabel('무게')
    plt.legend()
plt.show()