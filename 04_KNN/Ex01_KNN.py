import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([
    [1,2],
    [2,3],
    [1,3],
    [8,8],
    [9,7],
    [8,9]
])

y = np.array([0,0,0,1,1,1])

new_fruit = np.array([[2,2], [9,8]])

# KNN(K-Nearest Neighbors, K-최근접이웃)
# 새로운 데이터가 들어왔을 때, 그 주변의 가장 가까운 K개의 이웃을 보고 다수결로 클래스를 결정하는 알고리즘
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
pred = model.predict(new_fruit)
print("pred:", pred)

# 유클리드 거리 계산 - 사과(2, 2)
# [1,2] 0 → √((2-1)**2 + (2-2)**2) = √(1 + 0) = 1.000
# [2,3] 0 → √((2-2)**2 + (2-3)**2) = √(0 + 1) = 1.000
# [1,3] 0 → √((2-1)**2 + (2-3)**2) = √(1 + 1) = 1.414
# [8,8] 1 → √((2-8)**2 + (2-8)**2) = √(36 + 36) = 8.485
# [9,7] 1 → √((2-9)**2 + (2-7)**2) = √(49 + 25) = 8.602
# [8,9] 1 → √((2-8)**2 + (2-9)**2) = √(36 + 49) = 9.220

# 유클리드 거리 계산 - 바나나(9, 8)
# [1,2] 0 → √((9-1)**2 + (8-2)**2) = √(64 + 36) = 10.000
# [2,3] 0 → √((9-2)**2 + (8-3)**2) = √(49 + 25) = 8.602
# [1,3] 0 → √((9-1)**2 + (8-3)**2) = √(64 + 25) = 9.434
# [8,8] 1 → √((9-8)**2 + (8-8)**2) = √(1 + 0) = 1.000
# [9,7] 1 → √((9-9)**2 + (8-7)**2) = √(0 + 1) = 1.000
# [8,9] 1 → √((9-8)**2 + (8-9)**2) = √(1 + 1) = 1.414

plt.scatter(x[y == 0, 0], x[y == 0, 1], color='r', s=80, label='사과')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='y', s=80, label='바나나')
plt.scatter(new_fruit[:,0], new_fruit[:,1], color='g', s=80, label='새과일')
plt.legend()
plt.title('KNN 예제')
plt.xlabel('색상')
plt.ylabel('무게')
plt.show()