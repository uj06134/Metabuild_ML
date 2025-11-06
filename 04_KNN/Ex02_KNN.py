import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([
    [2, 9],
    [1, 5],
    [3, 7],
    [6, 2],
    [7, 3],
    [8, 4],
])

y = np.array([0, 0, 0, 1, 1, 1])
z = np.array([[4,6], [7,2]])

# KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
pred = model.predict(z)
print("pred:", pred)
distances, index = model.kneighbors(z)
print("distances:\n", distances)
print("index:\n", index)

# 그래프
plt.scatter(x[y==0, 0], x[y==0, 1], color = 'r', label = '불합격')
plt.scatter(x[y==1, 0], x[y==1, 1], color = 'b', label = '합격')
plt.scatter(z[:,0], z[:,1], color = 'k')
plt.legend()
plt.grid(True)
plt.show()