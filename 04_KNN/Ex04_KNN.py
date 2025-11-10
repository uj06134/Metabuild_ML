import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

n = 100
normal = np.random.normal([1,1],[1.0,1.0], (n//2,2)) # (50, 2)
# print(normal)

spam = np.random.normal([2.5,2.5],[1.0,1.0], (n//2,2)) # (50, 2)
# print(spam)

x = np.vstack((normal, spam))
y = np.array([0]*50 + [1]*50)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("실제값:",y_test)
print("예측값:",y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

new_emails = np.array([
    [1.2, 0.8],
    [3.5, 2.9],
    [2.0, 2.0]
])
y_pred2 = model.predict(new_emails)
print(y_pred2)

# 시각화
plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x_test[y_test==0, 0], x_test[y_test==0, 1], color = 'b', label = '정상메일(실제 0)')
plt.scatter(x_test[y_test==1, 0], x_test[y_test==1, 1], color = 'r', label = '스팸메일(실제 1)')
error = (y_test != y_pred)
plt.scatter(x_test[error,0], x_test[error,1], s=100, facecolors='none', edgecolors='k', label='오분류')
for i in range(len(y_pred2)):
    if y_pred2[i] == 0:
        color = 'g'
        label = f'새 이메일 {i + 1} (예측: 정상)'
    else:
        color = 'y'
        label = f'새 이메일 {i + 1} (예측: 스팸)'
    plt.scatter(new_emails[i,0], new_emails[i,1], color=color, marker='*', label=label)
plt.legend()
plt.grid(True)
plt.xlabel("광고 단어 개수")
plt.ylabel("링크 개수")
plt.title("KNN으로 스팸메일 분류")
plt.show()