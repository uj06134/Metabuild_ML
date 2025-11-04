import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

x_approved = np.random.randn(20,2) * 5 + [70, 750]
x_denied = np.random.randn(20,2) * 5 + [50, 600]
# print("x_approved:\n", x_approved)
# print("x_denied:\n", x_denied)

x = np.vstack([x_approved,x_denied])
y = np.array([1]*20 + [0]*20)

# 훈련,검증 데이터 분리
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 모델 훈련
model = SVC(kernel='rbf')
model.fit(x_train,y_train)

# 그래프
plt.scatter(x_train[y_train==0][:, 0], x_train[y_train==0][:, 1], c='b',s=60, edgecolors='k',label='거절')
plt.scatter(x_train[y_train==1][:, 0], x_train[y_train==1][:, 1], c='r',s=60, edgecolors='k',label='승인')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], num=100)
yy = np.linspace(ylim[0], ylim[1], num=100)
XX, YY = np.meshgrid(xx, yy)
xy = np.stack((XX.ravel(), YY.ravel())).T

z = model.decision_function(xy).reshape(XX.shape)
plt.contour(XX, YY, z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '-.'])
plt.legend()
plt.title("급여와 신용점수 기반 비선형 SVM 대출 분류 결과")
plt.show()

y_pred = model.predict(x_test)
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        result = "대출 가능"
    else:
        result = "대출 불가능"
    print("[급여 신용점수]", x_test[i], result)
