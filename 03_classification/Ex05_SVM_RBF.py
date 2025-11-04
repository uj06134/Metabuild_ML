import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 시험 점수
np.random.seed(111)

# 합격 데이터
X_pass = np.random.randn(20,2) * 5 + [70, 75]
# print("X_pass:\n", X_pass)

# 불합격 데이터
X_fail = np.random.randn(20,2) * 5 + [50, 55]
# print("X_fail:\n", X_fail)

# 두 클래스 합치기
x = np.vstack([X_pass,X_fail])
y = np.array([1]*20 + [0]*20)

# 훈련데이터, 검증데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42, stratify=y)

# 비선형(곡선) SVM 모델 학습
model = SVC(kernel='rbf')
model.fit(x_train,y_train)

# 그래프
plt.figure(figsize=(8,6))
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], color='blue', label='합격')
plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], color='red', label='불합격')
plt.legend()


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], num=100)
yy = np.linspace(ylim[0], ylim[1], num=100)
XX, YY = np.meshgrid(xx, yy)

xy = np.stack((XX.ravel(), YY.ravel())).T

z = model.decision_function(xy).reshape(XX.shape)
plt.contour(XX, YY, z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '-.'])
plt.contourf(XX, YY, z, levels=[z.min(),0,z.max()], colors=['pink','skyblue'], linestyles=['--', '-', '-.'], alpha=0.5)
plt.title("SVM 비선형(곡선) 분류: 결정경계 및 서포트 벡터 시각화")
plt.show()


y_pred = model.predict(x_test)
for x, y in zip(x_test, y_pred):
    result = '합격' if y else '불합격'
    print(f"시험점수{x} -> 예측: {result}")
