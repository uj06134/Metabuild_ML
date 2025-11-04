import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
print("y:\n", y)

# 훈련데이터, 검증데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42, stratify=y)

# 선형 SVM 모델 학습
model = SVC(kernel='linear')
model.fit(x_train,y_train)

# 회귀계수, 절편
w = model.coef_[0]
b = model.intercept_[0]
x_vals = np.linspace(np.min(x[:,0]-1), np.max(x[:,0]+1),100)

# 결정경계
decision_boundary = -(w[0] * x_vals + b) / w[1]
# 마진 + 1
margin_positive = -(w[0] * x_vals+ b-1) / w[1]
# 마진 - 1
margin_negative = -(w[0] * x_vals+ b+1) / w[1]

# 그래프
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], color='b', label='합격')
plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], color='r', label='불합격')
plt.plot(x_vals, decision_boundary, 'k-', label='결정경계')
plt.plot(x_vals, margin_positive,'k--' ,label='마진+1')
plt.plot(x_vals, margin_negative,'k-.' ,label='마진-1')

plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=100,
    facecolors='none',
    edgecolors='k',
    label="서포트 벡터"
)
plt.show()

# 검증데이터
y_pred = model.predict(x_test)
for x, y in zip(x_test, y_pred):
    result = '합격' if y else '불합격'
    print('시험 점수 {} -> 예측: {}'.format(x, result))

# 정확도
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 추가 데이터 검증
x_new = np.array([[80,70],[60,65],[45,50],[72,78]])
y_pred2 = model.predict(x_new)
for x, y in zip(x_new, y_pred2):
    result2 = '합격' if y else '불합격'
    print(f"시험점수{x} -> 예측: {result2}")
