import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 양성 데이터
X_class1 = np.array([[2,2], [3,3]])
# 음성 데이터
X_class2 = np.array([[-2,-2], [-3,-3]])

# 두 클래스 합치기
x = np.vstack([X_class1, X_class2])
y = np.array([0,0,1,1])

# 선형 SVM 모델 학습
model = SVC(kernel='linear')
model.fit(x,y)

# 결정경계(Decision Boundary)의 기울기(계수)와 절편 출력
print("회귀계수:", model.coef_)
print("절편:", model.intercept_)

# 결정경계 계산을 위한 x축 구간 설정
w = model.coef_[0]
b = model.intercept_[0]
x_vals = np.linspace(np.min(x[:,0]-1), np.max(x[:,0]+1),100)
print("x_vals:\n", x_vals)

# 결정경계(Decision Boundary) 계산
decision_boundary = -(w[0] * x_vals + b) / w[1]
print("decision_boundary:\n", decision_boundary)

# 마진(Margin) 선 계산
margin_positive = -(w[0] * x_vals + b-1) / w[1]
margin_negative = -(w[0] * x_vals + b+1) / w[1]
print("margn_positive:\n", margin_positive)
print("margin_negative:\n", margin_negative)

# 결정경계 위의 모든 점 좌표 생성
points = np.vstack([x_vals, decision_boundary]).T
print("points:\n", points)

# 결정 함수(Decision Function) 계산
decision_value = model.decision_function(points)
print("decision_value:\n", decision_value)

# 시각화
plt.figure(figsize=[7,5])

# 클래스별 산점도
plt.scatter(x[y==0,0], x[y==0,1], color='b', label='클래스 0')
plt.scatter(x[y==1,0], x[y==1,1], color='r', label='클래스 1')
# plt.scatter(X_class1[:,0], X_class1[:,1], color="b")
# plt.scatter(X_class2[:,0], X_class2[:,1], color="r")

# 결정경계 및 마진선 추가 (점선)
plt.plot(x_vals, decision_boundary, 'k-', label='결정경계')
plt.plot(x_vals, margin_positive, 'g-.', label="마진 +1")
plt.plot(x_vals, margin_negative, 'g--', label="마진 -1")

# 서포트 벡터 강조
plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=100,
    facecolors='none',
    edgecolors='black',
    label="서포트 벡터"
)

plt.legend()
plt.title("SVM 선형 분류: 결정경계 및 서포트 벡터 시각화")
plt.show()
