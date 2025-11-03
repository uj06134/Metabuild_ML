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

# 경계를 정의하는 핵심 점(서포트 벡터) 출력
print("model.support_vectors:\n",model.support_vectors_)

# 클래스별 산점도
plt.scatter(x[:,0], x[:,1], cmap="coolwarm", c=y)

# 서포트 벡터 강조 표시
plt.scatter(
    model.support_vectors_[:,0],    # x좌표
    model.support_vectors_[:,1],    # y좌표
    s=150,                          # 점 크기
    facecolors='none',              # 내부는 비움 (투명)
    edgecolors='k',                 # 테두리 색: 검정
    label="support vector"          # 범례 이름
)

# 결정경계: 서로 다른 클래스(분류 대상)를 구분하는 ‘경계선(혹은 경계면)’
ax = plt.gca()            # 현재 그래프의 축(Axis) 객체 가져오기
xlim = ax.get_xlim()      # x축 범위
ylim = ax.get_ylim()      # y축 범위

# np.linspace : 구간을 3등분하여 좌표 생성 (예: [-5,0,5])
xx = np.linspace(xlim[0], xlim[1], 3)
yy = np.linspace(ylim[0], ylim[1], 3)

# meshgrid : 위 두 배열로 2차원 평면 좌표 생성
XX, YY = np.meshgrid(xx, yy)

# 격자 좌표를 (N, 2) 형태로 합치기 → [x좌표, y좌표] 쌍
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# 격자점들을 노란색 점으로 표시 (결정함수 평가용 위치)
plt.scatter(xy[:,0], xy[:,1], color="yellow", edgecolors="black", s=100)

# 결정함수 계산
# z = 각 좌표점(xy)이 결정경계로부터 얼마나 떨어져 있는가
z = model.decision_function(xy).reshape(XX.shape)
ax.contour(
    XX, YY, z,
    colors="black",
    levels=[-1, 0, 1],
    linestyles=["--","-","-."],
    alpha=0.5)
plt.legend()
plt.title("SVM 선형 분류: 결정경계 및 서포트 벡터 시각화")
plt.show()