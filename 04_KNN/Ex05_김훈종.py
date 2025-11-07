# 100개의 데이터를 난수로 발생
# 거래시간
# 정상거래 10~17시, 10만원
# 이상거래 12~7시, 20만원

# 30% 테스트
# 실제와 예측이 다르면 오탐지 테두리 검은색
# 5개 이웃
# 정상거래 초록별
# 이상거래 노란색

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

n = 100
np.random.seed(42)

# 정상거래 10시~17시 , 10만원
normal_transactions = np.random.normal([13.5, 10], [2.0, 2.0], (n//2, 2))

# 이상거래 12시~7시, 20만원
abnormal_transactions = np.random.normal([3.5, 20], [2.0, 2.0], (n//2, 2))

# 데이터 합치기
x = np.vstack((normal_transactions, abnormal_transactions))
y = np.array([0]*(n//2) + [1]*(n//2))

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 모델 훈련 및 예측
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(x_train)
new_transactions = np.array([
    [14, 8],
    [3, 28],
    [9, 18],
])