import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc

np.random.seed(42)
data = {
    '면적': np.random.normal(85, 20, 200), # m²,
    '방수': np.random.randint(1, 6, 200),
    '욕실수': np.random.randint(1, 3, 200),
    '연식': np.random.randint(0, 30, 200),
    '지역': np.random.choice(['서울', '부산', '대전', '광주'], 200),
    '용도': np.random.choice(['주거', '상업', '공업'], 200),
    '가격': np.random.normal(4.5, 0.8, 200) * 1e5
}
df = pd.DataFrame(data)
# print(df)

# 결측치 넣기
df.loc[np.random.choice(df.index, 10, replace = False),'면적'] = np.nan
df.loc[np.random.choice(df.index, 5, replace = False),'연식'] = np.nan

# 결측치 갯수
print(df.isnull().sum())

# 결측치 행 모두 제거
df = df.dropna()
print(df.isnull().sum())

# 중복 데이터 행 제거
df = df.drop_duplicates()
print(df.shape)

# 가격 중간값 알아내기
price_med = df["가격"].median()
print("가격 중간값:", price_med)

# df['고가여부'] = 가격 > 가격 중간값
df['고가여부'] = (df['가격'] > price_med).astype(int)
print(df['고가여부'])

# 지역, 용도 원핫 인코딩
df = pd.get_dummies(df, columns=['지역', '용도'], dtype=int)
print(df.columns)

# x = 가격, 고가여부 아닌 칼럼이 독립변수
# y = 고가여부 종속변수
x = df[['면적', '방수', '욕실수', '연식', '지역_광주', '지역_대전', '지역_부산', '지역_서울', '용도_공업', '용도_상업', '용도_주거']]
y = df['고가여부']


# 학습데이터/테스트 분리 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 테스트 데이터 예측
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)

# 실제 데이터 confusion_matrix()
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 예측 확률
pred_proba = model.predict_proba(x_test)
y_score = pred_proba[:, 1]
print(y_score)

# 임계값(threshold)을 바꿔가며 FPR(위양성률)과 TPR(재현율)을 계산
fpr,tpr,thresholds = roc_curve(y_test,y_score)

# AUC 계산
auc_value = auc(fpr, tpr)

# 시각화
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()