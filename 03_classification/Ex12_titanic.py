import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# survived: 생존 여부 (0 = 사망, 1 = 생존)
# pclass: 티켓 등급(1 = 1등급, 2 = 2등급, 3 = 3등급)
# gender: 성별 (male / female)
# age: 나이 (float), 일부 결측값 존재
# sibsp: 함께 탑승한 형제자매 또는 배우자 수
# parch: 함께 탑승한 부모 또는 자녀 수
# fare: 탑승 요금 (float)
# embarked: 탑승 항구 코드 (C = Cherbourg(셰르부르, 셸부르), Q = Queenstown, S = Southampton)
# class: pclass와 동일한 정보지만 범주형(categorical) 타입 (First, Second, Third)
# who: 사람 유형 (man, woman, child)
# adult_male: 성인 남성 여부 (True/False)
# deck: 탑승객이 머무른 데크 (캐빈 위치); 대부분 결측값이 있음
# embark_town: 탑승 항구의 도시명 (Southampton, Cherbourg, Queenstown)
# alive: 생존 여부를 문자열로 표현 (yes / no)
# alone: 혼자 여행 여부 (True = 혼자, False = 동반자 있음)
df = sns.load_dataset('titanic')
print(df)

df.to_csv('../00_dataOut/titanic.csv', index=False)

# 컬럼 이름 변경
rdf = df.rename(columns={'sex':'gender'})
print(rdf.columns)

# 중복개수
dup_count = sum(rdf.duplicated())
print(dup_count)

# 첫행은 그대로 두고 나머지 행은 모두 삭제
rdf = rdf.drop_duplicates()
print(rdf)

# deck, embark_town 컬럼 삭제
rdf = rdf.drop(columns=['deck', 'embark_town'])
print(rdf.columns)
print(rdf.shape)

# age 컬럼 결측치 제거
rdf = rdf.dropna(subset=['age'])
# print(rdf.isnull().sum())

# 최빈값
rdf_mode = rdf['embarked'].mode()[0]
print(rdf_mode)

# embarked 컬럼 결측치 최빈값으로 채우기
rdf['embarked'] = rdf['embarked'].fillna(rdf_mode)
print(rdf.isnull().sum())
print(rdf['embarked'].value_counts())

# 필요한 컬럼만 추출
ndf = rdf[['survived', 'pclass', 'gender', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf)

# 원-핫 인코딩
# 범주형(문자형) -> 숫자형
gender_encode = pd.get_dummies(ndf['gender']).astype(int)
print(gender_encode)

embarked_encode = pd.get_dummies(ndf['embarked'], dtype="int", prefix='town')
print(embarked_encode)

pd.set_option('display.max_columns', None)
# 인코딩 데이터 합치기
ndf = pd.concat([ndf, gender_encode, embarked_encode], axis=1)

# 기존 컬럼 삭제
ndf = ndf.drop(columns=['gender', 'embarked'])
print(ndf)


# x: 독립변수(pclass, age ... town_S)
# y: 종속변수(survived)
x = ndf.drop(columns=['survived'])
y = ndf['survived']

# 데이터 표준화
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 모델 훈련
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

# 테스트 데이터 생존 예측
y_pred = model.predict(x_test)
print(y_pred)

# 혼동행렬
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
print(TN, FP, FN, TP)
# TPR = TP / (TP + FN)
# FPR = FP / (FP + TN)
print(cm)

# 한글 처리
plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 히트맵
cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'],
                     columns=['Negative', 'Positive'])

sns.heatmap(cm_df, annot=True, cmap='gray', linewidths=2, fmt='d')
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('혼동행렬 히트맵')
plt.show()

# 각 클래스(0 또는 1)에 대한 “예측 확률”을 반환
pred_proba = model.predict_proba(x_test)
y_score = pred_proba[:, 1]
# y_score = model.decision_function(x_test)

# 분류기의 임계값(threshold)을 다양하게 바꿔가며,
# 그때마다의 FPR(위양성률)과 TPR(재현율)을 계산해서 반환
fpr,tpr,thresholds = roc_curve(y_test,y_score)

# AUC 계산
auc_value = auc(fpr, tpr)

# 시각화
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={auc_value:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)

for i, thresh in enumerate(thresholds):
    if i % 20==0 :
        plt.text(fpr[i], tpr[i], f'{thresh:.2f}', fontsize=10)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
