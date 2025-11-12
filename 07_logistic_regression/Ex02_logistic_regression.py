import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 번호    컬럼명    설명    데이터형    예시
# 1    ALP    Alkaline Phosphatase (알칼리성 인산분해효소) — 간 기능 관련 효소 수치    float    293
# 2    SEX    성별 (1 = 남성, 2 = 여성)    int    1
# 3    ALB    Albumin (알부민) — 혈장 단백질, 간 기능 및 영양 상태 지표    float    3.8
# 4    ALT    Alanine Aminotransferase — 간 효소 (ALT/SGPT)    float    1.0
# 5    AST    Aspartate Aminotransferase — 간 효소 (AST/SGOT)    float    3.2
# 6    BIL    Bilirubin (빌리루빈) — 혈중 황달 수치    float    2.88
# 7    CHE    Cholinesterase — 간에서 생성되는 효소, 간 손상 시 감소    float    4.3
# 8    CHOL    Cholesterol (콜레스테롤) — 혈중 지방 수치    float    3.1
# 9    CREA    Creatinine (크레아티닌) — 신장 기능 지표    float    0.9
# 10    GGT    Gamma-Glutamyl Transferase — 간 손상 및 담도 질환 지표    float    2.7
# 11    PROT    Total Protein (총 단백질) — 혈중 단백질 수치    float    7.1
# 12    UREA    Urea (요소질소) — 신장 기능 지표    float    24.0
# 13    WBC    White Blood Cell Count (백혈구 수) — 면역상태 지표    float    5.4
# 14    RBC    Red Blood Cell Count (적혈구 수) — 혈액 산소 운반 능력    float    4.6
# 15    SGOT    Serum Glutamic-Oxaloacetic Transaminase (AST와 유사, 간 효소)    float    2.1
# 16    AGE    나이 (세)    int    62
# 17    HP    Hospital stay period or operation time — 수술 관련 정보 (병원체류일수 또는 수술횟수 등)    float    0
# 18    CLASS    예측 대상: 수술 후 결과 (0 = 생존, 1 = 사망)    int    0


dataIn = "../00_dataIn/"
filename = dataIn + "surgeryTest.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
print('data:',data)

# 로지스틱 회귀
# 범주(생존/사망)를 분류
# 특정 클래스에 속할 확률 (0~1 사이)
# ex) 생존 확률(0.7)/사망확률(0.3) => 1/0
# 선형결합 결과(직선)를 Sigmoid 함수에 넣어 0~1로 변환
# 방정식
# z = w1x1 + w2x2 + wnxn + b
# 임의의 w,b
# sigmoid함수로 예측 확률 계산 -> 손실계산
# 역전파 (가중치와 편향을 조정하면서 손실이 각 가중치에 얼마나 영향을 줬는지 계산)

x = data[:, 0:data.shape[1] - 1] # 독립변수
y = data[:, data.shape[1] - 1]   # 종속변수
print('x:',x)
print('y:',y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
predict_proba = model.predict_proba(x_test)
print('predict_proba:\n', predict_proba)

y_pred = model.predict(x_test)
print('y_pred:\n', y_pred)
print('y_test:\n', y_test)



