#                 예측
#         negative    positive
# 실제(N)   TN          FP
# 실제(P)   FN          TP

# TN : True Negative
# FP : False Positive
# FN : False Negative
# TP : True Positive
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 실제값 (y_test)과 예측값 (y_pred)
y_test = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

# 혼동행렬(Confusion Matrix) 계산
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 혼동행렬에서 각 항목 분리
TN, FP, FN, TP = cm.ravel()
# TN : 실제 0 → 예측 0  (정상인 것을 정상으로 예측)
# FP : 실제 0 → 예측 1  (정상인데 이상으로 예측)
# FN : 실제 1 → 예측 0  (이상인데 정상으로 예측)
# TP : 실제 1 → 예측 1  (이상인 것을 이상으로 예측)

# 정확도(Accuracy): 전체 예측 중 맞춘 비율
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("정확도:", accuracy)

# 정밀도(Precision): Positive(이상)으로 예측한 것 중 실제로 Positive인 비율
precision = TP / (TP + FP)
print("정밀도:", precision)

# 재현율(Recall, 민감도): 실제 Positive 중 모델이 맞게 예측한 비율
recall = TP / (TP + FN)
print("재현율:", recall)

# 특이도(Specificity): 실제 Negative 중 모델이 맞게 예측한 비율
specificity = TN / (TN + FP)
print("특이도:", specificity)

# 위양성률(FPR, False Positive Rate): 실제 Negative인데 Positive로 잘못 예측한 비율
FPR = FP / (FP + TN)
print("FPR:", FPR)

# F1-score: 정밀도와 재현율의 조화평균
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1-score:", f1_score)

# 분류 보고서 출력 (Precision, Recall, F1-score, Support)
print(classification_report(y_test, y_pred))

