#                 예측
#         negative    positive
# 실제(N)   TN          FP
# 실제(P)   FN          TP

# TN : True Negative
# FP : False Positive
# FN : False Negative
# TP : True Positive
import numpy as np
from sklearn.metrics import confusion_matrix

y_test = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
cm = confusion_matrix(y_test, y_pred)
print(cm)

TN, FP, FN, TP = cm.ravel()
# print(TN, FP, FN, TP)

accuracy = (TP + TN) / (TP + TN + FP + FN)
# accuracy = cm[0][0] + cm[1][1] / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("정확도:", accuracy)

precision = TP / (TP + FP)
# precision = cm[1][1] / (cm[1][1] + cm[0][1])
print("정밀도:", precision)

recall = TP / (TP + FN)
# recall = cm[1][1] / (cm[1][1] + cm[1][0])
print("재현율:", recall)

f1_score = 2 * precision * recall / (precision + recall)
print("F1-score", f1_score)



