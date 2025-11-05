import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

y_test = np.array([0, 0, 1, 1]) # 0:정상, 1:암
y_score = np.array([0.1, 0.4, 0.35, 0.8]) # 암일 확률

y_pred = (y_score >= 0.8).astype(int)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

TN, FP, FN, TP = cm.ravel()
print(TN, FP, FN, TP)

fpr = FP / (FP + TN)
tpr = TP / (TP + FN)
print("FPR:", fpr)
print("TPR:", tpr)

print("----------------------------")
# thresholds: 분류 기준이 되는 점수값들 (높은 점수부터 낮은 점수 순)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
print("FPR:", fpr)
print("TPR:", tpr)
print(thresholds)

# AUC 계산
roc_auc = auc(fpr, tpr)
print("roc_auc:", roc_auc)

# ROC 그래프 시각화
plt.figure(figsize=[6, 6])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 각 점에 threshold 표시
for i,thresh in enumerate(thresholds):
    plt.text(fpr[i], tpr[i], f'{thresh:.2f}', fontsize=10)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()