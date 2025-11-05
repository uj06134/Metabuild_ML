import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(11062)
y_true = np.random.randint(0, 2, size=100)
y_score = np.concatenate((
    np.random.uniform(0.0, 0.3, size=40),
    np.random.uniform(0.7, 1.0, size=60)
))

# print(y_true)
# print(y_score)

fpr, tpr, thresholds = roc_curve(y_true, y_score)

# ROC 그래프
plt.plot(fpr, tpr, lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# AUC 계산
roc_auc = auc(fpr, tpr)
print("roc_auc:", roc_auc)

