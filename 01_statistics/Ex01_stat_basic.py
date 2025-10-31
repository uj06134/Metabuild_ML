import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 모집단(Population): 연구 대상 전체 (예: 전국 학생의 키)
# 표본집단(Sample): 모집단 중 일부만 추출한 집단 (예: 서울 학생 100명)

x = [3, 5, 8, 11, 13, 8]
y = [1, 2, 3, 4, 5, 3]

count = len(x)
print(count)

avg_x = sum(x)/count
avg_y = sum(y)/count

print("avg_x:", avg_x)
print("avg_y:", avg_y)

print("mean_x:", np.mean(x))
print("mean_y:", np.mean(y))

x_y = [(xi, yi) for xi, yi in zip(x, y)]
print("x_y:", x_y)

# 분산(Variance):
#  각 데이터가 평균으로부터 얼마나 떨어져 있는지를 제곱해 평균낸 값
#  - 모집단 분산: σ² = Σ(x - μ)² / N
#  - 표본 분산: s² = Σ(x - ȳ)² / (n - 1)
var_x = sum((xi - avg_x)**2 for xi in x) / count
var_y = sum((yi - avg_y)**2 for yi in y) / (count-1)
print("var_x:", var_x)
print("var_y:", var_y)

var_x2 = np.var(x,ddof=0) # ddof: 표본 분산 / 표준편차 계산 시 나눌 때 얼마를 빼줄지 조정하는 역할
var_y2 = np.var(y,ddof=1)
print("var_x2:", var_x2)
print("var_y2:", var_y2)

# 표준편차(Standard Deviation):
#  분산의 제곱근
#  - 모집단 표준편차: σ = √(Σ(x - μ)² / N)
#  - 표본 표준편차: s = √(Σ(x - ȳ)² / (n - 1))
std_x = np.sqrt(var_x)
std_y = np.sqrt(var_y)
print("std_x:", std_x)
print("std_y:", std_y)

std_x2 = np.std(x,ddof=0)
std_y2 = np.std(y,ddof=1)
print("std_x2:", std_x2)
print("std_y2:", std_y2)

# 공분산(Covariance): 두 변수 간의 함께 변하는 정도를 수치로 나타낸 지표
#  - 모집단 공분산: σ_xy = Σ((x - μ_x)(y - μ_y)) / N
#  - 표본 공분산: s_xy = Σ((x - ȳ_x)(y - ȳ_y)) / (n - 1)
cov_xy = sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x, y)) / count
print("cov_xy:", cov_xy)

cov_xy2 = np.cov(x, y, ddof=0)[0,1]
print("cov_xy2:", cov_xy2)

# 상관계수(Correlation Coefficient): 두 변수가 얼마나 강하게, 그리고 어떤 방향으로 함께 변하는지를 −1과 +1 사이의 숫자로 표현한 값
#  - 모집단 상관계수: ρ = Σ((x - μ_x)(y - μ_y)) / √(Σ(x - μ_x)² × Σ(y - μ_y)²)
#  - 표본 상관계수: r = Σ((x - ȳ_x)(y - ȳ_y)) / √(Σ(x - ȳ_x)² × Σ(y - ȳ_y)²)
# 상관계수 = 공분산/ (x표준편차 * y표준편차)
std_x = np.std(x, ddof=0)
std_y = np.std(y, ddof=0)
corr = cov_xy / (std_x * std_y)
print("corr:", corr)

corr2 = np.corrcoef(x,y)[0,1]
print("corr2:", corr2)

# 통계
age = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
income = [2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000]
n = len(age)

avg_age = sum([a for a in age]) / n
print("avg_age:", avg_age)

avg_income = sum([i for i in income]) / n
print("avg_income:", avg_income)

var_age = sum([(a - avg_age)**2 for a in age]) / n
print("var_age:", var_age)

var_income = sum([(i - avg_income)**2 for i in income]) / n
print("var_income:", var_income)

std_age = var_age ** 0.5
print("std_age:", std_age)

std_income = var_income ** 0.5
print("std_income:", std_income)

cov = sum((a - avg_age) * (i - avg_income) for a, i in zip(age, income)) / n
cov2 = np.cov(age, income, ddof=0)[0, 1]
print("cov:", cov)
print("cov2:", cov2)

corr = cov / (std_age * std_income)
corr2 = np.corrcoef(age, income)[0, 1]
print("corr:", corr)
print("corr2:", corr2)

# DataFrame
df = pd.DataFrame({"Age":age, "Income":income})
print(df)

corr = df.corr()

plt.figure(figsize=(8,6))

# 히트맵
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Age-Income Correlation Heatmap")
plt.show()