import numpy as np
from matplotlib import pyplot as plt

# y = 3*x + 1
# -----------------------------
# 선형회귀(직선) 식의 일반형
#   y = a*x + b
#     a → 기울기(회귀계수, slope)
#     b → 절편(intercept)
# -----------------------------
#   a = 3  → 기울기 (x가 1 증가할 때 y는 3 증가)
#   b = 1  → 절편  (x=0일 때 y값)

x = np.array([1,3])
y = np.array([4,10])

g = (y[1] - y[0]) / (x[1] - x[0])
print("기울기:", g)

x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

# (1,2) (2,4) = 4-2 / 2-1 = 2
slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
print(slopes)

# (-2 * -2) + (-1 * 0) + (0 * -1) + (1 * 0) + (2 * 1) = 6
# (-2)**2 + (-1)**2 + 0**2 + 1**2 + 2**2 = 10

# 평균
mean_x = np.mean(x)
mean_y = np.mean(y)
# 기울기 구하기
# 분자: Σ(x - 평균x)(y - 평균y)
# 분모: Σ(x - 평균x)²
numerator = sum((x - mean_x) * (y - mean_y))
denominator = sum((x - mean_x) ** 2)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

print("기울기(slope):", slope)
print("절편(intercept):", intercept)