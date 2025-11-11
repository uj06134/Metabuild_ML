import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-10, 10, 5)
print('x:',x)
print()

# 0~1 사이 확률 형태로 변환하는 S자형 함수
sigmoid = 1 / (1+np.exp(-x))
print('sigmoid:',sigmoid)


# relu: 음수일때는 0으로, 양수일때는 그대로 나옴
relu = np.maximum(0, x)
print('relu:',relu)
plt.figure(figsize=[8,5])
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid, color='r')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(x, relu, color='b')
plt.grid(True)
plt.show()

