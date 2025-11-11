from keras.src.datasets import mnist
from matplotlib import pyplot as plt

# https://observablehq.com/@davidalber/mnist-browser
data = mnist.load_data()
# print(data)

(x_train, y_train), (x_test, y_test) = data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train[0], y_train[0])

nrow, ncol = 2, 3
fig, axes = plt.subplots(nrow, ncol, figsize=(10,6))

for idx in range(nrow * ncol): # idx: 0~5
    ax = axes[idx//ncol, idx%ncol]
    ax.imshow(x_train[idx])
    ax.axis('off')
    ax.set_title(f'label: {y_train[idx]}')
plt.show()