import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.datasets import mnist, fashion_mnist
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt

data = fashion_mnist.load_data()
# print(data)

(x_train, y_train), (x_test, y_test) = data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 28x28 이미지를 1차원 벡터로 변환하기 위해 열(column) 개수 계산
x_column = x_train.shape[1] * x_train.shape[2]
print('x_column:', x_column)

# (60000, 28, 28) → (60000, 784) 형태로 변환
# 신경망(Dense layer)은 1차원 입력만 받기 때문에 2차원 이미지를 평탄화해야 함
x_train = x_train.reshape(x_train.shape[0], x_column)
x_test = x_test.reshape(x_test.shape[0], x_column)
print('x_train:', x_train.shape)
print('x_test:', x_test.shape)

# 정규화 (Normalization)
# 픽셀 값은 0~255 사이의 정수이므로, 이를 255로 나누어 0~1 범위의 실수(float32)로 변환
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 원-핫 인코딩
CLASSES = len(class_names)
y_train = to_categorical(y_train, CLASSES)
y_test = to_categorical(y_test, CLASSES)

# 모델
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_column,)))
model.add(Dense(units=CLASSES, activation="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=1)
print('score:', score)

pred = model.predict(x_test)
print('x_test:', x_test[0] )

predict_class = np.argmax(pred[0])
print(f'prediction class : {predict_class}')