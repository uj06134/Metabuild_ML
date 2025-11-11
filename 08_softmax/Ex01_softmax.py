import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

x_data = np.random.rand(10,4)
y_labels = np.random.randint(0,3, size=(10,))
print('x_data:\n',x_data)
print('y_labels:',y_labels)

# 원-핫 인코딩
y_data = to_categorical(y_labels, num_classes=3)
print('y_data:\n',y_data)

model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)), # 4가지 정보(input_shape)
    Dense(3, activation='softmax')
])

# 다중분류 -> categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(x_data, y_data, epochs=10, validation_split=0.3, verbose=1)

