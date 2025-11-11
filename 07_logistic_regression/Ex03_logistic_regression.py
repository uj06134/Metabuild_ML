import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.src.layers import Dense, Input
from keras import Sequential

dataIn = "../00_data_in/"
filename = dataIn + "surgeryTest.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)
print('data:',data)

total_col = data.shape[1]
y_column = 1
x_column = total_col -  y_column
x = data[:, 0:x_column]  # 독립변수
y = data[:, x_column:]   # 종속변수

epochs = 30
batch_size = 10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


model = Sequential([
    Input(shape=(x_column,)), # 입력층
    Dense(30, activation="relu"), # 은닉층
    Dense(y_column, activation="sigmoid"), # 출력층
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Adam: 방향(모멘텀)과 속도조절(RMSprop)을 동시에 사용
# Momentum: 이전 단계의 기울기(gradient) 방향을 “관성”처럼 일정 부분 유지
# RMSprop: 각 파라미터의 기울기 크기를 제곱해 평균내어 학습률을 조정

# 학습
fit_hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
print('fit_hist:\n',fit_hist.history)

pred = model.predict(x_test)
print('pred:\n',pred)

pred_label = (pred >= 0.5).astype(int)
print('pred_label:\n',pred_label)