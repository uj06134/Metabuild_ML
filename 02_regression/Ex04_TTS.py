import numpy as np
from sklearn.model_selection import train_test_split

x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("X_train: ", x_train)
print("X_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)

