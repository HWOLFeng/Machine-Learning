import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():
    # x_train 是数据，y_train 是标签
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[:number]
    y_train = y_train[:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    # x_train = x_train
    # x_test = x_test
    # normalize
    # x_test = np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
# model.add(Dense(input_dim=28*28, units=633, activation='sigmoid'))
model.add(Dense(input_dim=28*28, units=633, activation='relu'))
# add Dropout after each hidden layer
model.add(Dropout(0.7))
# model.add(Dense(units=633, activation='sigmoid'))
model.add(Dense(units=633, activation='relu'))
model.add(Dropout(0.7))
# model.add(Dense(units=633, activation='sigmoid'))
model.add(Dense(units=633, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=10, activation='softmax'))
# mse : mean square error
# model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1000, epochs=20)
result = model.evaluate(x_test, y_test)

print('Test Acc', result[1])
