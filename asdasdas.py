import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, Dropout
from numpy import array
from keras import activations
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


x = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])


x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape[0]", x.shape[0])
print("x.shape[1]", x.shape[1])
print("x.shape", x.shape)
print("y.shape", y.shape)

def deep_cnn_advanced_nin():
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=2,  input_shape=(3, 1)))
    model.add(BatchNormalization())
    model.add(Activation='relu')
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=2,  input_shape=(3, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=2,  input_shape=(3, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=2, input_shape=(3, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(50, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')

    return model

model = deep_cnn_advanced_nin()
model.summary()

model.fit(x, y, epochs=1000, verbose=0)

#demonstrate prediction

x_input = array([60, 70, 90])
x_input = x_input.reshape(1, 3, 1)
yhat = model.predict(x_input, verbose=0)
print(yhat)

