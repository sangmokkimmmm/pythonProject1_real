import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, Dropout
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Activation



x = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([[50, 10], [60, 10], [70, 10], [80, 10]])

x = x.reshape((x.shape[0], x.shape[1], 1))

print("x.shape[0]", x.shape[0])
print("x.shape[1]", x.shape[1])
print("x.shape", x.shape)
print("y.shape", y.shape)

# define model

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, padding='SAME', input_shape=(4, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=2, padding='SAME', input_shape=(4, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))


model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(2))

model.compile(optimizer='adam', loss='mse')


model.summary()

#fit model
model.fit(x, y, epochs=1000, verbose=0)

#demonstrate prediction

x_input = array([60, 70, 90, 100])
x_input = x_input.reshape(1, 4, 1)
yhat = model.predict(x_input, verbose=0)
print(yhat)
