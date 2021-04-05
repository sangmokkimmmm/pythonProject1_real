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

def sklearn_csv_to_data(csv_file):
    df = pd.read_csv('capp.csv', header = 0)
    x = list(df.columns.values)
    data = df.values
    return x, data

x, data1 = sklearn_csv_to_data('capp.csv')

print(data1)

def sklearn_csv_to_data(csv_file):
    df = pd.read_csv('capp_y.csv', header = 0)
    y = list(df.columns.values)
    data = df.values
    return y, data

y, data2 = sklearn_csv_to_data('capp_y.csv')

print(data2)

def sklearn_csv_to_data(csv_file):
    df = pd.read_csv('capp_test.csv', header = 0)
    m = list(df.columns.values)
    data = df.values
    return m, data

m, data3 = sklearn_csv_to_data('capp_y.csv')

def sklearn_csv_to_data(csv_file):
    df = pd.read_csv('capp_test_y.csv', header = 0)
    n = list(df.columns.values)
    data = df.values
    return n, data

n, data4 = sklearn_csv_to_data('capp_y.csv')

print("data1.shape[0]", data1.shape[0])
print("data1.shape[1]", data1.shape[1])
print("data1.shape", data1.shape)
print("data2.shape", data2.shape)


data1 = data1.reshape((data1.shape[0], data1.shape[1], 1))
data3 = data3.reshape((data3.shape[0], data3.shape[1], 1))


#int->float 바꿔주기

data1 = data1.astype(float)
print(data1)

data2 = data2.astype(float)
print(data2)

###############################


model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, padding='SAME', input_shape=(8, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=2, padding='SAME', input_shape=(8, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=2, padding='SAME', input_shape=(8, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))



model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(2))

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])


model.summary()

#fit model

model.fit(data1, data2, epochs=1000, verbose=1, validation_data=(data3, data4))

#demonstrate prediction

#x_input = array([150, 75, 40, 50, 60, 30, 20, 10])
#x_input = x_input.reshape(1, 8, 1).
#yhat = model.predict(x_input, verbose=0)
#print(yhat)


# define model

