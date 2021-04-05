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



x = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(BatchNormalization)
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(BatchNormalization)
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

input = tf.keras.layers.Input(shape=(3, 1))

vgg_block01 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1))
vgg_block01 = tf.keras.layers.BatchNormalization()
vgg_block01 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1))
vgg_block01 = tf.keras.layers.BatchNormalization()
vgg_block01 = tf.keras.layers.MaxPool1D(pool_size=2)  # 14x14x64


vgg_block02 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1))
vgg_block02 = tf.keras.layers.BatchNormalization()
vgg_block02 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1))
vgg_block02 = tf.keras.layers.BatchNormalization()
vgg_block02 = tf.keras.layers.MaxPool1D(pool_size=2) # 8x8x128



#vgg_block03 = tf.keras.layers.Conv1D(filters=64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block02)
#vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
#vgg_block03 = tf.keras.layers.Conv1D(filters=64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block03)
#vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
#vgg_block03 = tf.keras.layers.Conv1D(filters=64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block03)
#vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
#vgg_block03 = tf.keras.layers.MaxPool1D(pool_size=2)(vgg_block03)  # 4x4x256

#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block03)
#vgg_block04 = tf.keras.layers.BatchNormalization(vgg_block04)
#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
#vgg_block04= tf.keras.layers.BatchNormalization(vgg_block04)
#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
#vgg_block04 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block04)  # 1x1x512


flatten = tf.keras.layers.Flatten()(vgg_block02)  # 512개

dense01 = tf.keras.layers.Dense(256, activation='relu')(flatten)

output = tf.keras.layers.Dense(10, activation='softmax')(dense01)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#fit model

model.fit(x, y, batch_size=64, epochs=10,
         validation_data=(x, y))

#demonstrate prediction

x_input = array([60, 70, 90])
x_input = x_input.reshape(1, 3, 1)
yhat = model.predict(x_input, verbose=0)
print(yhat)
