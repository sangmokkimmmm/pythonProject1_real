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

input = tf.keras.layers.Input(shape=(3, 1, 1))

vgg_block01 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1,1))
vgg_block01 = tf.keras.layers.BatchNormalization()
vgg_block01 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1,1 ))
vgg_block01 = tf.keras.layers.BatchNormalization()
vgg_block01 = tf.keras.layers.MaxPool1D(pool_size=2)  # 14x14x64


vgg_block02 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1,1 ))
vgg_block02 = tf.keras.layers.BatchNormalization()
vgg_block02 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1,1 ))
vgg_block02 = tf.keras.layers.BatchNormalization()
vgg_block02 = tf.keras.layers.MaxPool1D(pool_size=2) # 8x8x128

flatten = tf.keras.layers.Flatten()  # 512개

dense01 = tf.keras.layers.Dense(256, activation='relu')(flatten)

output = tf.keras.layers.Dense(10, activation='softmax')(dense01)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()