import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, Dropout
#df는 표 입니다.


x_train = pd.read_csv("capp.csv")
y_train = pd.read_csv("capp_y.csv")
x_test = pd.read_csv("capp_test.csv")
y_test = pd.read_csv("capp_test_y.csv")

#def minmax_norm(x_train_input):
 #   return (x_train - x_train.min()) / (x_train.max() - x_train.min())

#x_train = minmax_norm(x_train)

#def minmax_norm(y_train_input):
 #   return (y_train - y_train.min()) / (y_train.max() - y_train.min())

#y_train = minmax_norm(y_train)

#def minmax_norm(x_test_input):
 #   return (x_test - x_test.min()) / (x_test.max() - x_test.min())

#x_test = minmax_norm(x_test)

#def minmax_norm(y_test_input):
 #   return (y_test - y_test.min()) / (y_test.max() - y_test.min())

#y_test = minmax_norm(y_test)

#data 정규화

#x_train = np.expand_dims(x_train, axis=-1)
#x_test = np.expand_dims(x_test, axis=-1)
#inputshapes: [?, 16, 1, 256].

#data

input = tf.keras.layers.Input(shape=(7, 16, 1))

vgg_block0=tf.keras.layers.Conv1D


vgg_block01 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(input)
vgg_block01 = tf.keras.layers.BatchNormalization(vgg_block01)
vgg_block01 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block01)
vgg_block01 = tf.keras.layers.BatchNormalization(vgg_block01)
vgg_block01 = tf.keras.layers.MaxPool1D(pool_size=2)(vgg_block01)  # 14x14x64


vgg_block02 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block01)
vgg_block02 = tf.keras.layers.BatchNormalization(vgg_block02)
vgg_block02 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block02)
vgg_block02 = tf.keras.layers.BatchNormalization(vgg_block02)
vgg_block02 = tf.keras.layers.MaxPool1D(pool_size=2)(vgg_block02)  # 8x8x128

vgg_block03 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block02)
vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
vgg_block03 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block03)
vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
vgg_block03 = tf.keras.layers.Conv1D(64, kernel_size=2 , padding='SAME', activation='relu')(vgg_block03)
vgg_block03 = tf.keras.layers.BatchNormalization(vgg_block03)
vgg_block03 = tf.keras.layers.MaxPool1D(pool_size=2)(vgg_block03)  # 4x4x256

#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block03)
#vgg_block04 = tf.keras.layers.BatchNormalization(vgg_block04)
#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
#vgg_block04= tf.keras.layers.BatchNormalization(vgg_block04)
#vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
#vgg_block04 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block04)  # 1x1x512

flatten = tf.keras.layers.Flatten()(vgg_block03)  # 512개

dense01 = tf.keras.layers.Dense(256, activation='relu')(flatten)

output = tf.keras.layers.Dense(10, activation='softmax')(dense01)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10,
         validation_data=(x_test, y_test))

