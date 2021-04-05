
from keras import models
from keras.applications import VGG16
import tensorflow as tf #### tensorflow를 tf라고 지칭
import numpy as np

def vgg_block(in_layer, n_conv, n_filter, filter_size=(3, 3), reduce_size=True):
    layer = in_layer
    for i in range(n_conv):
        layer = tf.keras.layers.Con2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)

    if reduce_size:
        layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
    return layer


input = tf.keras.layers.Input(shape=(28, 28, 1))

vgg_block01 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(input)
vgg_block01 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(vgg_block01)
vgg_block01 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block01)  # 14x14x64

vgg_block02 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu')(vgg_block01)
vgg_block02 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu')(vgg_block02)
vgg_block02 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block02)  # 8x8x128

vgg_block03 = tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu')(vgg_block02)
vgg_block03 = tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu')(vgg_block03)
vgg_block03 = tf.keras.layers.Conv2D(256, (3, 3), padding='SAME', activation='relu')(vgg_block03)
vgg_block03 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block03)  # 4x4x256

vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block03)
vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
vgg_block04 = tf.keras.layers.Conv2D(512, (3, 3), padding='SAME', activation='relu')(vgg_block04)
vgg_block04 = tf.keras.layers.MaxPool2D((2, 2))(vgg_block04)  # 1x1x512

flatten = tf.keras.layers.Flatten()(vgg_block04)  # 512개

dense01 = tf.keras.layers.Dense(256, activation='relu')(flatten)

output = tf.keras.layers.Dense(10, activation='softmax')(dense01)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='spare_categorical_crossentropy', metrics=['accuracy'])
model.summary()

mode.fit(x_train, y_train, batch_size=64, epochs=10,
         validation_data=(x_test, y_test))
