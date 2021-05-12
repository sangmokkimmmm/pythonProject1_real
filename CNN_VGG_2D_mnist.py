import tensorflow as tf
import numpy as np

# Data Prepare START
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10,
         validation_data=(x_test, y_test))



# Data Prepare END
#def vgg_block(in_layer, n_conv, n_filter, filter_size=(3, 3), reduce_size=True):
    #layer = in_layer
    #for i in range(n_conv):
        #layer = tf.keras.layers.Con2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)

   # if reduce_size:
        #layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
    #return layer