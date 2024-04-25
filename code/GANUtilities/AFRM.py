import tensorflow as tf


class AFRM(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4]):
        self.size = size
        self.encoder = tf.keras.Sequential([])

    def call(self, x):
        print()
