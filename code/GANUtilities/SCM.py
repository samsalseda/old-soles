import tensorflow as tf
from GANUtilities.AFRM import AFRM
from GANUtilities.ReflectionPad2D import ReflectionPad2D


class SCM_encoder(tf.keras.layers.Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size
        # change to like size, size * 2, size * 4?
        self.encoder = tf.keras.Sequential([
            ReflectionPad2D(3, 3),
            tf.keras.layers.Conv2D(64, 7, padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            tf.keras.layers.Conv2D(128, 4, strides=(2, 2), padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            tf.keras.layers.Conv2D(256, 4, strides=(4, 4), padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU()]
        )

    def call(self, x):
        return self.encoder(x)


class SCM_decoder(tf.keras.layers.Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size
        # change to like size, size / 2, size / 4?
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            tf.keras.layers.Conv2DTranspose(128, 4, strides=(2, 2), padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            tf.keras.layers.Conv2D(64, 4, strides=(2, 2), padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU(),
            ReflectionPad2D(3, 3),
            tf.keras.layers.Conv2D(3, 7, padding="valid"),
            tf.keras.layers.GroupNormalization(groups=-1),
            tf.keras.layers.ReLU(),]
        )

    def call(self, x):
        return self.decoder(x)
