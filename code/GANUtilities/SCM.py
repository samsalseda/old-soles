import tensorflow as tf
from GANUtilities.ReflectionPad2D import ReflectionPad2D


class SCM_encoder(tf.keras.layers.Layer):
    """
    def __init__(self, size):
    def call(self, x):
    """

    def __init__(self, size):
        """
        Initializes the SCM encoder with a sequential set of convolution and normalization layers

        :param size: the size of the model image input
        """
        super().__init__()
        self.size = size
        self.encoder = tf.keras.Sequential(
            [
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
                tf.keras.layers.ReLU(),
            ]
        )

    def call(self, x):
        """
        Calls the SCM encoder.

        :param x: input to the layer
        :return: output from the layer
        """
        return self.encoder(x)


class SCM_decoder(tf.keras.layers.Layer):
    """
    def __init__(self, size):
    def call(self, x):
    """

    def __init__(self, size):
        """
        Initializes the SCM decoder with a sequential set of convolution transpose and normalization layers.

        :param size: the size of the model image input
        """
        super().__init__()
        self.size = size
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="valid"),
                tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
                tf.keras.layers.GroupNormalization(groups=-1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
                tf.keras.layers.GroupNormalization(groups=-1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(3, 7, padding="same"),
            ]
        )

    def call(self, x):
        """
        Calls the SCM decoder.

        :param x: input to the layer
        :return: output from the layer
        """
        return self.decoder(x)
