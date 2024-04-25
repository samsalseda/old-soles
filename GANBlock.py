import tensorflow as tf


class GANBlock(tf.keras.layers.Layer):
    def __init__(self):
        self.generator = gen_model
        self.discriminator = dis_model

    def call(self, inputs):
        g = self.generator(inputs)
        d = self.discriminator(g)
        return d


class Discriminator(tf.keras.layers.Layer):

    def __init__(self, input_size):
        self.input_size = input_size
        self.hidden_size = 32

        leaky_relu_layer = tf.keras.layers.LeakyReLU(negative_slope=0.5)
        self.disc1 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2d(
                    filters=self.hidden_size, kernel_size=4, strides=2, padding=1
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc2 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2d(
                    filters=self.hidden_size * 2, kernel_size=4, strides=2, padding=1
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc3 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2d(
                    filters=self.hidden_size * 4, kernel_size=4, strides=2, padding=1
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc4 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2d(
                    filters=self.hidden_size * 8, kernel_size=4, strides=2, padding=1
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc5 = tf.layers.Sequential(
            [
                tf.keras.layers.Conv2d(filters=1, kernel_size=4, strides=2, padding=1),
                leaky_relu_layer,
            ]
        )

        self.full_discriminaor = tf.layers.Sequential(
            [self.disc1, self.disc2, self.disc3, self.disc4, self.disc5]
        )

    def call(self, inputs):
        return self.full_discriminaor(inputs)
