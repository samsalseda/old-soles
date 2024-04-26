import tensorflow as tf
import Resnet
import ReflectionPad2D


class GANBlock(tf.keras.layers.Layer):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self, inputs):

        # Need z sampling of inputs
        g = self.generator(inputs)
        d = self.discriminator(g)
        return d


class Discriminator(tf.keras.layers.Layer):

    def __init__(self, input_size):
        self.input_size = input_size
        self.hidden_size = 32

        leaky_relu_layer = tf.keras.layers.LeakyReLU(negative_slope=0.5)
        self.disc1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.hidden_size,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="SAME",
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.hidden_size * 2,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="SAME",
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc3 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.hidden_size * 4,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="SAME",
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc4 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.hidden_size * 8,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="SAME",
                ),
                leaky_relu_layer,
                tf.keras.layers.SpectralNormalization(),
            ]
        )

        self.disc5 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=4, strides=(2, 2), padding="SAME"
                ),
                leaky_relu_layer,
            ]
        )

        self.full_discriminaor = tf.keras.Sequential(
            [self.disc1, self.disc2, self.disc3, self.disc4, self.disc5]
        )

    def call(self, inputs):
        return self.full_discriminaor(inputs)


class Generator(tf.keras.layers.Layer):

    def __init__(self, input_size, resnet=True, block_size=8):

        blocks = []
        if resnet:
            for i in range(block_size):
                block = Resnet.ResidualBlock(256)
                blocks.append(block)

        else:
            for i in range(block_size):
                block = tf.keras.layers.Dense(256)
                blocks.append(block)

        self.middle = tf.keras.Sequential(*blocks)

        self.generator = tf.keras.Sequential(
            [
                ReflectionPad2D.ReflectionPad2D(),
                tf.keras.layers.Conv2D(
                    filters=self.input_size,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="VALID",
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.SpectralNormalization(),
                tf.keras.layers.GroupNormalization(groups=-1),
                # May need to add more layers here
            ]
        )

    def call(self, inputs):
        pass
