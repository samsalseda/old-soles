import tensorflow as tf
from GANUtilities.MFFE import MFFEBlock


class GANBlock(tf.keras.layers.Layer):
    """
    def __init__(self):
    def call(self, inputs):
    """

    def __init__(self):
        """
        Initializes a GAN block layer.
        """
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self, inputs):
        """
        Calls the GAN Block.

        :param input: input to the layer
        :return: output from the layer
        """
        g = self.generator(inputs)
        d = self.discriminator(g)
        return d


class Discriminator(tf.keras.layers.Layer):
    """
    def __init__(self, input_size):
    def call(self, inputs):
    """

    def __init__(self, input_size):
        """
        Initializes a GAN block layer.

        :param input_size: size of the input
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 32

        leaky_relu_layer = tf.keras.layers.LeakyReLU(negative_slope=0.5)
        self.disc1 = tf.keras.Sequential(
            [
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        filters=self.hidden_size,
                        kernel_size=4,
                        strides=(2, 2),
                        padding="SAME",
                    )
                ),
                leaky_relu_layer,
            ]
        )

        self.disc2 = tf.keras.Sequential(
            [
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        filters=self.hidden_size * 2,
                        kernel_size=4,
                        strides=(2, 2),
                        padding="SAME",
                    )
                ),
                leaky_relu_layer,
            ]
        )

        self.disc3 = tf.keras.Sequential(
            [
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        filters=self.hidden_size,
                        kernel_size=4,
                        strides=(2, 2),
                        padding="SAME",
                    )
                ),
                leaky_relu_layer,
            ]
        )

        self.disc4 = tf.keras.Sequential(
            [
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        filters=self.hidden_size * 8,
                        kernel_size=4,
                        strides=(2, 2),
                        padding="SAME",
                    )
                ),
                leaky_relu_layer,
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
        """
        Calls the GAN discriminator.

        :param inputs: inputs to the layer
        :return: output from the layer
        """
        return self.full_discriminaor(inputs)


class Generator(tf.keras.layers.Layer):

    def __init__(self, size=[256, 4, 4], splits=4, resnet=True, block_size=8):
        super().__init__()
        self.height, self.width = size[1], size[2]
        self.MFFE = MFFEBlock(size, splits, resnet, block_size)

    def call(self, inputs):
        """
        Calls the GAN generator.

        :param inputs: inputs to the layer
        :return: output from the layer
        """
        return self.MFFE(inputs)
