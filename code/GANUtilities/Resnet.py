import tensorflow as tf
from GANUtilities.ReflectionPad2D import ReflectionPad2D


class ResidualBlock(tf.keras.layers.Layer):
    """
    def __init__(self, size):
    def call(self, inputs):
    """

    def __init__(self, input_size):
        """
        Initializes the resnet layer.

        :param input_size: the size of the model image input
        """
        super().__init__()
        self.input_size = input_size
        self.conv_block = tf.keras.Sequential(
            [
                ReflectionPad2D(),
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        self.input_size,
                        3,
                        strides=(1, 1),
                        padding="VALID",
                        dilation_rate=1,
                    )
                ),
                tf.keras.layers.GroupNormalization(groups=-1),
                tf.keras.layers.ReLU(),
                ReflectionPad2D(),
                tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        self.input_size,
                        3,
                        strides=(1, 1),
                        padding="VALID",
                        dilation_rate=1,
                    )
                ),
                tf.keras.layers.GroupNormalization(groups=-1),
            ]
        )

    def call(self, inputs):
        """
        Calls the resnet layer.

        :param x: input to the layer
        :return: output from the layer
        """
        return inputs + self.conv_block(inputs)
