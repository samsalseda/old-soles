import tensorflow as tf
from GANUtilities.ReflectionPad2D import ReflectionPad2D


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, input_size):
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
        return inputs + self.conv_block(inputs)
