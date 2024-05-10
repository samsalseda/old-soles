import tensorflow as tf


class ReflectionPad2D(tf.keras.layers.Layer):
    """
    def __init__(self, pad_x, pad_y):
    def call(self, input):
    """

    def __init__(self, pad_x=1, pad_y=1):
        """
        Initializes the reflection pad 2d layer.

        :param pad_x: padding left and right, defaults to 1
        :param pad_y: padding above and below, defaults to 1
        """
        super().__init__()
        self.paddings = tf.constant(
            [
                [0, 0],
                [
                    pad_x,
                    pad_x,
                ],
                [pad_y, pad_y],
                [0, 0],
            ]
        )

    def call(self, input):
        """
        Calls the reflection pad 2d layer.

        :param input: input to the layer
        :return: output from the layer
        """
        return tf.pad(input, self.paddings, mode="REFLECT")
