import tensorflow as tf


class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, pad_x=1, pad_y=1):
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
        return tf.pad(input, self.paddings, mode="REFLECT")
