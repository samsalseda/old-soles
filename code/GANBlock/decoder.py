import tensorflow as tf

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, size):
        self.size = size