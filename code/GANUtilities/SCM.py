import tensorflow as tf

class SCM(tf.keras.layers.Layer):
    def __init__(self, size):
        self.size = size