import tensorflow as tf

class GANBlock(tf.keras.layers.Layer):
    def __init__(self):
        self.generator = []
        self.discriminator = []