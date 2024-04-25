import tensorflow as tf

class AFRM(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4]):
        self.channels, self.heigh, self.width = size
        self.encoder = tf.keras.Sequential([tf.keras.layers.Conv2d(self.channels, (4, 4, self.channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2d(self.channels, (4, 4, self.channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2d(self.channels, (4, 4, self.channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2d(self.channels, (4, 4, self.channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU()])
        self.decoder = tf.keras.Sequential([])

    
    def call(self, x):
        print()