import tensorflow as tf
from AFRM import AFRM
from SCM import SCM_decoder, SCM_encoder

class MFFEBlock(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4], splits=4):
        self.encoder = SCM_encoder(size)
        self.decoder = SCM_decoder(size)

        self.AFRM = AFRM(size, splits)

        hidden_size = size[1] * size[2]

        self.middle = tf.keras.Sequential([tf.keras.layers.Conv2D(hidden_size)])

    def call(self, x):
        encoder_output = self.encoder(x)
        afrm_output = self.AFRM(encoder_output)
        middle_output = self.middle(afrm_output)
        decoder_output = self.decoder(middle_output)
        return decoder_output