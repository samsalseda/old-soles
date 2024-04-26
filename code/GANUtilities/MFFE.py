import tensorflow as tf
from AFRM import AFRM
from SCM import SCM_decoder, SCM_encoder
from Resnet import ResidualBlock


class MFFEBlock(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4], splits=4, resnet = True, block_size = 8):
        self.encoder = SCM_encoder(size)
        self.decoder = SCM_decoder(size)

        self.AFRM = AFRM(size, splits)

        # hidden_size = size[1] * size[2]

        blocks = []
        if resnet:
            for i in range(block_size):
                block = ResidualBlock(256)
                blocks.append(block)

        else:
            for i in range(block_size):
                block = tf.keras.layers.Dense(256)
                blocks.append(block)

        self.middle = tf.keras.Sequential(*blocks)

    def call(self, x):
        encoder_output = self.encoder(x)
        afrm_output = self.AFRM(encoder_output)
        middle_output = self.middle(afrm_output)
        decoder_output = self.decoder(middle_output)
        return decoder_output
