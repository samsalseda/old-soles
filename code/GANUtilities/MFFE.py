import tensorflow as tf
from GANUtilities.AFRM import AFRM
from GANUtilities.SCM import SCM_decoder, SCM_encoder
from GANUtilities.Resnet import ResidualBlock


class MFFEBlock(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4], splits=4, resnet=True, block_size=8):
        super().__init__()
        self.encoder = SCM_encoder(size)
        self.decoder = SCM_decoder(size)

        self.AFRM = AFRM(size, splits)

        # hidden_size = size[1] * size[2]

        blocks = []
        if resnet:  # TODO: is this correct?
            for i in range(block_size):
                block = ResidualBlock(256)
                blocks.append(block)

        else:
            for i in range(block_size):
                block = tf.keras.layers.Dense(256)
                blocks.append(block)

        self.middle = tf.keras.Sequential([*blocks])

    def call(self, x):
        print(x.shape)
        encoder_output = self.encoder(x)
        afrm_output = self.AFRM(encoder_output)
        print(f"afrm shape: {afrm_output.shape}")
        middle_output = self.middle(afrm_output)
        print(f"middle shape: {middle_output.shape}")
        decoder_output = self.decoder(middle_output)
        print(f"decoder shape: {decoder_output.shape}")
        return decoder_output
