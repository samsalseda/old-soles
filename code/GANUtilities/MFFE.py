import tensorflow as tf
from GANUtilities.AFRM import AFRM
from GANUtilities.SCM import SCM_decoder, SCM_encoder
from GANUtilities.Resnet import ResidualBlock


class MFFEBlock(tf.keras.layers.Layer):
    """
    def __init__(self, size=[256, 4, 4], splits=4, resnet=True, block_size=8):
    def call(self, x):
    """

    def __init__(self, size=[256, 4, 4], splits=4, resnet=True, block_size=8):
        """
        Initializes the MFFEBlock layer.

        :param size: the size of the block, defaults to [255, 4, 4]
        :param splits: number of splits, defaults to 4
        :param resnet: whether to use additional residual blocks, defaults to true
        :param block_size: number of residual blocks to use, defaults to 8
        """
        super().__init__()
        self.encoder = SCM_encoder(size)
        self.decoder = SCM_decoder(size)

        self.AFRM = AFRM(size, splits)

        blocks = []
        if resnet:
            for _ in range(block_size):
                block = ResidualBlock(256)
                blocks.append(block)

        else:
            for _ in range(block_size):
                block = tf.keras.layers.Dense(256)
                blocks.append(block)

        self.middle = tf.keras.Sequential([*blocks])

    def call(self, x):
        """
        Calls the MFFEBlock layer.

        :param x: input to the layer
        :return: output from the layer
        """
        encoder_output = self.encoder(x)
        afrm_output = self.AFRM(encoder_output)
        middle_output = self.middle(afrm_output)
        decoder_output = self.decoder(middle_output)
        return decoder_output
