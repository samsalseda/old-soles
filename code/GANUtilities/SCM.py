import tensorflow as tf


class SCM_encoder(tf.keras.layers.Layer):
    def __init__(self, size):
        self.size = size
        self.encoder = tf.keras.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )


class SCM_decoder(tf.keras.layers.Layer):
    def __init__(self, size):
        self.size = size
