import tensorflow as tf
import ReflectionPad2D

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, input_size):
        self.input_size = input_size
        self.conv_block = tf.keras.Sequential(
            [
                ReflectionPad2D.ReflectionPad2d(),
                tf.keras.layers.SpectralNormalization(tf.keras.layers.Conv2D(
                    filters=self.input_size, kernel_size=3, strides=(2,2), padding='VALID'
                )),
                tf.keras.layers.Groupnormalization(groups = -1),
                tf.keras.layers.ReLU(),
                
                ReflectionPad2D.ReflectionPad2d(),
                tf.keras.layers.SpectralNormalization(tf.keras.layers.Conv2D(
                    filters=self.input_size, kernel_size=3, strides=(2,2), padding='VALID'
                )),
                tf.keras.layers.Groupnormalization(groups = -1),

                #May need to add more layers here
            ]
        )
        

    def call(self, inputs):
        return inputs + self.conv_block(inputs)
    
