import tensorflow as tf
import numpy as np


class AFRM(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4], splits=4):
        super().__init__()
        self.num_channels, self.height, self.width = size
        self.num_splits = splits

        # print(size)
        # print(self.height/16)
        # print(np.log2(self.height/16))

        num_layers = int(np.log2(self.height/16/2))

        print(f"layers: {num_layers}")

        encoder_layers = [tf.keras.layers.Identity()]

        for i in range(num_layers):
            encoder_layers.append(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            encoder_layers.append(tf.keras.layers.Conv2D(
                    self.num_channels,
                    4,
                    strides=2
                ))
            encoder_layers.append(tf.keras.layers.GroupNormalization(groups=-1))
            encoder_layers.append(tf.keras.layers.ReLU())


        self.encoder = tf.keras.Sequential(
            [*encoder_layers]
        )

        decoder_layers = [tf.keras.layers.Identity()]

        for i in range(num_layers):
            #decoder_layers.append(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            decoder_layers.append(tf.keras.layers.Conv2DTranspose(
                    self.num_channels,
                    2,
                    strides=2,
                    padding="valid"
                ),)
            decoder_layers.append(tf.keras.layers.GroupNormalization(groups=-1))
            decoder_layers.append(tf.keras.layers.ReLU())

        self.decoder = tf.keras.Sequential(decoder_layers)

        self.lstm_size = int(self.num_channels * 4 * int(4 / splits)) #TODO: remove /8
        self.forward_LSTM = tf.keras.layers.LSTM(self.lstm_size, return_state=True, return_sequences=True)
        self.reverse_LSTM = tf.keras.layers.LSTM(self.lstm_size, return_state=True, return_sequences=True)

        self.conv_output = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    self.num_channels, 1, padding="valid", use_bias=True, dilation_rate=1
                ),
                tf.keras.layers.LeakyReLU(),
            ]
        )

        self.gamma = tf.Variable(tf.zeros(1))

    def call(self, x):
        batch_size = x.shape[0]
        output = x
        output = self.encoder(output)

        print(f"x: {x.shape}")
        print(f"output: {output.shape}")

        split_output = tf.reshape(
            tf.concat(
                tf.split(output, self.num_splits, axis=2), axis=0),
            (self.num_splits, -1, 1, int(self.lstm_size)),
        )


        reversed_split_output = tf.reshape(
            tf.reverse(
                tf.concat(
                    tf.split(output, self.num_splits, axis=2), axis=0), axis=[3]),
            (self.num_splits, -1, 1, int(self.lstm_size)),
        )
        

        initial_state =[tf.zeros((split_output[0].shape[0], split_output[0].shape[2])),
                       tf.zeros((split_output[0].shape[0], split_output[0].shape[2]))]
            #tf.zeros((1, batch_size, self.lstm_size)),
            #tf.zeros((1, batch_size, self.lstm_size))]
        fwd_output = []

        for i in range(self.num_splits):
            #print(i)
            #print(split_output[i].shape)
            lstm_output, initial_h, initial_c = self.forward_LSTM(
                #split_output[i], tf.reshape(initial_state, (split_output[i].shape))
                split_output[i], initial_state=initial_state
            )
            #print(initial_c.shape)
            #print(initial_h.shape)

            initial_state = [initial_c, initial_h]
            #print(f"lstm_output: {lstm_output.shape}")
            lstm_output = tf.reshape(
                lstm_output,
                (-1, int(self.num_channels), 4, 1),
            )

            if i == 0:
                fwd_output = lstm_output
            else:
                fwd_output = tf.concat([fwd_output, lstm_output], axis=3)

        initial_state =[tf.zeros((split_output[0].shape[0], split_output[0].shape[2])),
                       tf.zeros((split_output[0].shape[0], split_output[0].shape[2]))]
        
        rvs_output = []

        for i in range(self.num_splits):
            lstm_output, initial_h, initial_c = self.reverse_LSTM(
                reversed_split_output[i], initial_state
            )
            lstm_output = tf.reshape(
                lstm_output,
                (-1, self.num_channels, 4, 1),
            )

            initial_state = [initial_c, initial_h]

            if i == 0:
                rvs_output = lstm_output
            else:
                rvs_output = tf.concat([rvs_output, lstm_output], axis=3)

        print(fwd_output.shape)
        print(rvs_output.shape)

        reshaped_input = tf.transpose(tf.concat([fwd_output, rvs_output], 1), [0, 2, 3, 1])
        print(reshaped_input.shape)
        output = self.conv_output(reshaped_input)
        print(output.shape)
        output = self.decoder(output)
        print(output.shape)

        output = self.gamma * output + x

        print("finished call!")
        print()
        print()
        print()
        
        return output
