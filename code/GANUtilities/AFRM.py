import tensorflow as tf

class AFRM(tf.keras.layers.Layer):
    def __init__(self, size=[256, 4, 4], splits=4):
        self.num_channels, self.height, self.width = size
        self.num_splits = splits
        self.encoder = tf.keras.Sequential([tf.keras.layers.Conv2D(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2D(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2D(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2D(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU()])
        
        self.decoder = tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2DTranspose(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2DTranspose(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            tf.keras.layers.Conv2DTranspose(self.num_channels, (4, 4, self.num_channels), strides=2, padding='same'),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU()])
        
        self.lstm_size = int(self.channel * 4 * int(4 / splits))
        self.forward_LSTM = tf.keras.layers.LSTM(self.lstm_size, return_state=True)
        self.reverse_LSTM = tf.keras.layers.LSTM(self.lstm_size, return_state=True)

        self.ffwd = tf.keras.Sequential([tf.keras.layers.Conv2D(2 * self.num_channels, self.num_channels, padding='valid'),
                                         tf.keras.layers.LeakyReLU()])

        self.gamma = tf.Variable(tf.zeros(1))
        
    
    def call(self, x):
        batch_size = x.shape[0]
        output = x 
        output = self.encoder(output)

        split_output = tf.reshape(tf.concat(tf.split(output, self.num_splits, axis=3)), (self.num_splits, -1, 1, self.lstm_size))
        reversed_split_output = tf.reshape(tf.reverse(tf.concat(tf.split(output, self.num_splits, axis=3)), axis=3),  (self.num_splits, -1, 1, self.lstm_size))

        initial_state = (tf.zeros((2, batch_size, self.lstm_size)), tf.zeros((2, batch_size, self.lstm_size)))
        fwd_output = []

        for i in range(self.num_splite):
            lstm_output, initial_state = self.forward_LSTM(split_output[i], initial_state)
            lstm_output = tf.reshape(lstm_output, (-1, self.num_channels, self.height, int(self.width/self.num_splits)))

            if i == 0:
                fwd_output = lstm_output
            else:
                fwd_output = tf.concat([fwd_output, lstm_output])

        initial_state = (tf.zeros((2, batch_size, self.lstm_size)), tf.zeros((2, batch_size, self.lstm_size)))
        rvs_output = []

        for i in range(self.num_splite):
            lstm_output, initial_state = self.reverse_LSTM(reversed_split_output[i], initial_state)
            lstm_output = tf.reshape(lstm_output, (-1, self.num_channels, self.height, int(self.width/self.num_splits)))

            if i == 0:
                rvs_output = lstm_output
            else:
                rvs_output = tf.concat([rvs_output, lstm_output])

        output = self.ffwd(tf.concat(fwd_output, rvs_output, axis=1))
        output = self.decoder(output)

        return self.gamma * output + x