import argparse
import tensorflow as tf


def parse_args(args=None):
    """
    Perform command-line argument parsing (other otherwise parse arguments with defaults).
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example:
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["rnn", "transformer"],
        help="Type of model to train",
    )
    if args is None:
        return parser.parse_args()  ## For calling through command line
    return parser.parse_args(args)  ## For calling through notebook.


def main(self, args):
    gist, color = get_feature_dicts()


class ShoeGenerationModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, sketch_images, real_images):
        pass

    def compile(self, optimizer, loss, metrics):
        """
        Create a facade to mimic normal keras fit routine
        """
        self.optimizer = optimizer
        self.total_loss_function = loss
        self.accuracy = metrics

    def train(self, real_images, sketch_images, batch_size=30):
        """
        Runs through all Epochs and trains


        """
        avg_loss = 0
        avg_acc = 0
        avg_prp = 0

        num_batches = int(len(sketch_images) / batch_size)
        total_loss = total_seen = total_correct = 0

        indicies_unshuffled = tf.range(len(sketch_images))
        indicies = tf.random.shuffle(indicies_unshuffled)
        train_captions_shuffled = tf.gather(train_captions, indicies)
        train_image_features_shuffled = tf.gather(train_image_features, indicies)

        for index, end in enumerate(
            range(batch_size, len(train_captions) + 1, batch_size)
        ):
            start = end - batch_size
            batch_image_features = train_image_features_shuffled[start:end, :]
            decoder_input = train_captions_shuffled[start:end, :-1]
            decoder_labels = train_captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                probs = self(batch_image_features, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                # print(probs.shape)
                # print(decoder_labels.shape)
                # print(padding_index)
                loss = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)

            print(
                f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}",
                end="",
            )
        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather)
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)

        return avg_loss, avg_acc, avg_prp

    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)

    # def get_config(self):
    #     return {"decoder": self.decoder}  ## specific to ImageCaptionModel

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)


if __name__ == "__main__":
    main(parse_args())
    # call prorocessing, build the model
    # train using the preprocessed image stuff
