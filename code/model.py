import argparse
import tensorflow as tf
import numpy as np


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
    # TODO: Change arguments to include batch_size, maybe even something about number of GAN blocks to use, etc.
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
    # TODO: call prorocessing, build the model
    # TODO: train using the preprocessed image stuff
    # TODO: test using the preprocessed image stuff
    pass


class ShoeGenerationModel(tf.keras.Model):

    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.generator, self.discriminator = generator, discriminator
        # TODO: convert parameters to lists, for multile GAN blocks to iterate thorugh

    @tf.function
    def call(self, sketch_images, real_images):
        return self.discriminator(self.generator(sketch_images))

    def compile(self, optimizer, loss, metrics):
        """
        Create a facade to mimic normal keras fit routine
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def train(self, real_images, sketch_images, batch_size=30, epochs=1):
        """
        Runs through all Epochs and trains
        """
        for e in range(epochs):
            avg_loss = 0
            avg_acc = 0

            num_batches = int(len(sketch_images) / batch_size)
            total_loss = 0

            indicies_unshuffled = tf.range(len(sketch_images))
            indicies = tf.random.shuffle(indicies_unshuffled)
            train_real_images_shuffled = tf.gather(real_images, indicies)
            train_sketch_images_shuffled = tf.gather(sketch_images, indicies)

            for index, end in enumerate(
                range(batch_size, len(real_images) + 1, batch_size)
            ):
                start = end - batch_size
                batch_sketch_images = train_sketch_images_shuffled[start:end, :]
                batch_real_images = train_real_images_shuffled[start:end, :]

                with tf.GradientTape() as tape:
                    output = self.discriminator(self.generator(batch_sketch_images))
                    loss = self.loss(
                        output, batch_real_images
                    )  # TODO: ADD PARAMS like in losses.py file
                    metrics = []
                    for metric in self.metrics:
                        metrics += [
                            metric(output, batch_real_images)
                        ]  ## TODO: ADJUST params as needed
                    np.asarray(metrics)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                total_loss += loss
                avg_loss = float(total_loss / (start - end))
                avg_metrics = float(metrics / (start - end))

                print(
                    f"\r[Epoch: {e} \t Batch Index: {index+1}/{num_batches}]\t batch_loss={avg_loss:.3f}\t batch_metrics: {avg_acc:.3f}\t",
                    end="",
                )

        return avg_loss, avg_metrics

    def train(self, real_images, sketch_images):
        """
        Runs through all Epochs and trains
        """
        avg_loss = 0
        avg_acc = 0

        total_loss = 0
        num_examples = len(real_images)

        output = self.discriminator(self.generator(sketch_images))
        loss = self.loss(output, real_images)  # TODO: ADD PARAMS like in losses.py file
        metrics = []
        for metric in self.metrics:
            metrics += [metric(output, real_images)]  ## TODO: ADJUST params as needed
        np.asarray(metrics)

        total_loss += loss
        avg_loss = float(total_loss / num_examples)
        avg_metrics = float(metrics / num_examples)

        print(
            f"\r[Testing: loss={avg_loss:.3f}\t metrics: {avg_acc:.3f}\t",
            end="",
        )

        return avg_loss, avg_metrics

    # def get_config(self):
    #     base_config = super().get_config()
    #     config = {
    #         "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
    #     }
    #     return {**base_config, **config}

    # @classmethod
    # def from_config(cls, config):
    #     decoder_config = config.pop("decoder")
    #     decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
    #     return cls(decoder, **config)


if __name__ == "__main__":
    main(parse_args())
