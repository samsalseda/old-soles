import argparse
import tensorflow as tf
import numpy as np
from GANBlock import Generator, Discriminator


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
        "--num_blocks",
        type=int,
        default=5,
        help="number of GAN Blocks in the model",
    )
    if args is None:
        return parser.parse_args()  ## For calling through command line
    return parser.parse_args(args)  ## For calling through notebook.


def main(self, args):
    # TODO: call prorocessing, build the model
    # TODO: train using the preprocessed image stuff
    # TODO: test using the preprocessed image stuff
    
    generators = []
    discriminators = []
    res = 256

    for i in range(args.num_blocks):
        
        generators += [Generator([256, res, res])]
        discriminators += [Discriminator(res * res)]

        res /= 2

        generators.reverse()
        discriminators.reverse()

    model = ShoeGenerationModel(generators, discriminators)


class ShoeGenerationModel(tf.keras.Model):

    def __init__(self, generators, discriminators, **kwargs):
        super().__init__(**kwargs)
        self.generators, self.discriminators = generators, discriminators
        # TODO: convert parameters to lists, for multile GAN blocks to iterate thorugh

    @tf.function
    def call(self, sketch_images, real_images):
        input_images = tf.keras.layers.Resizing(self.generators[0].height, self.generators[0])(sketch_images)

        disc_outputs, gen_outputs, real_images_resized = [], [], []

        for generator, discriminator in zip(self.generators, self.discriminators):
            resizing_layer = tf.keras.layers.Resizing(generator.height * 2, generator.width * 2)

            generated_images = generator(input_images)
            gen_outputs += [generated_images]
            disc_outputs += [discriminator(generated_images)]
            upsampled_real = resizing_layer(real_images)
            real_images_resized += [upsampled_real]

            generated_images = resizing_layer(generated_images)
            upsampled_sketch = resizing_layer(generator.height * 2, generator.width * 2)(sketch_images)
            input_images = tf.concat(upsampled_sketch, generated_images)
    
        return tf.convert_to_tensor(gen_outputs), tf.convert_to_tensor(disc_outputs), tf.convert_to_tensor(real_images_resized)

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
            last_losses = 0

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
                    gen_outputs, disc_outputs, real_images_resized = self.call(batch_sketch_images, real_images)
                    losses = []
                    metrics_2d = []
                    for gen_output, disc_output, real_image, discriminator in zip(gen_outputs, disc_outputs, real_images_resized, self.discriminators):
                        losses += [self.loss(
                            gen_output, real_image, disc_output, discriminator(real_image))]  # TODO: ADD PARAMS like in losses.py file
                        metrics = []
                        for metric in self.metrics:
                            metrics += [
                                metric(gen_output, real_image)
                            ]  ## TODO: ADJUST params as needed
                        metrics_2d += [metrics]
                        np.asarray(metrics_2d)

                for loss in losses:
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                last_losses += loss
                avg_loss = float(last_losses / (start - end))
                avg_metrics = float(metrics / (start - end))

                print(
                    f"\r[Epoch: {e} \t Batch Index: {index+1}/{num_batches}]\t batch_loss={avg_loss:.3f}\t batch_metrics: {avg_acc:.3f}\t",
                    end="",
                )

        return avg_loss, avg_metrics

    def test(self, real_images, sketch_images):
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
