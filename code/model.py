import argparse
import tensorflow as tf
import numpy as np
from GANBlock import Generator, Discriminator
from preprocess_temp import process
from losses import total_loss
from metrics import SSIM, PSNR
import matplotlib.pyplot as plt


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
        default=4,
        help="number of GAN Blocks in the model",
    )
    if args is None:
        return parser.parse_args()  ## For calling through command line
    return parser.parse_args(args)  ## For calling through notebook.


def main(args):
    # TODO: call prorocessing, build the model
    # TODO: train using the preprocessed image stuff
    # TODO: test using the preprocessed image stuff

    sketches, real = process()
    sketches = tf.convert_to_tensor(sketches, dtype=tf.float64)
    real = tf.convert_to_tensor(real, dtype=tf.float64)

    generators = []
    discriminators = []
    res = 256
    if args.num_blocks > 4 or args.num_blocks < 1:
        print(
            "Too many GAN Blocks, must be less or equal to 4 and greater than or equal to 1"
        )
        return

    for _ in range(args.num_blocks):

        generators += [Generator([256, res, res])]
        discriminators += [Discriminator(res * res)]

        res /= 2

    generators.reverse()
    discriminators.reverse()

    model = ShoeGenerationModel(generators, discriminators)

    loss, metrics = model.train(sketches, real)

    print(f"loss: {loss}")
    print(f"accuracy: {metrics}")


class ShoeGenerationModel(tf.keras.Model):

    def __init__(self, generators, discriminators, **kwargs):
        super().__init__(**kwargs)
        self.generators, self.discriminators = generators, discriminators
        self.optimizer = tf.keras.optimizers.Adam(0.00005)
        self.metric_list = [SSIM, PSNR]
        # TODO: convert parameters to lists, for multile GAN blocks to iterate thorugh

    @tf.function
    def call(self, sketch_images, real_images):
        height, width = self.generators[0].height, self.generators[0].width
        input_images = tf.image.resize(sketch_images, [int(height), int(width)])

        disc_outputs, gen_outputs, real_images_resized = [], [], []

        for generator, discriminator in zip(self.generators, self.discriminators):
            resizing_layer = tf.keras.layers.Resizing(
                int(generator.height * 2), int(generator.width * 2)
            )

            generated_images = generator(input_images)
            #print(f"generated images shape: {generated_images.shape}")

            gen_outputs += [generated_images]
            disc_outputs += [discriminator(generated_images)]
            resizing_layer_images = tf.keras.layers.Resizing(
                int(generator.height), int(generator.width)
            )
            upsampled_real = resizing_layer_images(real_images)
            real_images_resized += [upsampled_real]

            generated_images = resizing_layer(generated_images)
            upsampled_sketch = resizing_layer(sketch_images)
            input_images = tf.concat([upsampled_sketch, generated_images], axis=-1)

        return gen_outputs, disc_outputs, real_images_resized

    def compile(self, optimizer, loss, metrics):
        """
        Create a facade to mimic normal keras fit routine
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metric_list = metrics

    def train(self, real_images, sketch_images, batch_size=5, epochs=10):
        """
        Runs through all Epochs and trains
        """
        epochs_np = np.array([list(range(epochs))])
        losses_np = np.array([])
        for e in range(epochs):
            avg_loss = 0
            avg_acc = 0

            num_batches = int(len(sketch_images) / batch_size)
            last_losses = 0
            avg_metrics = avg_acc = avg_loss = 0

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
                    gen_outputs, disc_outputs, real_images_resized = self.call(
                        batch_sketch_images, batch_real_images
                    )
                    summed_losses = 0
                    metrics_2d = []
                    for gen_output, disc_output, real_image, discriminator in zip(
                        gen_outputs,
                        disc_outputs,
                        real_images_resized,
                        self.discriminators,
                    ):
                        #print(gen_output.shape)
                        #print(real_image.shape)
                        summed_losses += total_loss(
                                gen_output,
                                real_image,
                                disc_output,
                                discriminator(
                                    real_image
                                ),  # TODO: THIS line of discriminator(batch_real_imgages) IS AN ERROR. IT SHOULD BE THE BATCH OF RESIZED IMAGES
                            )
                        
                        metrics = []
                        for metric in self.metric_list:
                            metrics += [
                                metric(gen_output, real_image)
                            ]  ## TODO: ADJUST params as needed
                        metrics_2d += [metrics]
                    metrics_2d = np.asarray(metrics_2d)

                
                gradients = tape.gradient(summed_losses, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

                avg_loss = float(summed_losses / (batch_size))
                avg_metrics = metrics_2d / (start - end)

                print(f"\r[Epoch: {e} \t Batch Index: {index+1}/{num_batches}]\t batch_loss={avg_loss:.3f}\t batch_metrics: ",
                    end="",)
                
                #print(avg_metrics)
                
                losses_np = np.append(losses_np, avg_loss)
                #print("losses_np: " + str(losses_np))

        print(losses_np)
        print(epochs_np)
        plt.plot(epochs_np.reshape(-1),losses_np)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
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
        for metric in self.metric_list:
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
