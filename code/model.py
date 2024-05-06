import argparse
import tensorflow as tf
import numpy as np
from GANBlock import Generator, Discriminator
from preprocess_temp import process
from losses import total_loss, adversarial_loss
from metrics import SSIM, PSNR
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
from SaveImage import save_image
>>>>>>> 4f887af82da87e9fa92fe7b6f61df7c932c3c887


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
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="number of epochs over which to train the model",
    )

    if args is None:
        return parser.parse_args()  ## For calling through command line
    return parser.parse_args(args)  ## For calling through notebook.


def main(args):
    # TODO: call prorocessing, build the model
    # TODO: train using the preprocessed image stuff
    # TODO: test using the preprocessed image stuff

    sketches = np.load("code/input_images_train.npy")
    real = np.load("code/target_images_train.npy")

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

    train_loss, train_metrics = model.train(sketches, real, epochs=args.num_epochs)

    print(f"\ntrain_loss: {train_loss}")
    #print(f"accuracy: {metrics}")

    model.test(real, sketches)
    #print(f"\ntest_loss: {test_loss}")
    #print(f"accuracy: {metrics}")

    #model.predict(sketches)


class ShoeGenerationModel(tf.keras.Model):

    def __init__(self, generators, discriminators, **kwargs):
        super().__init__(**kwargs)
        self.generators, self.discriminators = generators, discriminators
        self.optimizer = tf.keras.optimizers.Adam(0.00005)
        self.metric_list = [SSIM, PSNR]
        # TODO: convert parameters to lists, for multile GAN blocks to iterate thorugh

    @tf.function
    def call(self, sketch_images, real_images, is_training=True):
        height, width = self.generators[0].height, self.generators[0].width
        input_images = tf.image.resize(sketch_images, [int(height), int(width)])

        disc_outputs, gen_outputs, real_images_resized, disc_outputs_real = [], [], [], []

        gen_num = 0
        for generator, discriminator in zip(self.generators, self.discriminators):
            resizing_layer = tf.keras.layers.Resizing(
                int(generator.height * 2), int(generator.width * 2)
            )

            # print(generator.height)
            # print(generator.width)

            generated_images = generator(input_images)

            # if not is_training:
            #     for i in range(len(generated_images)):
            #         sess = tf.Session()
            #         with sess.as_default():
            #             img_array = generated_images[i].eval()
            #         save_image(img_array, f"layer {gen_num}, image {i}")
            #     gen_num += 1
            #     print(f"generated images shape: {generated_images.shape}")

            gen_outputs += [generated_images]
            #print(f"generator shape: {generated_images.shape}")
            disc_outputs += [discriminator(generated_images)]
            resizing_layer_images = tf.keras.layers.Resizing(
                int(generator.height), int(generator.width)
            )
            upsampled_real = resizing_layer_images(real_images)
            #print(upsampled_real.shape)
            real_images_resized += [upsampled_real]
            disc_outputs_real += [discriminator(upsampled_real)]

            generated_images = resizing_layer(generated_images)
            upsampled_sketch = resizing_layer(sketch_images)
            input_images = tf.concat([upsampled_sketch, generated_images], axis=-1)

        return gen_outputs, disc_outputs, real_images_resized, disc_outputs_real

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

            indices_unshuffled = tf.range(len(sketch_images))
            indices = tf.random.shuffle(indices_unshuffled)
            train_real_images_shuffled = tf.gather(real_images, indices)
            train_sketch_images_shuffled = tf.gather(sketch_images, indices)

            for index, end in enumerate(
                range(batch_size, len(real_images) + 1, batch_size)
            ):
                start = end - batch_size
                batch_sketch_images = train_sketch_images_shuffled[start:end, :]
                batch_real_images = train_real_images_shuffled[start:end, :]

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_outputs, disc_outputs, real_images_resized, disc_outputs_real = self.call(
                        batch_sketch_images, batch_real_images
                    )
                    generator_losses = 0
                    discriminator_losses = 0
                    metrics_2d = []

                    for gen_output, disc_output, real_image, disc_output_real in zip(
                        gen_outputs,
                        disc_outputs,
                        real_images_resized,
                        disc_outputs_real,
                    ):
                        #print(gen_output.shape)
                        #print(real_image.shape)
                        generator_losses += total_loss(
                                gen_output,
                                real_image,
                                disc_output,
                                disc_output_real,  # TODO: THIS line of discriminator(batch_real_imgages) IS AN ERROR. IT SHOULD BE THE BATCH OF RESIZED IMAGES
                            )
                        
                        discriminator_losses += adversarial_loss(disc_output, disc_output_real, True)
                        
                        metrics = []
                        for metric in self.metric_list:
                            metrics += [
                                metric(gen_output, real_image)
                            ]  ## TODO: ADJUST params as needed
                        metrics_2d += [metrics]
                    metrics_2d = np.asarray(metrics_2d)

                
                gradients = gen_tape.gradient(generator_losses, self.trainable_variables)
                #print(gradients)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

<<<<<<< HEAD
                avg_loss = float(summed_losses / (batch_size))
=======
                gradients = disc_tape.gradient(discriminator_losses, self.trainable_variables)
                #print(gradients)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

                avg_loss = float(generator_losses / (start - end))
>>>>>>> 4f887af82da87e9fa92fe7b6f61df7c932c3c887
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
        avg_gen_loss = avg_disc_loss = 0
        avg_acc = 0

        generator_losses = discriminator_losses = 0
        num_examples = len(real_images)

        metrics_2d = []

        gen_outputs, disc_outputs, real_images_resized, disc_outputs_real = self.call(sketch_images, real_images, is_training=False)
        for gen_output, disc_output, real_image, disc_output_real in zip(
                        gen_outputs,
                        disc_outputs,
                        real_images_resized,
                        disc_outputs_real,
                    ):
            generator_losses += total_loss(
                                gen_output,
                                real_image,
                                disc_output,
                                disc_output_real,  # TODO: THIS line of discriminator(batch_real_imgages) IS AN ERROR. IT SHOULD BE THE BATCH OF RESIZED IMAGES
                            )
                        
            discriminator_losses += adversarial_loss(disc_output, disc_output_real, True)
            
            metrics = []
            for metric in self.metric_list:
                metrics += [
                    metric(gen_output, real_image)
                ]  ## TODO: ADJUST params as needed
            metrics_2d += [metrics]
        metrics_2d = np.asarray(metrics_2d)

        avg_gen_loss = float(generator_losses / num_examples)
        avg_disc_loss = float(discriminator_losses/ num_examples)
        #avg_metrics = float(metrics_2d / num_examples)

        print(
            f"\r[Testing: gen_loss={avg_gen_loss:.3f}\t disc_loss={avg_disc_loss:.3f}\t metrics: {avg_acc:.3f}\t",
            end="",
        )

        return avg_gen_loss, avg_disc_loss#, avg_metrics

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
