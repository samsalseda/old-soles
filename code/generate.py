import numpy as np
import tensorflow as tf
from code.modelr import ShoeGenerationModel
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Define the class with image generation functionality
class ShoeGenerationModelWithImageGeneration():
    def preprocess_sketches(self, sketches):
        # Preprocess your sketches as needed (e.g., normalization)
        preprocessed_sketches = sketches / 255.0  # Normalize pixel values
        return preprocessed_sketches

    def generate_images(self, sketches):
        # Generate images from sketches using the loaded model
        generated_images = self.predict(sketches)
        return generated_images

    def postprocess_images(self, images):
        # Postprocess generated images as needed (e.g., denormalization)
        postprocessed_images = (images * 255.0).astype(np.uint8)  # Denormalize pixel values
        return postprocessed_images

# # Load the saved model
# # model = tf.keras.models.load_model("weights/my_model.keras")
# model = tf.keras.models.load_model("weights/my_model.h5")
# print("red" + model)
# # Display the model summary
# # model.summary()

# # # Optionally, you can also plot the model architecture
# # plot_model(model, to_file='model_summary.png', show_shapes=True)

# # Create an instance of the extended model class
# extended_model = ShoeGenerationModelWithImageGeneration(model)

# # Example usage
# sketches = np.load('data/target_images_train.npy')
# sketches = extended_model.preprocess_sketches(sketches)
# generated_images = extended_model.generate_images(sketches)
# generated_images = extended_model.postprocess_images(generated_images)

# # Visualize or save the generated images
# plt.figure(figsize=(10, 10))
# for i in range(min(9, generated_images.shape[0])):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(generated_images[i])
#     plt.axis('off')
# plt.show()
