import tensorflow as tf

def distance_loss(generated_image, real_image):
    return tf.reduce_sum(real_image - generated_image)

def adversarial_loss(disc_output_generated, disc_output_real):
    return tf.reduce_mean(tf.math.log(disc_output_real)) + tf.reduce_mean(tf.math.log(1 - disc_output_generated))

def style_loss(generated_image, real_image):
    gram_x = tf.matmul(generated_image, generated_image, transpose_a=True)
    gram_y = tf.matmul(real_image, real_image, transpose_a=True)
    return tf.reduce_sum(gram_y - gram_x)

#def perceptual_loss():
    #TODO: look into VGG model

def total_loss(self, generated_image, real_image, disc_output_generated, disc_output_real, lambda_d, lambda_a, lambda_s, lambda_p):
    loss = lambda_d * self.distance_loss(generated_image, real_image)
    loss += lambda_a * self.adversarial_loss(disc_output_generated, disc_output_real)
    loss += lambda_s * self.style_loss(generated_image, real_image)
    #loss += lambda_p * self.perceptual_loss()

    return loss