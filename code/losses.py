import tensorflow as tf


#Useful resource: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def distance_loss(generated_image, real_image):
    return tf.reduce_sum(tf.pow(real_image - generated_image, 2))


def adversarial_loss(disc_output_generated, disc_output_real):
    return tf.reduce_mean(
        tf.math.log(tf.clip_by_value(disc_output_real, 0.001, 10000000))
    ) + tf.reduce_mean(
        tf.math.log(tf.clip_by_value(1 - disc_output_generated, 0.001, 10000000))
    )


def style_loss(generated_image, real_image):
    gram_x = tf.matmul(generated_image, generated_image, transpose_a=True)
    gram_y = tf.matmul(real_image, real_image, transpose_a=True)
    return tf.reduce_sum(tf.pow(gram_y - gram_x, 2))


# def perceptual_loss():
# TODO: look into VGG model


def total_loss(
    generated_image,
    real_image,
    disc_output_generated,
    disc_output_real,
    lambda_d=1.5,
    lambda_a=10.0,
    lambda_s=0.1,
    lambda_p=250.0,
):
    loss = lambda_d * distance_loss(generated_image, real_image)
    loss += lambda_a * adversarial_loss(disc_output_generated, disc_output_real)
    loss += lambda_s * style_loss(generated_image, real_image)
    # loss += lambda_p * perceptual_loss()

    return loss
