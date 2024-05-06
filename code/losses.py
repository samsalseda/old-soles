import tensorflow as tf


#Useful resource: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def distance_loss(generated_image, real_image):
    return tf.reduce_sum(tf.pow(real_image - generated_image, 2))


def adversarial_loss(disc_output_generated, disc_output_real, is_disc):
    
    if is_disc:
        return tf.reduce_mean(
            tf.math.log(tf.clip_by_value(disc_output_real, 0.001, 10000000))
        ) + tf.reduce_mean(
            tf.math.log(tf.clip_by_value(1 - disc_output_generated, 0.001, 10000000))
        )
    else:
        return tf.reduce_mean(
            tf.math.log(tf.clip_by_value(disc_output_generated, 0.001, 10000000))
        )

def compute_gram(x):
    b, h, w, ch = x.shape
    f = tf.reshape(x, (b, ch, w * h))
    f_T = tf.transpose(f, [0, 2, 1])
    G = tf.matmul(f, f_T) / (h * w * ch)

    return G

def style_loss(generated_image, real_image):
    gram_x = compute_gram(generated_image)
    gram_y = compute_gram(real_image)
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
    loss += lambda_a * adversarial_loss(disc_output_generated, disc_output_real, False)
    loss += lambda_s * style_loss(generated_image, real_image)
    # loss += lambda_p * perceptual_loss()

    return loss
