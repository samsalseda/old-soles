import tensorflow as tf


#Useful resource: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def distance_loss(generated_image, real_image):
    """
    Returns the L2 distance loss between the real image and the generated image. 

    """
    return tf.reduce_sum(tf.pow(real_image - generated_image, 2))


def adversarial_loss(disc_output_generated, disc_output_real, is_disc):
    """
    Returns the adversarial loss for the generator or the discriminator, as determined by 
    the is_disc parameter. The generator loss is the mean of log(D(G(x))), while the 
    discriminator loss is the mean of log(D(x)) - log(D(G(x))).
     
    """
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
    """
    Computes the gram matrix of an input tensor. Gram(X) = X_TX, where X_T is the transpose
    of X. 

    """
    b, h, w, ch = x.shape
    f = tf.reshape(x, (b, ch, w * h))
    f_T = tf.transpose(f, [0, 2, 1])
    G = tf.matmul(f, f_T) / (h * w * ch)

    return G

def style_loss(generated_image, real_image):
    """
    Returns the L2 distance loss between the Gram matrices of the generated image and the
    real image. 
    """
    gram_x = compute_gram(generated_image)
    gram_y = compute_gram(real_image)
    return tf.reduce_sum(tf.pow(gram_y - gram_x, 2))


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
    """
    Returns the total loss for the generator. This loss is made up of the distance_loss, 
    adversarial_loss and style_loss, each weighted by an inputted coefficient (lambda_d, 
    lambda_a and lambda_s, respectively).
    """
    loss = lambda_d * distance_loss(generated_image, real_image)
    loss += lambda_a * adversarial_loss(disc_output_generated, disc_output_real, False)
    loss += lambda_s * style_loss(generated_image, real_image)

    return loss
