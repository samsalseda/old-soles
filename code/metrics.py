import tensorflow as tf


# https://www.tensorflow.org/api_docs/python/tf/image/psnr
def SSIM(generated_image, real_image):
    return tf.image.ssim(generated_image, real_image, max_val=1)


# https://www.tensorflow.org/api_docs/python/tf/image/ssim
def PSNR(generated_image, real_image):
    return tf.image.psnr(generated_image, real_image, max_val=1)


# Found this implementation of FID
# https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
# https://github.com/tsc2017/Frechet-Inception-Distance

# inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg')

# def compute_embeddings(dataloader, count):
#     image_embeddings = []


#     for _ in tqdm(range(count)):
#         images = next(iter(dataloader))
#         embeddings = inception_model.predict(images)


#         image_embeddings.extend(embeddings)


#     return np.array(image_embeddings)

#  def calculate_fid(real_embeddings, generated_embeddings):
#      # calculate mean and covariance statistics
#      mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
#      mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
#      # calculate sum squared difference between means
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = linalg.sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if np.iscomplexobj(covmean):
#        covmean = covmean.real
#      # calculate score
#      fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#      return fid
