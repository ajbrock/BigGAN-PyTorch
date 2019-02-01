import numpy as np
from scipy import linalg # For FID

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3

# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits
    
# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


# Calculate Inception Score mean + std given softmax'd logits and number of splits
def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)


# Loop and run the sampler and the net until it accumulates num_inception_images
# activations. Return the pool, the logits, and the labels (if one wants 
# incepiton accuracy the labels of the generated class will be needed)
def accumulate_inception_activations(sample, net, num_inception_images=50000):
  pool, logits, labels = [], [], []
  while (np.shape(np.concatenate(logits, 0))[0] if len(logits) else 0) < num_inception_images:
    with torch.no_grad():
      images, labels_val = sample()
      pool_val, logits_val = net(images.float())
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(labels_val.cpu())]
  return np.concatenate(pool, 0), np.concatenate(logits, 0), np.concatenate(labels, 0)
  # inception_pool[i * inception_batch_size : (i + 1) * inception_batch_size] = activation.data.cpu().numpy()
  # inception_predictions[i * inception_batch_size : (i + 1) * inception_batch_size] = y_inception

# Load and wrap the Inception model
def load_inception_net(parallel=False):
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  if parallel:
    print('Parallelizing Inception module...')
    inception_model = nn.DataParallel(inception_model)
  return inception_model


# This produces a function which takes in an iterator which returns a set number of samples
# and iterates until it accumulates config['num_inception_images'] images.
# The iterator can return samples with a different batch size than used in
# training, using the setting confg['inception_batchsize']
def prepare_inception_metrics(dataset, parallel, no_fid=False):
  # Load metrics; this is intentionally not in a try-except loop so that
  # the script will crash here if it cannot find the Inception moments.
  # By default, remove the "hdf5" from dataset
  dataset = dataset.strip('_hdf5')
  data_mu = np.load(dataset+'_inception_moments.npz')['mu']
  data_sigma = np.load(dataset+'_inception_moments.npz')['sigma']
  # Load network
  net = load_inception_net(parallel)
  def get_inception_metrics(sample, num_inception_images, num_splits=10):
    print('Gathering activations...')
    pool, logits, labels = accumulate_inception_activations(sample, net, num_inception_images)
    print('Calculating Inception Score...')
    IS_mean, IS_std = calculate_inception_score(logits, num_splits)
    if no_fid:
      FID = 9999.0
    else:
      print('Calculating means and covariances...')
      mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
      FID = calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
    return IS_mean, IS_std, FID
  return get_inception_metrics