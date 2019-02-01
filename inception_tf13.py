# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
# Run on eddie with /home/s1580274/group/myconda/tensorflow/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
# import scipy.misc
import math
import sys
from tqdm import tqdm, trange

MODEL_DIR = '/home/s1580274/scratch'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Inception with TF1.3 or earlier.
# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 500
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in trange(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()
  
# ims = np.load('GAN_D40_K4_C100_seed13_hinge_n_dis5_proj_ccbn_Gnl_relu_Dnl_relu_100epochs_G_save1_samples1.npz')['x']
# fname = 'GAN_D40_K4_C100_seed13_hinge_n_dis5_proj_ccbn_Gnl_relu_Dnl_relu_100epochs_G_samples.npz'
# fname = 'GAN_D40_K4_C100_seed49_hinge_n_dis5_proj_cat_embed_up_z_ccbn_shared_Gnl_relu_Dnl_relu_800epochs'
# fname='GAN_D40_K4_C10_seed49_hinge_n_dis5_proj_ccbn_shared_Gnl_relu_Dnl_relu_205epochs'
# fname = 'GAN_stn_D40_K4_C100_seed0_hinge_bs64_n_dis5_proj_vae_embed_kl_up_z_ccbn_shared_Gnl_relu_Dnl_relu_300epochs_G_best'
#fname='/home/s1580274/scratch/GAN_2wide_D40_K4_I128_seed0_hinge_bs128_n_dis5_proj_vae_embed_up_z_ccbn_shared_Gnl_relu_Dnl_relu_100epochs_G_best'
#fname = '/home/s1580274/scratch/samples/BigGAN_I32_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gshared_BPT/samples.npz'
fname = '/home/s1580274/scratch/samples/BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0/samples.npz'
print('loading %s ...'%fname)
ims = np.load(fname)['x']# + '_samples.npz')['x']
# ims = 
import time
t0 = time.time()
inc = get_inception_score(list(ims.swapaxes(1,2).swapaxes(2,3)), splits=10)
t1 = time.time()
print('Inception took %3f seconds, score of %3f +/- %3f.'%(t1-t0, inc[0], inc[1]))