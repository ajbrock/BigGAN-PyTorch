#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
Andy's Notes: Need to properly credit things based on where we got them.
To do: enable more in-depth validation splits
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import json
from argparse import ArgumentParser
import animal_hash

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets as dset

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128',
    help='Which Dataset to train on, out of I128, I256, C10, C100;'
         'Append "_hdf5" to use the hdf5 version for ISLVRC (default: %(default)s)')
  parser.add_argument(
    '--model', type=str, default='model',
    help='Name of the model module, to permit multiple disparate files if desired (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--hashname', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config (default: %(default)s)') 
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers; consider using less for HDF5 (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)')
  parser.add_argument(
    '--num_epochs', type=int, default=100,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--test_every', type=int, default=5000,
    help='Test every X iterations (default: %(default)s)')
  parser.add_argument(
    '--G_eval_mode', action='store_true', default=False,
    help='Run G in eval mode (running stats?) at save+sample / testtime? (default: %(default)s)')
  parser.add_argument(
    '--save_every', type=int, default=2000,
    help='Save every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='How many copies to save (default: %(default)s)')
  parser.add_argument(
    '--num_best_copies', type=int, default=2,
    help='How many previous best checkpoints to save (default: %(default)s)')
  parser.add_argument(
    '--which_best', type=str, default='IS',
    help='Which metric to use to determine when to save new "best"'
         'checkpoints, one of IS or FID (default: %(default)s)')
  parser.add_argument(
    '--no_fid', action='store_true', default=False,
    help='Calculate IS only, not FID? (default: %(default)s)')
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  parser.add_argument(
    '--base_root', type=str, default='',
    help='Default location to store all weights, samples, data, and logs (default: %(default)s)')
  parser.add_argument(
    '--dataset_root', type=str, default='/home/s1580274/scratch/data/',
    help='Default location where data is stored (default: %(default)s)')    
  parser.add_argument(
    '--weights_root', type=str, default='/home/s1580274/scratch/weights',
    help='Default location to store weights (default: %(default)s)') 
  parser.add_argument(
    '--logs_root', type=str, default='/home/s1580274/scratch/logs',
    help='Default location to store logs (default: %(default)s)')
  parser.add_argument(
    '--samples_root', type=str, default='/home/s1580274/scratch/samples',
    help='Default location to store samples (default: %(default)s)')  
  parser.add_argument(
    '--weights_suffix', type=str, default='',
    help='Suffix for experiment name for loading weights for sampling (consider "best0") (default: %(default)s)')  
  parser.add_argument(
    '--pbar', type=str, default='mine',
    help='Type of progressbar to use; one of "mine" or "tqdm" (default: %(default)s)')
  parser.add_argument(
    '--G_ch', type=int, default=64,
    help='Channel multiplier for G default: %(default)s)')
  parser.add_argument(
    '--D_ch', type=int, default=64,
    help='Channel multiplier for D default: %(default)s)')
  parser.add_argument(
    '--G_shared', action='store_true', default=False,
    help='Use shared embeddings in G? (default: %(default)s)')
  parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')  
  parser.add_argument(
    '--hier', action='store_true', default=False,
    help='Use hierarchical z in G? (default: %(default)s)')
  parser.add_argument(
    '--ema', action='store_true', default=False,
    help='Keep an ema of G''s weights? (default: %(default)s)')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=False,
    help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
  parser.add_argument(
    '--ema_start', type=int, default=0,
    help='When to start updating the EMA weights (default: %(default)s)')
  parser.add_argument(
    '--load_weights', action='store_true', default=False,
    help='Load pretrained weights? (default: %(default)s)')
  parser.add_argument(
    '--cross_replica', action='store_true', default=False,
    help='Cross_replica batchnorm in G?(default: %(default)s)')
  parser.add_argument(
    '--num_G_accumulations', type=int, default=1,
    help='Number of passes to accumulate G''s gradients over (default: %(default)s)')
  parser.add_argument(
    '--G_batch_size', type=int, default=0,
    help='Batch size to use for G; if 0, same as D (default: %(default)s)')
  parser.add_argument(
    '--num_D_steps', type=int, default=2,
    help='Number of D steps per G step (default: %(default)s)')
  parser.add_argument(
    '--num_D_accumulations', type=int, default=1,
    help='Number of passes to accumulate D''s gradients over (default: %(default)s)')
  parser.add_argument(
    '--merge_step', action='store_true', default=False,
    help='Use merge_step during training? (default: %(default)s)')
  parser.add_argument(
    '--split_D', action='store_true', default=False,
    help='Run D twice rather than concatenating inputs? (default: %(default)s)')
  parser.add_argument(
    '--G_lr', type=float, default=5e-5,
    help='Learning rate to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_lr', type=float, default=2e-4,
    help='Learning rate to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B1', type=float, default=0.0,
    help='Beta1 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B1', type=float, default=0.0,
    help='Beta1 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B2', type=float, default=0.999,
    help='Beta2 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B2', type=float, default=0.999,
    help='Beta2 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--num_G_SVs', type=int, default=1,
    help='Number of SVs to track in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SVs', type=int, default=1,
    help='Number of SVs to track in D (default: %(default)s)')
  parser.add_argument(
    '--num_G_SV_itrs', type=int, default=1,
    help='Number of SV itrs in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SV_itrs', type=int, default=1,
    help='Number of SV itrs in D (default: %(default)s)')
  parser.add_argument(
    '--G_nl', type=str, default='relu',
    help='Activation function for G (default: %(default)s)')
  parser.add_argument(
    '--D_nl', type=str, default='relu',
    help='Activation function for D (default: %(default)s)')
  parser.add_argument(
    '--G_attn', type=str, default='64',
    help='What resolutions to use attention on for G (underscore separated) (default: %(default)s)')
  parser.add_argument(
    '--D_attn', type=str, default='64',
    help='What resolutions to use attention on for D (underscore separated) (default: %(default)s)')
  parser.add_argument(
    '--G_init', type=str, default='ortho',
    help='Init style to use for G (default: %(default)s)')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Init style to use for D(default: %(default)s)')
  parser.add_argument(
    '--G_param', type=str, default='SN',
    help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)(default: %(default)s)')
  parser.add_argument(
    '--D_param', type=str, default='SN',
    help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)(default: %(default)s)')
  parser.add_argument(
    '--G_ortho', type=float, default=0,#1e-4, # 1e-4 should be default for BigGAN
    help='Modified ortho reg (default: %(default)s)')
  parser.add_argument(
    '--D_ortho', type=float, default=0.0,
    help='Modified ortho reg strength in D (default: %(default)s)')
  parser.add_argument(
    '--norm_style', type=str, default='bn',
    help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], ln [layernorm], gn [groupnorm] (default: %(default)s)')
  parser.add_argument(
    '--which_train_fn', type=str, default='GAN',
    help='How2trainyourbois (default: %(default)s)')
  parser.add_argument(
    '--load_in_mem', action='store_true', default=False,
    help='Load all data into memory? default: %(default)s)')  
  parser.add_argument(
    '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
    help='Pin data into memory through dataloader? (default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  return parser


# Convenience dicts
dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder, 'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
             'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5, 'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
             'C10': dset.CIFAR10, 'C100': dset.CIFAR100}
imsize_dict = {'I32': 32, 'I32_hdf5': 32, 
               'I64': 64, 'I64_hdf5': 64, 
               'I128': 128, 'I128_hdf5': 128, 
               'I256': 256, 'I256_hdf5': 256,
               'C10': 32, 'C100': 32}
root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
             'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'C10': 'cifar', 'C100': 'cifar'}
nclass_dict = {'I32': 1000, 'I32_hdf5': 1000,
               'I64': 1000, 'I64_hdf5': 1000, 
               'I128': 1000, 'I128_hdf5': 1000,  
               'I256': 1000, 'I256_hdf5': 1000,
               'C10': 10, 'C100': 100}
classes_per_sheet_dict = {'I32': 50, 'I32_hdf5': 50,
                          'I64': 50, 'I64_hdf5': 50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'C10': 10, 'C100': 100}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}

class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__
        
class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = 0 if size[0] == img.size[0] else np.random.randint(low=0,high=img.size[0] - size[0])
    j = 0 if size[1] == img.size[1] else np.random.randint(low=0,high=img.size[1] - size[1])    
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__

# Convenience function to centralize all data loaders
def get_data_loaders(dataset, dataset_root=None, augment=False, batch_size=64, num_workers=8,
                     shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, **kwargs):
                     
  # Test which cluster we're on and select a root appropriately
  if dataset_root is None:
    if dataset in ['C10', 'C100']:
      print('Using CIFAR dataset root in scratch...')
      if os.path.isdir('/home/s1580274/scratch/data/'):
        dataset_root = '/home/s1580274/scratch/data/'
        print('On Eddie, using the eddie root location %s...' % dataset_root)
      elif os.path.isdir('/home/visionlab/andy/boilerplate'):
        dataset_root = '/home/visionlab/andy/boilerplate/'
        print('On Nessie, using Nessie root location %s...' % dataset_root)
    elif os.path.isdir('/home/s1580274/scratch/data/'):
      dataset_root = '/home/s1580274/scratch/data/'
      print('On Eddie, using the eddie root location %s...' % dataset_root)
    elif os.path.isdir('/jmain01/home/JAD003/sxr01/axb64-sxr01/data/'):
      dataset_root = '/jmain01/home/JAD003/sxr01/axb64-sxr01/data/'
      print('On Jade, using the Jade root location %s...' % dataset_root)
    elif os.path.isdir('/home/abrock/imagenet/train_imgs'):
      dataset_root = '/home/abrock/imagenet/train_imgs'
      print('On Robotarium, using the Robotarium root location %s...' % dataset_root)
    else:
      print('No root directories found!')
    
  # Append /FILENAME.hdf5 to root if using hdf5
  dataset_root += '%s' % root_dict[dataset]
  print('Using dataset root location %s' % dataset_root)
    
    
  which_dataset = dset_dict[dataset]
  norm_mean = [0.5,0.5,0.5]
  norm_std = [0.5,0.5,0.5]
  image_size = imsize_dict[dataset]
  
  # HDF5 datasets have their own inbuilt transform
  if 'hdf5' in dataset:
    train_transform = None
  else:
    if augment:
      print('Data will be augmented...')
      if dataset in ['C10', 'C100']:
        train_transform = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()]
      else:
        train_transform = [RandomCropLongEdge(),
                         transforms.Resize(image_size),
                         transforms.RandomHorizontalFlip()]
    else:
      print('Data will not be augmented...')
      if dataset in ['C10', 'C100']:
        train_transform = []
      else:
        train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]
      # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
    train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])
  train_set = which_dataset(root=dataset_root, transform=train_transform,
                            load_in_mem=load_in_mem)
  
  # Prepare loader; the loaders list is for forward compatibility with
  # using validation / test splits.
  loaders = []
  loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                   'drop_last': drop_last} # By default drop last incomplete batch
  train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=shuffle, **loader_kwargs)
  loaders.append(train_loader)                           
  return loaders

# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  
# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
  for key in ['weights_root', 'logs_root', 'samples_root']:
    if not os.path.exists(config[key]):
      print('Making directory %s for %s...' % (config[key], key))
      os.mkdir(config[key])
# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict, self.target_dict = self.source.state_dict(), self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!
        
  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay + self.source_dict[key].data * (1 - decay))

        
# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = 2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w)
      param.grad.data += strength * grad.view(param.shape)
      
# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or param in blacklist:
        continue
      w = param.view(param.shape[0], -1)
      grad = 2 * torch.mm(torch.mm(w, w.t()) - torch.eye(w.shape[0], device=w.device), w)
      param.grad.data += strength * grad.view(param.shape)

 # Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, state_dict, weights_root, experiment_name, G_ema=None):
  root = '/'.join([weights_root, experiment_name])
  print('Saving weights to %s...' % root)
  torch.save(G.state_dict(), '%s_G.pth' % root)
  torch.save(G.optim.state_dict(), '%s_G_optim.pth' % root)
  torch.save(D.state_dict(), '%s_D.pth' % root)
  torch.save(D.optim.state_dict(), '%s_D_optim.pth' % root)
  torch.save(state_dict, '%s_state_dict.pth' % root)
  if G_ema is not None:
    torch.save(G_ema.state_dict(), '%s_G_ema.pth' % root) 
      
# load weights
def load_weights(G, D, state_dict, weights_root, experiment_name, G_ema=None):
  root = '/'.join([weights_root, experiment_name])
  print('Loading weights from %s...' % root )
  if G is not None:
    G.load_state_dict(torch.load('%s_G.pth' % root))
    G.optim.load_state_dict(torch.load('%s_G_optim.pth' % root))
  if D is not None:
    D.load_state_dict(torch.load('%s_D.pth' % root))
    D.optim.load_state_dict(torch.load('%s_D_optim.pth' % root))
  # Load state dict
  for item in state_dict:
    state_dict[item] = torch.load('%s_state_dict.pth' % root)[item]
  if G_ema is not None:
    G_ema.load_state_dict(torch.load('%s_G_ema.pth' % root))

''' MetricsLogger originally stolen from VoxNet source code.'''
class MetricsLogger(object):
  def __init__(self, fname, reinitialize=False):
    self.fname = fname
    self.reinitialize = reinitialize
    if os.path.exists(self.fname):
      if self.reinitialize:
        print('{} exists, not deleting...'.format(self.fname))
        #os.path.remove(self.fname)

  def log(self, record=None, **kwargs):
    """
    Assumption: no newlines in the input.
    """
    if record is None:
      record = {}
    record.update(kwargs)
    record['_stamp'] = time.time()
    with open(self.fname, 'a') as f:
      f.write(json.dumps(record, ensure_ascii=True) + '\n')

class MyLogger(object):
  def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
    self.root = fname
    self.reinitialize = reinitialize
    self.metrics = []
    self.logstyle = logstyle # One of '%3.3f' or like '%3.3e'
    #for suffix in 'D_loss_real', 'D_loss_fake', 'G_loss', 'recon', 'kl':
  # Delete log if re-starting and log already exists 
  def reinit(self, item):
    if os.path.exists('%s_%s.log' % (self.root, item)):
      if self.reinitialize:
        print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
        os.remove('%s_%s.log' % (self.root, item))
  # Log in plaintext; this is designed for being read/plotted in matlab (sorry not sorry)
  def log(self, itr, **kwargs):
    for arg in kwargs:
      if arg not in self.metrics:
        self.reinit(arg)
        self.metrics += [arg]
      with open('%s_%s.log' % (self.root, arg), 'a') as f:
        f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))
  
"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA.
"""
def progress(items, desc='', total=None, min_delay=0.1):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                    desc, n+1, total, n / float(total) * 100), end=" ")
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))), end=" ")
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))
                                                 

# Sample function for use with inception metrics
def sample(G, z_, y_, config):
  with torch.no_grad():
    z_.normal_()
    y_.random_(0, nclass_dict[config['dataset']])
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (z_, G.shared(y_)))
    else:
      G_z = G(z_, G.shared(y_))    
    return G_z, y_


# Sample function for sample sheets
#samples_root = '/home/s1580274/scratch/samples'
# ideally batch_size should line up with samples_per_class and num_classes
# in some convenient way
# e.g. 10 samples_per_class, 20 sheets, 1000 classes, yields 50 classes per sheet
# and 500 images per sheet, so a batch size of 20 or 100 would be good.
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel, 
                 samples_root, experiment_name, folder_number):
  # Prepare sample directory
  if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
    os.mkdir('%s/%s' % (samples_root, experiment_name))
  if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
    os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))    
  # loop over total number of sheets
  for i in range(num_classes // classes_per_sheet):
    ims = []
    y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
    for j in range(samples_per_class):
      z = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
      with torch.no_grad():
        if parallel:
          o = nn.parallel.data_parallel(G, (z, G.shared(y)))
        else:
          o = G(z, G.shared(y))
        
      ims += [o]
    # This line should properly unroll the images
    out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2], ims[0].shape[3]).data.cpu()
    # The path for the samples    
    image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name, folder_number, i) 
    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=samples_per_class, normalize=True)
                                 
  
# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
  lerp_step = 1. / (num_midpoints + 1)
  lerp = torch.arange(0, 1 + lerp_step, lerp_step).cuda()
  return ((x0 * (1 - lerp.view(1,-1,1))) + (x1 * lerp.view(1,-1,1)))
 
 
# interp sheet function
# Supports class-wise and intra-class interpolation
# Andy's note: Haven't tested this in a while so I don't remember if it works or not, it may need some updating.
# Interps are low priority at the moment, as is documenting this function.
def interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
                 samples_root, experiment_name, folder_number, sheet_number=0,
                 fix_z=False, fix_y=False):
  # Prepare zs and ys  
  if fix_z: # If fix Z, only sample 1 z per row
    zs = torch.randn(num_per_sheet, 1, G.dim_z, device='cuda')
    zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
  else:
    zs = interp(torch.randn(num_per_sheet, 1, G.dim_z, device='cuda'),
                torch.randn(num_per_sheet, 1, G.dim_z, device='cuda'),
                num_midpoints).view(-1, G.dim_z)
  if fix_y: # If fix y, only sample 1 z per row
    ys = sample_1hot(num_per_sheet, num_classes)
    ys = G.shared(ys).view(num_per_sheet, 1, -1)
    ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
  else:  
    ys = interp(G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
  with torch.no_grad():
    if parallel:
      out_ims = nn.parallel.data_parallel(G, (zs, ys))
    else:
      out_ims = G(zs, ys)
  interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
  image_filename = '%s/%s/%d/interp%s%d.jpg' % (samples_root, experiment_name, 
                                                folder_number, interp_style,
                                                sheet_number) 
  torchvision.utils.save_image(out_ims, image_filename,
                               nrow=samples_per_class, normalize=True)

                               
# Convenience debugging function to print out gradnorms and shape from each layer
# May need to rewrite this so we can actually see which parameter is which
def print_grad_norms(net):
    gradsums = [[float(torch.norm(param.grad).cpu()), float(torch.norm(param).cpu()), param.shape]  for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    #['%3.3e,%3.3e, %s' % (float(torch.abs(param.grad).sum().cpu()), float(torch.abs(param).sum().cpu()), str(param.shape)) for param in G.parameters()]
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0], gradsums[item_index][1], str(gradsums[item_index][2])) for item_index in order])

    
# Name an experiment based on its config
def name_from_config(config):
  name = '_'.join([
  item for item in [
  'Big%s' % config['which_train_fn'],
  config['dataset'],
  'seed%d' % config['seed'],
  'Gch%d' % config['G_ch'],
  'Dch%d' % config['D_ch'],
  'bs%d' % config['batch_size'],
  'nDs%d' % config['num_D_steps'] if config['num_D_steps'] > 1 else None,
  'nDa%d' % config['num_D_accumulations'] if config['num_D_accumulations'] > 1 else None,
  'nGa%d' % config['num_G_accumulations'] if config['num_G_accumulations'] > 1 else None,
  'Glr%2.1e' % config['G_lr'],
  'Dlr%2.1e' % config['D_lr'],
  'GB%3.3f' % config['G_B1'] if config['G_B1'] !=0.0 else None,
  'GBB%3.3f' % config['G_B2'] if config['G_B2'] !=0.999 else None,
  'DB%3.3f' % config['D_B1'] if config['D_B1'] !=0.0 else None,
  'DBB%3.3f' % config['D_B2'] if config['D_B2'] !=0.999 else None,
  'Gnl%s' % config['G_nl'],
  'Dnl%s' % config['D_nl'],
  'Ginit%s' % config['G_init'],
  'Dinit%s' % config['D_init'],
  'G%s' % config['G_param'] if config['G_param'] != 'SN' else None,
  'D%s' % config['D_param'] if config['D_param'] != 'SN' else None,
  'Gattn%s' % config['G_attn'] if config['G_attn'] is not '0' else None,
  'Dattn%s' % config['D_attn'] if config['D_attn'] is not '0' else None,
  'Gortho%2.1e' % config['G_ortho'] if config['G_ortho'] > 0.0 else None,
  'Dortho%2.1e' % config['D_ortho'] if config['D_ortho'] > 0.0 else None,
  config['norm_style'] if config['norm_style'] != 'bn' else None,  
  'cr' if config['cross_replica'] else None,
  'Gshared' if config['G_shared'] else None,
  'hier' if config['hier'] else None,
  'ema' if config['ema'] else None,
  config['weights_suffix'] if config['weights_suffix'] else None,  
  ]
  if item is not None])
  # Instead of a d
  if config['hashname']:
    return hashname(name)
  else:
    return name

    
# A simple function to produce a unique experiment name from the animal hashes.
def hashname(name):
  h = hash(name)
  a = h % len(animal_hash.a)
  h = h // len(animal_hash.a)
  b = h % len(animal_hash.b)
  h = h // len(animal_hash.c)
  c = h % len(animal_hash.c)
  return animal_hash.a[a] + animal_hash.b[b] + animal_hash.c[c] 
  
  
# Convenience function to count the number of parameters
def count_parameters(module):
  print('Number of parameters: {}'.format(
    sum([p.data.nelement() for p in module.parameters()])))
  
  
# Convenience function to sample an index, not actually a 1-hot 
# def sample_1hot(batch_size, num_classes, device='cuda'):
  # return torch.randint(low=0, high=num_classes, size=(batch_size,),
          # device=device, dtype=torch.int64, requires_grad=False)
       
       
# Convenience function to prepare a z and y vector       
def prepare_zy(G_batch_size, dim_z, nclasses, device='cuda'):
  z_ = torch.randn(G_batch_size, dim_z, requires_grad=False, device=device)
  y_ = torch.randint(low=0, high=nclasses, 
                     size=(G_batch_size,), device=device, 
                     dtype=torch.int64, requires_grad=False)
  return z_, y_
