""" BigGAN: The Authorized Unofficial PyTorch release
    A. Brock
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    The original code is written in TensorFlow and deeply integrated with
    the Google infrastructure, such that open sourcing the training code would
    require an inordinate amount of time and effort. This project is my own
    personal catharsis, and in some ways is research into whether it's possible
    to replicate the results (my own results) with an order of magnitude
    less compute (128 TPU for a 128x128 model; I'm going to be trying on
    4xTitan X).

    Differences between this code and the original code are primarily in small
    details (I'm coding off the top of my head and without the original code
    as reference, so we'll see how it goes). The main change is that this code
    relies on gradient accumulation to spoof large minibatches (megabatches?).
    This is, however, not precisely equivalent, as we employ cross-replica
    BatchNorm, meaning the activation moments will be calculated wrt subsets
    of the megabatch. This code is also not going to have any of the research
    bells and whistles that the original code does--this is designed to be a
    sleek, straightforward, and readable implementation that practitioners can
    use, rely on as a reference, and hopefully build upon. Most of the things
    listed in the negative results appendix of the paper are options still
    available in the internal code, but will not be available here.

    Let's go.
"""

# To-do: optionally rewrite inception metrics to use a single
#        pool and logits array which gets rewritten as opposed to
#        relying on garbage collection to free up the memory.
# +Add standing stats aggregation
# -Write and test EMA
# -Write and test efficient orthogonality
# -Write logging functionality for SVs in a better way than we used to havew
# -Now that we have abunch of tests in test.py, why not wrap them up into
# proper little unit tests? Be explicit in the printout about what it is they
# test (and what they don't)
# -test gradient accumulation things; specifically, can we rescale the loss
#  instead of having to mess with LRs
# -write standing stats
# -get this running on CIFAR first
# -nah, get it running SAGAN first
# -Consider a more in-depth experiment management system with names to allow
#  for easy loading/saving, multiple models, etc.
# -add experiment identifier string--that way we can run multiple experiments
# with the same setting (including seed) and still tell them apart.
# -resolve --ema vs --use_ema options
# -write simple code to port from TF-hub weights to pytorch model, then ensure
#  that it produces the same outputs (write a series of unit tests basically and
#  just compare outputs from one versus the other.
# -add G and D loss stats somewhere (an events file? a csv?) that gets dumped,
#  and output them to the progress bar's postfix
# -figure out if it's the double-batch in D that's taking up so much memory
# or if it's something else. What if we remove all relus and replace them with
# identity function? What's the memory consumption dealio here?
# -write tests for Dct, iDCT, color conversion
# -rewrite to have it be "norm" layer and just specify the norm type for G
# -add init type option?
# -consider changing the downsampling patterns on the CIFAR D
# -test out if spectral norm is actually being used by loading a weightsfile and
#  writing a test for it
# -make sure random seeds are actually being used
# -cudnn.benchmark!
# -simple test: make SN optional in D and G, train without, compare to training with
# -improve progress bar options
# -write code to produce sample .npzs, maybe a "sample.py" script.
# is the issue just that we're using BN in training mode?
# -write SVD parameterization code, paying special attention to column vs row orthogonality
# -write code to log full spectra from the SVD params, and code to load, plot individual SVs, and animate spectra
# -Write parser for filenames that takes in a filename and prints out
# something like "This is model X with G width of X, [attention at X resolutions]/no attention..."
# -test ycbcr by getting the RGB representation of each color channel?
# -Note that when running on DCT dataset we need inception metrics still...
# -Robustify logs to accept arbitrary metrics
# Idea: write a hash (like gfycat) to assign a random adjective-animal-verb name
# to each experiment

# Instructions: first, accumulate inception metrics for your chosen dataset by
# running python calculate_inception_moments.py. This will use the default PyTorch
# inception model to compute the IS of your training data (by default
# ImageNet @ 128x128 pixels) and the feature moments needed for FID.

# Random notes: interp_17 in zipem is good

# -either specify base batch size + num accumulations or overall batch size and num accumulations?
# is_train = False
# >>> with torch.set_grad_enabled(is_train):
# ...     y = x * 2
# >>> y.requires_grad
# False

# Idea:
# let accumulated means and vars hang around in cross-rep bns and don't worry about it
# accumulations make things weird
#

# Design notes: In this code I pass around the full config dict a lot by
# matching up kwargs in the functions they're used in. This slightly reduces
# readability from the main script but makes it easier to change this script
# by reducing the number of places one needs to update the code whenever a
# new config option is added. It also saves space in the main script <_<

# How to use:
# Call . load.sh on eddie first
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

#config: model, load_weights, ema, ema_decays, num_inception_images, inception_batchsize,
# dataset
# EITHER batch_size + num_accumulations OR overall_batch_size + num_accumulations
# seed for inits


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)

  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = utils.name_from_config(config)
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config).cuda()
  D = model.Discriminator(**config).cuda()
  if config['fp16']:
    print('Casting G and D to float16...')
    G, D = G.half(), D.half()
    #print('Actually, only casting G to float16')
    # G = G.half()
    # D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**config).cuda()
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    ema = None

  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['load_weights']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)

    if config['BN_sync']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'], experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, reinitialize=(not config['resume']), logstyle=config['logstyle'])


  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations']
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size})

  # Prepare inception metrics: FID and IS
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device='cuda', fp16=config['fp16'])
  # z_ = torch.randn(G_batch_size, G.dim_z, requires_grad=False).cuda()
  # y_ = torch.randint(low=0, high=config['n_classes'],
                     # size=(G_batch_size,), device='cuda',
                     # dtype=torch.int64, requires_grad=False)

  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config)
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample, G=G_ema if config['ema'] and config['use_ema'] else G,
                             z_=z_, y_=y_, config=config)

  # Prepare save and sample function
  def save_and_sample():
    utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name,
                           G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
      utils.save_weights(G, D, state_dict, config['weights_root'],
                         '%s_copy%d' % (experiment_name, state_dict['save_num']),
                         G_ema if config['ema'] else None)
      state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
      # For now, every time we save, also save sample sheets
      utils.sample_sheet(G_ema if config['ema'] and config['use_ema'] else G,
                         classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                         num_classes=config['n_classes'],
                         samples_per_class=10, parallel=config['parallel'],
                         samples_root=config['samples_root'],
                         experiment_name=experiment_name,
                         folder_number=state_dict['itr'],
                         z_=z_)

  # prepare test function
  def test():
    print('Gathering inception metrics...')
    IS_mean, IS_std, FID = get_inception_metrics(sample, 50000, num_splits=10)
    print('Itr %d: Inception Score is %3.3f +/- %3.3f, FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
      or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
      print('%s improved over previous best, saving checkpoint...' % config['which_best'])
      utils.save_weights(G, D, state_dict, config['weights_root'],
                         '%s_best%d' % (experiment_name, state_dict['save_best_num']),
                         G_ema if config['ema'] else None)
      state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean), IS_std=float(IS_std), FID=float(FID))

  print('Beginning training...')
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):
    # Increment epoch counter
    state_dict['epoch'] += 1
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0])
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['fp16']:
        x, y = x.cuda().half(), y.cuda()
      else:
        x, y = x.cuda(), y.cuda()
      metrics = train(x, y)
      train_log.log(itr=int(state_dict['itr']), **metrics)

      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      #Could also do this for TQDM using set_postfix.
      if config['pbar'] == 'mine':# and time.time() - t_last > min_delay:
          print(', '.join(['itr: %d' % state_dict['itr']] + ['%s : %+4.3f' % (key, metrics[key]) for key in metrics]), end=' ')
          #t_last = time.time()

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        save_and_sample()

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        test()



def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()