"""Utilities for converting TFHub BigGAN generator weights to PyTorch.

Recommended usage:

To convert all BigGAN variants and generate test samples, use:

```bash
CUDA_VISIBLE_DEVICES=0 python converter.py --generate_samples
```

See `parse_args` for additional options.
"""

import os
import argparse

import h5py
import torch
from torchvision.utils import save_image
import tensorflow as tf
import tensorflow_hub as hub

import biggan

DEVICE = 'cuda'
HDF5_TMPL = 'biggan-{}.h5'
PTH_TMPL = 'biggan-{}.pth'
MODULE_PATH_TMPL = 'https://tfhub.dev/deepmind/biggan-{}/2'
Z_DIMS = {
    128: 120,
    256: 140,
    512: 128}
RESOLUTIONS = list(Z_DIMS)


def dump_tfhub_to_hdf5(module_path, hdf5_path, redownload=False):
    """Loads TFHub weights and saves them to intermediate HDF5 file.

    Args:
        module_path ([Path-like]): Path to TFHub module.
        hdf5_path ([Path-like]): Path to output HDF5 file.

    Returns:
        [h5py.File]: Loaded hdf5 file containing module weights.
    """
    if os.path.exists(hdf5_path) and (not redownload):
        print('Loading BigGAN hdf5 file from:', hdf5_path)
        return h5py.File(hdf5_path, 'r')

    print('Loading BigGAN module from:', module_path)
    tf.reset_default_graph()
    hub.Module(module_path)
    print('Loaded BigGAN module from:', module_path)

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)

    print('Saving BigGAN weights to :', hdf5_path)
    h5f = h5py.File(hdf5_path, 'w')
    for var in tf.global_variables():
        val = sess.run(var)
        h5f.create_dataset(var.name, data=val)
        print(f'Saving {var.name} with shape {val.shape}')
    h5f.close()
    return h5py.File(hdf5_path, 'r')


class TFHub2Pytorch(object):

    TF_ROOT = 'module'

    NUM_GBLOCK = {
        128: 5,
        256: 6,
        512: 7
    }

    w = 'w'
    b = 'b'
    u = 'u0'
    v = 'u1'
    gamma = 'gamma'
    beta = 'beta'

    def __init__(self, state_dict, tf_weights, resolution=256, load_ema=True, verbose=False):
        self.state_dict = state_dict
        self.tf_weights = tf_weights
        self.resolution = resolution
        self.verbose = verbose
        if load_ema:
            for name in ['w', 'b', 'gamma', 'beta']:
                setattr(self, name, getattr(self, name) + '/ema_b999900')

    def load(self):
        self.load_generator()
        return self.state_dict

    def load_generator(self):
        GENERATOR_ROOT = os.path.join(self.TF_ROOT, 'Generator')

        for i in range(self.NUM_GBLOCK[self.resolution]):
            name_tf = os.path.join(GENERATOR_ROOT, 'GBlock')
            name_tf += f'_{i}' if i != 0 else ''
            self.load_GBlock(f'GBlock.{i}.', name_tf)

        self.load_attention('attention.', os.path.join(GENERATOR_ROOT, 'attention'))
        self.load_linear('linear', os.path.join(self.TF_ROOT, 'linear'), bias=False)
        self.load_snlinear('G_linear', os.path.join(GENERATOR_ROOT, 'G_Z', 'G_linear'))
        self.load_colorize('colorize', os.path.join(GENERATOR_ROOT, 'conv_2d'))
        self.load_ScaledCrossReplicaBNs('ScaledCrossReplicaBN',
                                        os.path.join(GENERATOR_ROOT, 'ScaledCrossReplicaBN'))

    def load_linear(self, name_pth, name_tf, bias=True):
        self.state_dict[name_pth + '.weight'] = self.load_tf_tensor(name_tf, self.w).permute(1, 0)
        if bias:
            self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_snlinear(self, name_pth, name_tf, bias=True):
        self.state_dict[name_pth + '.module.weight_u'] = self.load_tf_tensor(name_tf, self.u).squeeze()
        self.state_dict[name_pth + '.module.weight_v'] = self.load_tf_tensor(name_tf, self.v).squeeze()
        self.state_dict[name_pth + '.module.weight_bar'] = self.load_tf_tensor(name_tf, self.w).permute(1, 0)
        if bias:
            self.state_dict[name_pth + '.module.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_colorize(self, name_pth, name_tf):
        self.load_snconv(name_pth, name_tf)

    def load_GBlock(self, name_pth, name_tf):
        self.load_convs(name_pth, name_tf)
        self.load_HyperBNs(name_pth, name_tf)

    def load_convs(self, name_pth, name_tf):
        self.load_snconv(name_pth + 'conv0', os.path.join(name_tf, 'conv0'))
        self.load_snconv(name_pth + 'conv1', os.path.join(name_tf, 'conv1'))
        self.load_snconv(name_pth + 'conv_sc', os.path.join(name_tf, 'conv_sc'))

    def load_snconv(self, name_pth, name_tf, bias=True):
        if self.verbose:
            print(f'loading: {name_pth} from {name_tf}')
        self.state_dict[name_pth + '.module.weight_u'] = self.load_tf_tensor(name_tf, self.u).squeeze()
        self.state_dict[name_pth + '.module.weight_v'] = self.load_tf_tensor(name_tf, self.v).squeeze()
        self.state_dict[name_pth + '.module.weight_bar'] = self.load_tf_tensor(name_tf, self.w).permute(3, 2, 0, 1)
        if bias:
            self.state_dict[name_pth + '.module.bias'] = self.load_tf_tensor(name_tf, self.b).squeeze()

    def load_conv(self, name_pth, name_tf, bias=True):

        self.state_dict[name_pth + '.weight_u'] = self.load_tf_tensor(name_tf, self.u).squeeze()
        self.state_dict[name_pth + '.weight_v'] = self.load_tf_tensor(name_tf, self.v).squeeze()
        self.state_dict[name_pth + '.weight_bar'] = self.load_tf_tensor(name_tf, self.w).permute(3, 2, 0, 1)
        if bias:
            self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_HyperBNs(self, name_pth, name_tf):
        self.load_HyperBN(name_pth + 'HyperBN', os.path.join(name_tf, 'HyperBN'))
        self.load_HyperBN(name_pth + 'HyperBN_1', os.path.join(name_tf, 'HyperBN_1'))

    def load_ScaledCrossReplicaBNs(self, name_pth, name_tf):
        self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.beta).squeeze()
        self.state_dict[name_pth + '.weight'] = self.load_tf_tensor(name_tf, self.gamma).squeeze()
        self.state_dict[name_pth + '.running_mean'] = self.load_tf_tensor(name_tf + 'bn', 'accumulated_mean')
        self.state_dict[name_pth + '.running_var'] = self.load_tf_tensor(name_tf + 'bn', 'accumulated_var')
        self.state_dict[name_pth + '.num_batches_tracked'] = torch.tensor(
            self.tf_weights[os.path.join(name_tf + 'bn', 'accumulation_counter:0')][()], dtype=torch.float32)

    def load_HyperBN(self, name_pth, name_tf):
        if self.verbose:
            print(f'loading: {name_pth} from {name_tf}')
        beta = name_pth + '.beta_embed.module'
        gamma = name_pth + '.gamma_embed.module'
        self.state_dict[beta + '.weight_u'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'), self.u).squeeze()
        self.state_dict[gamma + '.weight_u'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'), self.u).squeeze()
        self.state_dict[beta + '.weight_v'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'), self.v).squeeze()
        self.state_dict[gamma + '.weight_v'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'), self.v).squeeze()
        self.state_dict[beta + '.weight_bar'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'), self.w).permute(1, 0)
        self.state_dict[gamma +
                        '.weight_bar'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'), self.w).permute(1, 0)

        cr_bn_name = name_tf.replace('HyperBN', 'CrossReplicaBN')
        self.state_dict[name_pth + '.bn.running_mean'] = self.load_tf_tensor(cr_bn_name, 'accumulated_mean')
        self.state_dict[name_pth + '.bn.running_var'] = self.load_tf_tensor(cr_bn_name, 'accumulated_var')
        self.state_dict[name_pth + '.bn.num_batches_tracked'] = torch.tensor(
            self.tf_weights[os.path.join(cr_bn_name, 'accumulation_counter:0')][()], dtype=torch.float32)

    def load_attention(self, name_pth, name_tf):

        self.load_snconv(name_pth + 'theta', os.path.join(name_tf, 'theta'), bias=False)
        self.load_snconv(name_pth + 'phi', os.path.join(name_tf, 'phi'), bias=False)
        self.load_snconv(name_pth + 'g', os.path.join(name_tf, 'g'), bias=False)
        self.load_snconv(name_pth + 'o_conv', os.path.join(name_tf, 'o_conv'), bias=False)
        self.state_dict[name_pth + 'gamma'] = self.load_tf_tensor(name_tf, self.gamma)

    def load_tf_tensor(self, prefix, var, device='0'):
        name = os.path.join(prefix, var) + f':{device}'
        return torch.from_numpy(self.tf_weights[name][:])


def convert_biggan(resolution, weight_dir, redownload=False, no_ema=False, verbose=False):
    module_path = MODULE_PATH_TMPL.format(resolution)
    hdf5_path = os.path.join(weight_dir, HDF5_TMPL.format(resolution))
    pth_path = os.path.join(weight_dir, PTH_TMPL.format(resolution))

    tf_weights = dump_tfhub_to_hdf5(module_path, hdf5_path, redownload=redownload)
    G = getattr(biggan, f'Generator{resolution}')()
    state_dict = G.state_dict()

    converter = TFHub2Pytorch(state_dict, tf_weights, resolution=resolution,
                              load_ema=(not no_ema), verbose=verbose)
    state_dict = converter.load()
    G.load_state_dict(state_dict)
    torch.save(state_dict, pth_path)
    return G


def generate_sample(G, z_dim, batch_size, filename, parallel=False):
    G.eval()
    G.to(DEVICE)
    if parallel:
      G = torch.nn.DataParallel(G)
    with torch.no_grad():
        label = torch.LongTensor(batch_size, 1).random_() % 1000
        oh = torch.zeros(batch_size, 1000).scatter_(1, label.long(), 1).to(DEVICE)
        z = torch.tensor(biggan.truncated_z_sample(batch_size, z_dim), dtype=torch.float32).to(DEVICE)
        images = G(z, oh)
        save_image(biggan.denorm(images), filename, scale_each=True, normalize=True)
    # if full_sheets:
      # import utils
      # config = {'dataset': 'I128', 'parallel': parallel, 'samples_root': '/home/s1580274/scratch/samples/', 'itr': 0}
      # G.shared = lambda label: torch.zeros(label.shape[0], 1000).to(label.device).scatter_(1, label.long().view(-1,1), 1)#.to(DEVICE)
      # G.dim_z = z_dim
      # import utils
      # utils.sample_sheet(G,
                           # classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']], 
                           # num_classes=utils.nclass_dict[config['dataset']], 
                           # samples_per_class=10, parallel=config['parallel'],
                           # samples_root=config['samples_root'], 
                           # experiment_name='TFHUBW',
                           # folder_number=-1,
                           # )

def parse_args():
    usage = 'Parser for conversion script.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        '--resolution', '-r', type=int, default=None, choices=[128, 256, 512],
        help='Resolution of TFHub module to convert. Converts all resolutions if None.')
    parser.add_argument(
        '--redownload', action='store_true', default=False,
        help='Redownload weights and overwrite current hdf5 file, if present.')
    parser.add_argument(
        '--weights_dir', type=str, default='pretrained_weights')
    parser.add_argument(
        '--samples_dir', type=str, default='pretrained_samples')
    parser.add_argument(
        '--no_ema', action='store_true', default=False,
        help='Do not load ema weights.')
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='Additionally logging.')
    parser.add_argument(
        '--generate_samples', action='store_true', default=False,
        help='Generate test sample with pretrained model.')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size used for test sample.')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Parallelize G?')       
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    if args.resolution is not None:
        G = convert_biggan(args.resolution, args.weights_dir,
                           redownload=args.redownload,
                           no_ema=args.no_ema, verbose=args.verbose)
        if args.generate_samples:
            filename = os.path.join(args.samples_dir, f'biggan{args.resolution}_samples.jpg')
            generate_sample(G, Z_DIMS[args.resolution], args.batch_size, filename, args.parallel)
    else:
        for res in RESOLUTIONS:
            G = convert_biggan(res, args.weights_dir,
                               redownload=args.redownload,
                               no_ema=args.no_ema, verbose=args.verbose)
            if args.generate_samples:
                filename = os.path.join(args.samples_dir, f'biggan{res}_samples.jpg')
                generate_sample(G, Z_DIMS[res], args.batch_size, filename, args.parallel)