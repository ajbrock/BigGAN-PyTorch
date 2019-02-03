import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


# Architectures for G
# New strategy: pass in attention values as G_attn='64_128' then
# attention[int(item)] = True for item in G_attn.split('_')
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

def l2normalize(v, eps=1e-4):
  return v / (v.norm() + eps)


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
  return truncation * values


def denorm(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)


class SpectralNorm(nn.Module):
  def __init__(self, module, name='weight', power_iterations=1):
    super(SpectralNorm, self).__init__()
    self.module = module
    self.name = name
    self.power_iterations = power_iterations
    if not self._made_params():
      self._make_params()

  def _update_u_v(self):
    u = getattr(self.module, self.name + "_u")
    v = getattr(self.module, self.name + "_v")
    w = getattr(self.module, self.name + "_bar")

    height = w.data.shape[0]
    _w = w.view(height, -1)
    for _ in range(self.power_iterations):
      v = l2normalize(torch.matmul(_w.t().detach(), u))
      u = l2normalize(torch.matmul(_w.detach(), v))

    sigma = u.dot((_w).mv(v))
    setattr(self.module, self.name, w / sigma.expand_as(w))

  def _made_params(self):
    try:
      getattr(self.module, self.name + "_u")
      getattr(self.module, self.name + "_v")
      getattr(self.module, self.name + "_bar")
      return True
    except AttributeError:
      return False

  def _make_params(self):
    w = getattr(self.module, self.name)

    height = w.data.shape[0]
    width = w.view(height, -1).data.shape[1]

    #u = P(w.data.new(height).normal_(0, 1), requires_grad=False)
    #v = P(w.data.new(height).normal_(0, 1), requires_grad=False)
    self.module.register_buffer(self.name + '_u', w.data.new(height).normal_(0, 1))
    self.module.register_buffer(self.name + '_v', w.data.new(width).normal_(0, 1))
    getattr(self.module, self.name + "_u").data = l2normalize(getattr(self.module, self.name + "_u").data)
    getattr(self.module, self.name + "_v").data = l2normalize(getattr(self.module, self.name + "_v").data)
    #u.data = l2normalize(u.data)
    #v.data = l2normalize(v.data)
    w_bar = P(w.data)

    del self.module._parameters[self.name]
    #self.module.register_parameter(self.name + "_u", u)
    #self.module.register_parameter(self.name + "_v", v)
    self.module.register_parameter(self.name + "_bar", w_bar)

  def forward(self, *args):
    self._update_u_v()
    return self.module.forward(*args)


class SelfAttention(nn.Module):
  """ Self Attention Layer"""

  def __init__(self, in_dim, activation=F.relu):
    super().__init__()
    self.chanel_in = in_dim
    self.activation = activation

    self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
    self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
    self.pool = nn.MaxPool2d(2, 2)
    self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False))
    self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False))
    self.gamma = nn.Parameter(torch.zeros(1))

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    m_batchsize, C, width, height = x.size()
    N = height * width

    theta = self.theta(x)
    phi = self.phi(x)
    phi = self.pool(phi)
    phi = phi.view(m_batchsize, -1, N // 4)
    theta = theta.view(m_batchsize, -1, N)
    theta = theta.permute(0, 2, 1)
    attention = self.softmax(torch.bmm(theta, phi))
    g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
    attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize, -1, width, height)
    out = self.o_conv(attn_g)
    return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
    self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class GBlock(nn.Module):
  def __init__(
    self,
    in_channel,
    out_channel,
    kernel_size=[3, 3],
    padding=1,
    stride=1,
    n_class=None,
    bn=True,
    activation=F.relu,
    upsample=True,
    downsample=False,
    z_dim=148,
  ):
    super().__init__()

    self.conv0 = SpectralNorm(
      nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
    )
    self.conv1 = SpectralNorm(
      nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True)
    )

    self.skip_proj = False
    if in_channel != out_channel or upsample or downsample:
      self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
      self.skip_proj = True

    self.upsample = upsample
    self.downsample = downsample
    self.activation = activation
    self.bn = bn
    if bn:
      self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
      self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)

  def forward(self, input, condition=None):
    out = input

    if self.bn:
      out = self.HyperBN(out, condition)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, scale_factor=2)
    out = self.conv0(out)
    if self.bn:
      out = self.HyperBN_1(out, condition)
    out = self.activation(out)
    out = self.conv1(out)

    if self.downsample:
      out = F.avg_pool2d(out, 2)

    if self.skip_proj:
      skip = input
      if self.upsample:
        skip = F.interpolate(skip, scale_factor=2)
      skip = self.conv_sc(skip)
      if self.downsample:
        skip = F.avg_pool2d(skip, 2)
    else:
      skip = input
    return out + skip


class Generator(nn.Module):
  def __init__(self, n_classes=1000, G_ch=64, G_shared=True,dim_z=128,
               G_lr=1e-4, G_B1=0.0, G_B2=0.999, init='xavier',G_attn='0',
               G_activation=nn.ReLU(inplace=False),
               **kwargs):
    super().__init__()

    self.ch = G_ch
    self.G_shared = True
    self.init = init
    self.dim_z = dim_z
    self.shared = nn.Embedding(n_classes, 128)#, bias=False)
    self.activation = G_activation

    self.first_view = 16 * self.ch

    self.G_linear = SpectralNorm(nn.Linear(dim_z, 4 * 4 * 16 * self.ch))

    #z_dim = code_dim + 28

    self.GBlock = nn.ModuleList([
      GBlock(16 * self.ch, 16 * self.ch, n_class=n_classes, z_dim=dim_z, activation=self.activation),
      GBlock(16 * self.ch, 8 * self.ch,  n_class=n_classes, z_dim=dim_z, activation=self.activation),
      GBlock(8 * self.ch, 4 * self.ch,   n_class=n_classes, z_dim=dim_z, activation=self.activation),
      GBlock(4 * self.ch, 2 * self.ch,   n_class=n_classes, z_dim=dim_z, activation=self.activation),
      GBlock(2 * self.ch, 1 * self.ch,   n_class=n_classes, z_dim=dim_z, activation=self.activation),
    ])

    self.sa_id = 3
    #self.num_split = len(self.GBlock) + 1
    self.G_attn = G_attn
    if '64' in self.G_attn:
      self.attention = SelfAttention(2 * self.ch)
    else:
      self.attention = None

    self.ScaledCrossReplicaBN = BatchNorm2d(1 * self.ch, eps=1e-4)
    self.colorize = SpectralNorm(nn.Conv2d(1 * self.ch, 3, [3, 3], padding=1))

    self.init_weights()
    self.lr, self.B1, self.B2 = G_lr, G_B1, G_B2
    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0)
  # Initialize
  def init_weights(self):
    # print('Skipping initialization for now...')
    self.param_count = 0
    for module in self.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        # pass # Uncomment this to skip init step on convs and linears
        if self.init == 'ortho':
          if hasattr(module, 'weight_bar'):
            init.orthogonal_(module.weight_bar)
          else:
            init.orthogonal_(module.weight)
        elif self.init == 'N02':
          if hasattr(module, 'weight_bar'):
            init.normal_(module.weight_bar, 0, 0.02)
          else:
            init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          if hasattr(module, 'weight_bar'):
            init.xavier_uniform_(module.weight_bar)
          else:
            init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
      #else:
        #print(type(module))
    print('Param count for G''s initialized parameters: %d' % self.param_count)
  def forward(self, z, y):
    #codes = torch.chunk(input, self.num_split, 1)
    #class_emb = self.linear(class_id)  # 128

    # out = self.G_linear(codes[0])
    out = self.G_linear(z)
    #out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
    out = out.view(-1, self.first_view, 4, 4)
    #for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
    for i, GBlock in enumerate(self.GBlock):
      #condition = torch.cat([code, class_emb], 1)
      out = GBlock(out, y)
      if i == self.sa_id and self.attention is not None:
        out = self.attention(out)

    out = self.ScaledCrossReplicaBN(out)
    out = F.relu(out)
    out = self.colorize(out)
    return torch.tanh(out)

# class Generator(nn.Module):
  # def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               # G_kernel_size=3, G_attn='64', n_classes=1000,
               # num_G_SVs=1, num_G_SV_itrs=1,
               # G_shared=True, shared_dim=None, hier=False,
               # cross_replica=False,
               # G_activation=nn.ReLU(inplace=False),
               # G_lr=5e-5, G_B1=0.0, G_B2=0.999,
               # G_init='ortho', G_param='SN', norm_style='bn',
               # **kwargs):
    # super(Generator, self).__init__()
    # # Channel width mulitplier
    # self.ch = G_ch
    # # Dimensionality of the latent space
    # self.dim_z = dim_z
    # # The initial spatial dimensions
    # self.bottom_width = bottom_width
    # # Resolution of the output
    # self.resolution = resolution
    # # Kernel size?
    # self.kernel_size = G_kernel_size
    # # Attention?
    # self.attention = G_attn
    # # number of classes, for use in categorical conditional generation
    # self.n_classes = n_classes
    # # Use shared embeddings?
    # self.G_shared = G_shared
    # # Dimensionality of the shared embedding? Unused if not using G_shared
    # self.shared_dim = shared_dim if shared_dim else dim_z
    # # Hierarchical latent space?
    # self.hier = hier
    # # Cross replica batchnorm?
    # self.cross_replica = cross_replica
    # # nonlinearity for residual blocks
    # self.activation = G_activation
    # # Initialization style
    # self.init = G_init
    # # Parameterization style
    # self.G_param = G_param
    # # Normalization style
    # self.norm_style = norm_style

    # # Architecture dict
    # self.arch = G_arch(self.ch, self.attention)[resolution]

    # # If using hierarchical latents, adjust z
    # if self.hier:
      # self.num_slots = len(self.arch['in_channels']) + 1 # Number of places z slots into
      # self.z_chunk_size = (self.dim_z // self.num_slots)
      # self.dim_z = self.z_chunk_size *  self.num_slots # Recalculate latent dimensionality for even splitting into chunks
    # else:
      # self.num_slots = 1
      # self.z_chunk_size = 0

    # # Which convs, batchnorms, and linear layers to use
    # if self.G_param == 'SN':
      # self.which_conv = functools.partial(layers.SNConv2d,
                          # kernel_size=3, padding=1,
                          # num_svs=num_G_SVs, num_itrs=num_G_SV_itrs)
      # self.which_linear = functools.partial(layers.SNLinear,
                          # num_svs=num_G_SVs, num_itrs=num_G_SV_itrs)
      # self.which_embedding = functools.partial(layers.SNEmbedding,
                              # num_svs=num_G_SVs, num_itrs=num_G_SV_itrs)
    # # PyTorch inbuilt spectral norm? Use lambdas here since functools.partial doesn't quite cut it
    # elif self.G_param == 'PTSN':
      # self.which_conv = lambda *args, **kwargs: nn.utils.spectral_norm(functools.partial(nn.Conv2d, kernel_size=3, padding=1)(*args, **kwargs))
      # self.which_linear = lambda *args, **kwargs: nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
      # self.which_embedding = lambda *args, **kwargs: nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))
      # # self.which_linear = lambda in_ch, out_ch: nn.utils.spectral_norm(nn.Linear(in_ch, out_ch))
      # # self.which_embedding = lambda num_embeddings, embedding_size: nn.utils.spectral_norm(nn.Embedding(num_embeddings, embedding_size))
    # else:
      # self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      # self.which_linear = nn.Linear
      # self.which_embedding = nn.Embedding

    # bn_linear = functools.partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding
    # self.which_bn = functools.partial(layers.ccbn,
                          # which_linear=bn_linear,
                          # cross_replica=self.cross_replica,
                          # input_size=self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes,
                          # norm_style=self.norm_style)


    # # Prepare model
    # # If not using shared embeddings, self.shared is just a passthrough
    # self.shared = self.which_embedding(n_classes, self.shared_dim) if G_shared else layers.identity()
    # # First linear layer
    # self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    # self.arch['in_channels'][0] * (self.bottom_width **2))

    # # self.blocks is a doubly-nested list of modules, the outer loop intended
    # # to be over blocks at a given resolution (resblocks and/or self-attention)
    # self.blocks = []
    # for index in range(len(self.arch['out_channels'])):
      # self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             # out_channels=self.arch['out_channels'][index],
                             # which_conv=self.which_conv,
                             # which_bn=self.which_bn,
                             # activation=self.activation,
                             # upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       # if self.arch['upsample'][index] else None))]]

      # # If attention on this block, attach it to the end
      # if self.arch['attention'][self.arch['resolution'][index]]:
        # print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        # self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # # Turn self.blocks into a ModuleList so that it's all properly registered.
    # self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # # output layer: batchnorm-relu-conv. Optionally use an nn.BatchNorm2d, or
    # # use a non-spectral conv, by toggling these comments.
    # self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                # cross_replica=self.cross_replica),
                                    # # nn.BatchNorm2d(self.arch['out_channels'][-1] * self.ch,affine=True),
                                    # self.activation,
                                    # #nn.Conv2d(self.arch['out_channels'][-1] * self.ch, 3, 3,padding=1))
                                    # self.which_conv(self.arch['out_channels'][-1], 3)) # Consider using a non-spectral conv here

    # # Initialize weights
    # self.init_weights()

    # # Set up optimizer
    # self.lr, self.B1, self.B2 = G_lr, G_B1, G_B2
    # self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           # betas=(self.B1, self.B2), weight_decay=0)

    # # LR scheduling, left here for forward compatibility
    # # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # # self.j = 0



  # # Notes on this forward function: we pass in a y vector which has
  # # already been passed through G.shared to enable easy class-wise
  # # interpolation later. If we passed in the one-hot and then ran it through
  # # G.shared in this forward function, it would be harder to handle.
  # def forward(self, z, y):

    # # If hierarchical, concatenate zs and ys
    # if self.hier:
      # zs = torch.split(z, self.z_chunk_size, 1)
      # z = zs[0]
      # ys = [torch.cat([y, item], 1) for item in zs[1:]]
    # else:
      # ys = [y] * len(self.blocks)
    # # First linear layer
    # h = self.linear(z)
    # # Reshape
    # h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    # # Loop over blocks
    # for index, blocklist in enumerate(self.blocks):
      # # Second inner loop in case block has multiple layers
      # for block in blocklist:
        # h = block(h, ys[index])
    # # Apply batchnorm-relu-conv-tanh at output
    # return torch.tanh(self.output_layer(h))


# Discriminator
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999,
               D_param='SN', output_dim=1,
               D_init='ortho', **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)

    # PyTorch inbuilt spectral norm? Use lambdas here since functools.partial doesn't quite cut it
    elif self.D_param == 'PTSN':
      self.which_conv = lambda *args, **kwargs: nn.utils.spectral_norm(functools.partial(nn.Conv2d, kernel_size=3, padding=1)(*args, **kwargs))
      self.which_linear = lambda *args, **kwargs: nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
      self.which_embedding = lambda *args, **kwargs: nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    # Embedding for projection discrimination
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2 = D_lr, D_B1, D_B2
    self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(D_B1, D_B2), weight_decay=0)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    # print('Skipping initialization for now...')
    self.param_count = 0
    for module in self.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        # pass # Uncomment this to skip init step on convs and linears
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
      #else:
        #print(type(module))
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):
    # If training G, enable grad
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))

    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension
    if split_D:
      D_fake = self.D(G_z, gy)
      if x is not None:
        D_real = self.D(x, dy)
        return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    # If real data is provided, concatenate it with the Generator's output
    # along the batch dimension for improved efficiency.
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      # Get Discriminator output
      D_out = self.D(D_input, D_class)
      if x is not None:
        return torch.split(D_out, [G_z.shape[0], x.shape[0]]) # D_fake, D_real
      else:
        if return_G_z:
          return D_out, G_z
        else:
          return D_out
