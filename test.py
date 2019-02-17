""" Test spectral norm """
import numpy as np
import torch
import layers
num_svs=4
# Uncomment to test conv layers
l = layers.SNConv2D(16,64,3, num_svs=num_svs)
## torch.nn.init.orthogonal_(l.base_weight) # Uncomment to see what's up with ortho init
x = torch.randn(7,16,24,24)
## Uncomment to test linear layers
# l = layers.SNLinear(16,64,num_svs=num_svs)
# x = torch.randn(7,16)
## Uncomment to test embeddings
# l = layers.SNEmbedding(16,64,num_svs=num_svs)
# x = torch.tensor([0,1,2,3,4,5,6,7])
y = l(x)
# u,s,v = np.linalg.svd(l.base_weight.view(64,-1).detach().numpy(),compute_uv=True)
u,s,v = np.linalg.svd(l.base_weight.view(16,-1).detach().numpy(),compute_uv=True)
print(torch.dot(torch.tensor(u[:,0]),l.u[0][0]))
print(torch.dot(l.u[1][0],l.u[0][0]))
d = []
for i in range(100):
  y = l(x)
  d += [torch.stack([torch.dot(torch.tensor(u[:,i]),l.u[i][0]) for i in range(num_svs)])]

# layers.power_iteration(l.base_weight.view(l.base_weight.size(0), -1), l.u)
# w = l.base_weight.view(64,-1).detach()
# [torch.dot(torch.tensor(u[:,0]),w[:,i])  for i in range(144)]
""" Test SN in parallel """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
num_svs=4
l = torch.nn.DataParallel(layers.SNConv2D(16,64,3, num_svs=num_svs).cuda())
xx = torch.randn(7,16,24,24)
x = torch.cat([xx]*4,0).cuda()

y = l(x)
u,s,v = np.linalg.svd(l.module.base_weight.view(64,-1).detach().cpu().numpy(),compute_uv=True)
print(torch.dot(torch.tensor(u[:,0]),l.module.u[0][0].cpu()))
print(torch.dot(l.module.u[1][0],l.module.u[0][0]))
d = []
for i in trange(100):
  y = l(x)
  d += [torch.stack([torch.dot(torch.tensor(u[:,i]),l.module.u[i][0].cpu()) for i in range(num_svs)])]

  
""" Test attention """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
import torch.nn.functional as F
l = layers.Attention(64)
x = torch.randn(7,64,24,24)
y = l(x)
# self=l
# theta = self.theta(x)
# phi = F.max_pool2d(self.phi(x), [2,2])
# g = F.max_pool2d(self.g(x), [2,2])
# theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
# phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
# g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
# beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
# o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))

# # In TF:
# theta = bs, location_num, num_ch / 8
# phi = batch_size, ds_location_num, num_ch / 8
# theta dot phi, transpose_b = true = (bs, location, ch/8) * (bs, ch / 8, ds_location_num) ->  bs, location, ds_location_num

# g = batch_size, downsample, num_ch // 2
# attn_g = attn dot g = (bs, location, ds_location) dot (bs, ds_location, ch/2) -> bs, location, ch / 2
# attn_g .view(batch_size, h, w, num_channels / 2)
# o conv up


""" Test sync batchnorm """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
# import apex

class crbn(nn.Module):
  def __init__(self, num_features, num_devices=4, affine=False, eps=1e-5):
    super(crbn, self).__init__()
    self.num_devices = num_devices
    self.eps = eps
    # Arrays holding means and average-squared vals
    self.ms, self.m2s = [None] * self.num_devices, [None] * self.num_devices
    self.store_complete, self.gather_complete = [False] * self.num_devices, [False] * self.num_devices
  # Reset arrays  
  def reset_arrays(self):
    for i in range(self.num_devices):
      # self.ms[i] = None 
      # self.m2s[i] = None
      self.store_complete[i] = False
      self.gather_complete[i] = False
      
  def forward(self, x):
    print(x.device)
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    num = float(torch.prod(torch.tensor(x.shape)[[0,2,3]])) #.cuda().float()
    m = torch.sum(x, [0, 2, 3], keepdim=True) / num
    m2 = torch.sum(x ** 2, [0, 2, 3], keepdim=True) / num
    # Store means in self arrays
    self.ms[x.device.index] = m
    self.m2s[x.device.index] = m2
    self.store_complete[x.device.index] = True
    # print(len([item is not None for item in self.ms]),len([item is not None for item in self.m2s]))
    # Wait until all arrays have been stored
    print(id(self))
    while any([not item for item in self.store_complete]):
      pass
    # while (len([item is not None for item in self.ms]) < self.num_devices 
      # and len([item is not None for item in self.m2s]) < self.num_devices):
      # pass
    # Gather arrays to this device
    m = nn.parallel.gather(self.ms, x.device.index).mean(0, keepdim=True)
    m2 = nn.parallel.gather(self.m2s, x.device.index).mean(0, keepdim=True)
    # Note that we have successfully gathered
    self.gather_complete[x.device.index] = True
    # On lead device, wait for all to complete and then reset
    if x.device.index == 0:      
      while any([not item for item in self.gather_complete]):
        pass
      print('Resetting...')
      self.reset_arrays()
      # self.ms, self.m2s = [], []
      # for i in range(self.num_devices):
        
        # self.complete[i] = False#False]*4
      # )
    # Calculate variance
    var = m2 - m**2
    print(var.shape, m.shape)
    return (x - m) / torch.sqrt(var + self.eps)
    
x = (torch.randn(28,64,24,24) * (1 + torch.randn(1,64,1,1)) + torch.randn(1,64,1,1))
gain = (torch.randn(28,64,1,1) * (1 + torch.randn(28,64,1,1)) + torch.randn(28,64,1,1))
bias = (torch.randn(28,64,1,1) * (1 + torch.randn(28,64,1,1)) + torch.randn(28,64,1,1))
bn = nn.DataParallel(crbn(64).cuda())
y = bn(x, gain=gain, bias=bias)
y0 = l0.cuda()(x.cuda()) * gain.cuda() + bias.cuda()
# m0 = bn(x)

# class MyMod(nn.Module):
  # def __init__(self, num_features, sync):
    # super(MyMod, self).__init__()
    # if sync:
      # self.bn = apex.parallel.SyncBatchNorm(num_features, affine=False)
    # else:
      # self.bn = nn.BatchNorm2d(num_features, affine=False)
  # def forward(self,x):
    # return self.bn(x)


class MyMod(nn.Module):
  def __init__(self, num_features, sync):
    super(MyMod, self).__init__()
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.results = []
  def forward(self,x):
    out = self.bn(x)
    self.results.append(out)
    return out
  
l0 = MyMod(64, False).cuda()
l1 = nn.DataParallel(MyMod(64, False).cuda()) # non-synchronous
torch.distributed.init_process_group('nccl')
l2 = nn.parallel.DistributedDataParallel(MyMod(64, True).cuda()) # synchronous

# x = torch.randn(28,64,24,24).cuda()
x = (torch.randn(28,64,24,24) * (1 + torch.randn(1,64,1,1)) + torch.randn(1,64,1,1))
# True mean
mm = x.mean(3).mean(2).mean(0)
# True var
vv = x.permute(1,0,2,3).contiguous().view(64,-1).var(1,unbiased=True)#False)


y0a = [l0(x[i * 7 : (i + 1) * 7]) for i in range(4)] # Individually batchnorm'd for refs
y0b = l0(x)
y1 = l1(x)
y2 = l2(x)


""" Test CCBN in layers """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn

which_linear = nn.Embedding#layers.SNEmbedding # If using SN this will have the problem of the spectral norm getting updated all the time
bn = nn.DataParallel(layers.ccbn(64, 100, which_linear, cross_replica=True).cuda())
bn = layers.ccbn(64, 100, which_linear, cross_replica=False)
x = (torch.randn(28,64,24,24) * (1 + torch.randn(1,64,1,1)) + torch.randn(1,64,1,1))
y = torch.arange(28)#.cuda()
z = bn(x, y)
mm = x.mean(3).mean(2).mean(0)
# True var
vv = x.permute(1,0,2,3).contiguous().view(64,-1).var(1,unbiased=True)#False)
# gain, bias = bn.module.gain, bn.module.bias
# y = y.cuda()
gain, bias = bn.gain, bn.bias
z2 = layers.fused_bn(x, mm.view(1,-1,1,1), vv.view(1,-1,1,1), gain=1+ gain(y).cpu().view(28,64,1,1), bias=bias(y).cpu().view(28,64,1,1))

""" Test non-CCBN bn """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
bn = nn.DataParallel(layers.bn(64, cross_replica=True))
x = (torch.randn(28,64,24,24) * (1 + torch.randn(1,64,1,1)) + torch.randn(1,64,1,1))
y2 = bn(x).cpu()
mm = x.mean(3).mean(2).mean(0)
# True var
vv = x.permute(1,0,2,3).contiguous().view(64,-1).var(1,unbiased=False)
z = (x - mm.view(1,-1,1,1))  * torch.rsqrt(vv.view(1,-1,1,1) + 1e-5)

""" Test GBlock """
import numpy as np
import torch
import layers
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
import functools
# which_linear = nn.Embedding#layers.SNEmbedding # If using SN this will have the problem of the spectral norm getting updated all the time
which_linear = layers.SNLinear
block = nn.DataParallel(layers.GBlock(64,128,upsample=nn.Upsample(scale_factor=2),
                      which_conv=functools.partial(layers.SNConv2d, kernel_size=3, padding=1),
                      which_bn=functools.partial(layers.ccbn, which_linear=which_linear, input_size=128, cross_replica=True)).cuda())
# block2 = layers.GBlock(64,128,upsample=nn.Upsample(scale_factor=2),
                      # which_conv=functools.partial(layers.SNConv2d, kernel_size=3, padding=1),
                      # which_bn=functools.partial(layers.ccbn, which_linear=which_linear, input_size=128, cross_replica=False))
x = (torch.randn(28,64,24,24) * (1 + torch.randn(1,64,1,1)) + torch.randn(1,64,1,1))
# y = torch.arange(28)#.cuda()
y = torch.randn(28,128)
z = block(x, y)


# Param counts: ch 64, all tricks: params: G: 31850628, D: 39448258, total 71298886
# Param counts: ch 96, all tricks: params: G: 70305988, D: 87982370, total 158288358 GOOD
# Param counts: ch 64, no tricks:  params: G: 42017412, D: 39448258, total 81465670 GOOD
# Param counts: ch 96, no tricks:  params: G: 85556164, D: 87982370, total 173538534 GOOD
# Param counts: Ch 96, shared, no hier:    G: 72664516, D: 87982370, total 160646886 GOOD
# We expect that for our models, we'll ahve the same D but a G param of 31978628
# Since the above counts don't include the shared embedding.
""" Test generator """
import numpy as np
import torch
import layers
import model
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
import functools
batch_size=7 * 1#torch.cuda.device_count()
device='cuda'
# G = model.Generator(ch=64, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)

G = model.Generator(G_ch=64, hier=True, cross_replica=True,G_shared=True, dim_z=128, shared_dim=128).to(device)
# G = model.Generator(ch=96, hier=False, cross_replica=False, G_shared=False, dim_z=128, shared_dim=128).to(device)
print('Number of params in G: {}'.format(-128000 + sum([p.data.nelement() for p in G.parameters()])))
G = nn.DataParallel(G)
#G.forward = lambda x: nn.data_parallel(G, x)
z = torch.randn(batch_size,128).cuda()
y = utils.sample_1hot(batch_size, 1000,'cuda')#torch.randn(batch_size, 128).cuda()
Gz = G(z, G.shared(y))
import torchvision
samples_root = '/home/s1580274/scratch/samples'
torchvision.utils.save_image(Gz.data.cpu(), '%s/test1.jpg' % samples_root, nrow=torch.cuda.device_count(), normalize=True)


'samples/' + weights_fname + '_' + str(epoch) + '.jpg', nrow=16, normalize=True)

for index in range(30):
  import time
  t1 = time.time()
  x = G(z,y)
  t2 = time.time()
  dt = t2-t1
  print(dt)
x.backward(1)

h = torch.randn(7,256,32,32)
zy = torch.randn(7,149)
xx = G.blocks[3](h,zy)

""" Test discriminator """
# Discriminator's param count works
import model
import torch
device='cuda'
batch_size=28
D = model.Discriminator(D_ch=96).to(device)
print('Number of params in D: {}'.format(sum([p.data.nelement() for p in D.parameters()])))
D = nn.DataParallel(D)
# x = (torch.randn(28,3,128,128) * (1 + torch.randn(1,3,1,1)) + torch.randn(1,3,1,1)).to(device)
y = torch.arange(batch_size).to(device)
dx = D(x, y)
dx.sum().backward(1)

""" test G_D """
import model
import torch
device='cuda'
batch_size=7 * torch.cuda.device_count()
device='cuda'
G = model.Generator(G_ch=96, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)
print('Number of params in G: {}'.format(-128000 + sum([p.data.nelement() for p in G.parameters()])))
D = model.Discriminator(D_ch=96).to(device)
print('Number of params in D: {}'.format(sum([p.data.nelement() for p in D.parameters()])))
GD = model.G_D(G, D)
GD = torch.nn.DataParallel(GD)
gz = torch.randn(batch_size,120).to(device)
gy = torch.arange(batch_size).to(device)#torch.randn(batch_size, 128).to(device)
dx = (torch.randn(batch_size,3,128,128) * (1 + torch.randn(1,3,1,1)) + torch.randn(1,3,1,1)).to(device)
dy = torch.arange(batch_size).to(device)

G_z = nn.parallel.data_parallel(G, (gz, G.shared(dy)))
# D0 = GD(gz,gy)
# D_fake, D_real = GD(gz, gy, dx, dy)

# message for pytorch high command
I'm getting "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation" when I use built-in batchnorm +nn.dataparallel, and  I have no idea why; I basically have
`output_layer = nn.Sequential(nn.BatchNorm2d(num_features), nn.ReLU(inplace=False), nn.Conv2d(num_features,3,3))`
and in forward():
`h = all_previous_layers(input)`
`return output_layer(h)`
in my forward method.
If I remove the batchnorm2d module the error goes away, but I just don't see why it should be affecting anything

""" Test saving"""
import model
import torch
device='cuda'
batch_size=7 * torch.cuda.device_count()
device='cuda'
G = model.Generator(G_ch=96, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)
print('Number of params in G: {}'.format(-128000 + sum([p.data.nelement() for p in G.parameters()])))
D = model.Discriminator(D_ch=96).to(device)
print('Number of params in D: {}'.format(sum([p.data.nelement() for p in D.parameters()])))
GD = model.G_D(G, D)
GD = torch.nn.DataParallel(GD)
D_fake, D_real = GD(gz, gy, dx, dy)
torch.save(G.state_dict())

""" Test EMA """
import torch
import model
import utils
device='cuda'
batch_size=7 * torch.cuda.device_count()
device='cuda'
G = model.Generator(G_ch=96, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)
G_ema = model.Generator(G_ch=96, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)
ema = utils.ema(G, G_ema, 0.9999)
G.blocks[0][0].conv1.state_dict().keys()
q0=G.state_dict()['output_layer.2.base_weight']
q1=G_ema.state_dict()['output_layer.2.base_weight']

""" Test loading ema """
import torch
import model
import utils
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': True, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': False,
          'resolution': 32}
device='cuda'
batch_size=7 * torch.cuda.device_count()
device='cuda'
G = model.Generator(**config).to(device)
G_ema = model.Generator(**config).to(device)
ema = utils.ema(G, G_ema, 0.9999)
root = '/exports/eddie/scratch/s1580274/weights/BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_seed2_cr_ema'
# q = '_state_dict.pth'
d = torch.load('%s_state_dict.pth' % root)
G_ema.load_state_dict(torch.load('%s_G_ema.pth' % root))
G.load_state_dict(torch.load('%s_G.pth' % root))

G_ema.state_dict()['linear.weight']



""" Test ortho reg and modified ortho reg """
import torch
s1, s2 = 64, 128
w = torch.randn(s1, s2, requires_grad=True)# * 0.02
ortho = 0.5 * torch.sum((torch.mm(w,w.t()) - torch.eye(s1))**2)
do_dw = torch.autograd.grad(ortho,w)[0] # autograd's result
myd = 2*torch.mm(torch.mm(w,w.t()) - torch.eye(s1), w)
# myd = 2*torch.mm(w, torch.mm(w.t(),w) - torch.eye(s1)) Alternate way
# mydo_dw =2 * torch.mm(torch.mm(w.t(),w),)

# modified ortho reg
import torch
s1, s2 = 128, 128
w = torch.randn(s1, s2, requires_grad=True)# * 0.02
ortho = 0.5 * torch.sum((torch.mm(w,w.t())* (1. - torch.eye(s1)))**2)
do_dw = torch.autograd.grad(ortho,w)[0] # autograd's result
myd = 2*torch.mm(torch.mm(w,w.t())*(1. - torch.eye(s1)), w)
# construction by parts
part1 = torch.mm(w, w.t())
torch.diag(part1) *= 0

import torch
import model
import utils
device='cuda'
batch_size=7 * torch.cuda.device_count()
device='cuda'
G = model.Generator(G_ch=96, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)

""" Test sample sheet """
import numpy as np
import torch
import layers
import model
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
import functools
import os
import torchvision
batch_size=50#torch.cuda.device_count()
device='cuda'
# G = model.Generator(ch=64, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)

G = model.Generator(G_ch=64, hier=True, cross_replica=True,G_shared=True, dim_z=128, shared_dim=128).to(device)

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
    y = torch.arange(i, i + classes_per_sheet, device='cuda')    
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

sample_sheet(G, 50, 1000, 10, True, '/home/s1580274/scratch/samples', 'sample_sheet_test', 0)    

""" Test interps """
import numpy as np
import torch
import layers
import model
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
import functools
import os
import torchvision
batch_size=50#torch.cuda.device_count()
device='cuda'
# G = model.Generator(ch=64, hier=True, cross_replica=False, G_shared=True, dim_z=120, shared_dim=128).to(device)

G = model.Generator(G_ch=64, hier=True, cross_replica=True,G_shared=True, dim_z=128, shared_dim=128).to(device)
# Interp function; expects x0 and x1 to be of shape (shape0, 1, shape...)
def interp(x0, x1, num_midpoints):
  lerp_step = 1. / (num_midpoints + 1)
  lerp = torch.arange(0, 1 + lerp_step, lerp_step).cuda()
  return ((x0 * (1 - lerp.view(1,-1,1))) + (x1 * lerp.view(1,-1,1)))
  
# interp sheet function
def interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
           samples_root, experiment_name, folder_number, sheet_number=0
           fix_z=False, fix_y=False)
  # Prepare zs and ys
  
  # If fix Z, only sample 1 z per row
  if fix_z:
    zs = torch.randn(num_per_sheet, 1, G.dim_z, device='cuda')
    zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
  else:
    zs = interp(torch.randn(num_per_sheet, 1, G.dim_z, device='cuda'),
                torch.randn(num_per_sheet, 1, G.dim_z, device='cuda'),
                num_midpoints).view(-1, G.dim_z)
  if fix_y:
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
                               
                               
""" Test to see what we're saving"""
import torch                        
d=torch.load('/home/s1580274/scratch/weights/BigGAN_Gch64_Dch64_bs16_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr5.0e-04_Gortho1.0e-04_seed0_cr_copy0_G_optim.pth')


""" Test dataloader for range problems """
import torch
import numpy as np
import utils
config={'dataset': 'I128', 'hdf5': True, 'batch_size': 64, 'hdf5': True}
loaders = utils.get_data_loaders(**config)
# requires modifying datasets.py to also return the index
for x, y, i in loaders[0]:
  print(x.shape, y.shape)
  break

from PIL import Image
import h5py as h5
from tqdm import tqdm, trange
root = '/home/s1580274/scratch/data/ILSVRC128.hdf5'
y = []
with h5.File(root, 'r') as f:
  img = f['imgs'][i[0]]
timg = ((torch.from_numpy(img).float() / 255) - 0.5) * 2 

# Timg should be equal to  x[0]

""" trained models, see what's up """
import torch
import model
import utils
import torch.nn as nn
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': False, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': False,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'D_nl': nn.ReLU(inplace=True)}
device='cuda'
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
# fname = 'BigGAN_I128_Gch64_Dch64_bs128_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed0'
fname = 'BigLARS3_I128_Gch64_Dch64_bs64_nDs2_Glr5.0e-05_Dlr2.0e-04_Gnlir_Dnlir_Ginitxavier_Dinitxavier_seed2'
root = '/exports/eddie/scratch/s1580274/weights/' + fname
# q = '_state_dict.pth'
d = torch.load('%s_state_dict.pth' % root)
G.load_state_dict(torch.load('%s_G.pth' % root))

# Generate images
G.eval()
G.train()
gz = torch.randn(batch_size, 128).to(device)
gy = torch.arange(batch_size).to(device)
with torch.no_grad():
  x = G(gz, G.shared(gy))

torchvision.utils.save_image(x, 'test_G2.jpg',nrow=batch_size, normalize=True)

# check singular values
import layers
svs, us, vs = layers.power_iteration(G.linear.weight, G.linear.u, update=False)
w = G.linear.weight.data.cpu().numpy()
from numpy.linalg import svd
sp = svd(w, full_matrices=False, compute_uv=False)
c = G.blocks[1][0].conv1
svs, us, vs = layers.power_iteration(c.weight.view(c.weight.size(0), -1), c.u, update=False)
w = c.weight.view(c.weight.size(0), -1).data.cpu().numpy()
sp = svd(w, full_matrices=False, compute_uv=False)

# Load D
import torch
import model
import utils
import torch.nn as nn
device='cuda'
batch_size=16

#fname = 'BigGAN_I128_Gch64_Dch64_bs128_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed0'
fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed27'
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': False, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': False,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'D_nl': nn.ReLU(inplace=True)}
D = model.Discriminator(**config).to(device)
root = '/exports/eddie/scratch/s1580274/weights/' + fname
D.load_state_dict(torch.load('%s_D.pth' % root))


import layers
l = D.blocks[0][0].conv1
svs, us, vs = layers.power_iteration(l.weight.view(l.weight.size(0), -1), l.u, update=False)
w = l.weight.view(l.weight.size(0), -1).data.cpu().numpy()
from numpy.linalg import svd
sp = svd(w, full_matrices=False, compute_uv=True)

import functools
from torch.nn import functional as F
up = functools.partial(F.interpolate, scale_factor=2)

G.optim.load_state_dict(torch.load('%s_G_optim.pth' % root)

""" Test: given the HDF5 dataloader and the non-hdf5 dataloader, 
    query an index. Are the results the same? """
import torch
import utils
import numpy as np
from PIL import Image
# d1 = utils.get_data_loaders('I128', augment=False, batch_size=64, num_workers=1,
                     # shuffle=False, load_in_mem=False, hdf5=False,
                     # pin_memory=True)
d2 = utils.get_data_loaders('I128', augment=False, batch_size=64, num_workers=1,
                     shuffle=False, load_in_mem=False, hdf5=True,
                     pin_memory=True)
for x, y in d2[0]:
  qq = None
  break
im = d2[0].dataset[1][0]
img = Image.fromarray(np.uint8(255 * ((im.numpy() + 1) / 2. )).transpose(1,2,0))
img.save('testds1.jpg')
q = np.load('imagenet_imgs.npz')['imgs'] # list indicating which img is which


""" Test old sngan dataloders """
import torch
import utils
import numpy as np
from PIL import Image
d = utils.get_data_loaders('I128', augment=False, validate=False, test=True,
                     batch_size=50, fold='all', validate_seed=0, validate_split=0.1, num_workers=4,load_in_mem=False)
for x, y in d[0]:
  qq = None
  break
  
  """ Check to make sure all D layers participate by looking at each of their grads? no wait we already did that"""
  
  
""" Compare singular values and gradients for old layers versus new layers """
""" Plan: instantiate two layers each witht he same initial weight and u estimates,
    run a single power iteration, look at the results and the grads
    """
  """ Addition january 25 2019: also compare against nn.utils.spectral_norm """
  

import torch
import utils
import numpy as np
import layers
import old_layers
import torch.nn as nn
x = torch.randn(5,7,64,64)
w = torch.randn(12,7,3,3)
u = torch.randn(1,12)
l = layers.SNConv2d(7,12,3,bias=False)
l.weight.data[:] = w
l.u[0][:] = u
l2 = old_layers.SNConv2D(7,12,3, bias=False)
l2.weight.data[:] = w
l2.u[:] = u


import torch
import utils
import numpy as np
import layers
import old_layers
import torch.nn as nn
l3 = nn.utils.spectral_norm(nn.Conv2d(7,12,3,bias=False))
l = layers.SNConv2d(7,12,3,bias=False)
l.weight.data[:] = l3.weight.data
l.u[0][:] = l3.weight_u.data
l2 = old_layers.SNConv2D(7,12,3, bias=False)
l2.weight.data[:] = l3.weight.data
l2.u[:] = l3.weight_u.data
import model_alex as ma
l4 = ma.SpectralNorm(nn.Conv2d(7,12,3,bias=False))
l4.module.weight_bar.data[:] = l3.weight.data
l4.module.weight_u.data[:] = l3.weight_u.data
l4.module.weight_v.data[:] = l3.weight_v.data


#W0, W2 = l.W_(), l2.W_
x = torch.randn(5,7,64,64)
y0 = l(x)
y2 = l2(x)
y3 = l3(x)
y4 = l4(x)
g0, g2, g3 = torch.autograd.grad(torch.sum(y0), l.weight)[0], torch.autograd.grad(torch.sum(y2), l2.weight)[0], torch.autograd.grad(torch.sum(y3), l3.weight_orig)[0]
g4 = torch.autograd.grad(torch.sum(y4), l4.module.weight_bar)[0]

torch.mean(torch.abs(y0-y2))
torch.mean(torch.abs(y0-y3))
torch.mean(torch.abs(y2-y3))
torch.mean(torch.abs(g2-g3))
torch.mean(torch.abs(g4-g3))



""" Check DCT dataset range, I think it's -8, 8?... """from PIL import Image
import h5py as h5
from tqdm import tqdm, trange
import utils
from PIL import Image
import numpy as np
root = '/home/s1580274/scratch/data/ILSVRC128_DCT.hdf5'
y = []
i =0
with h5.File(root, 'r') as f:
  img = f['imgs'][i]

root = '/home/s1580274/scratch/data/ILSVRC128.hdf5'
i =0
with h5.File(root, 'r') as f:
  img2 = f['imgs'][i]

x = torch.tensor(img)
y = ((torch.tensor(img2).float() / 255.) * 2) - 1.
channel, i, j = 0, 3,4
a = y[channel, 8 * i : 8 * (i + 1), 8 * j : 8 * (j + 1)]
b = x[channel * 64 :64 * (channel + 1), i, j]
  
x = utils.idct(torch.tensor(img).unsqueeze(0))

im = Image.fromarray(np.uint8(255 * ((x.squeeze().numpy() + 1) / 2. )).transpose(1,2,0))
im.save('test_dct1.jpg')
#timg = ((torch.from_numpy(img).float() / 255) - 0.5) * 2 

#img = Image.fromarray(np.uint8(255 * ((im.numpy() + 1) / 2. )).transpose(1,2,0))
#img.save('testds1.jpg')

""" Load DCT model and test its output ranges """
import torch
import model_dct2
import utils
import torch.nn as nn
device='cuda'
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': False, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 16, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'D_nl': nn.ReLU(inplace=True)}
#fname = 'BigGAN_I128_DCT_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-05_Dlr4.0e-05_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed0_Gshared_dct'
#fname = 'BigGAN_I128_DCT_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed1_Gshared_dct'
#fname = 'BigGAN_I128_DCT_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gattn0_Dattn0_seed34_cr_Gshared_dct'
fname = 'BigGAN_I128_DCT_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gattn0_Dattn0_seed1_cr_Gshared_dct'
root = '/exports/eddie/scratch/s1580274/weights/' + fname
G = model_dct2.Generator(**config)
G.load_state_dict(torch.load('%s_G.pth' % root))
G = G.cuda()
G_batch_size, dim_z, nclasses = 16, 128, 1000
z_ = torch.randn(G_batch_size, dim_z, requires_grad=False, device=device)
y_ = torch.randint(low=0, high=nclasses, 
                     size=(G_batch_size,), device=device, 
                     dtype=torch.int64, requires_grad=False)
torch.set_grad_enabled(False) 
gz, h = G(z_, G.shared(y_),True)
# try zeroing out some coeffs?
i = 32
gz[:,i:64] = 0
gz[:,64 + i:128] = 0
gz[:,128 + i:] = 0                   
x = utils.idct(gz)
import torchvision
torchvision.utils.save_image(x.cpu(), 'testGdct.jpg',nrow=G_batch_size,normalize=False)
                               
                               
""" Vae test """
import torch
import model_dct_vae2 as model
import utils
import torch.nn as nn
device='cuda'
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': True, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 16, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'D_nl': nn.ReLU(inplace=True), 'G_nl': nn.ReLU(inplace=True)}
fname = 'BigVAE_I128_DCT_Gch64_Dch64_bs256_Glr2.0e-05_Dlr2.0e-05_GB0.500_DB0.500_Gnlir_Dnlir_Ginitxavier_Dinitxavier_seed1_cr_Gshared_dct_copy0'
root = '/exports/eddie/scratch/s1580274/weights/' + fname
G = model.Generator(**config)
D = model.Discriminator(**config).cuda()
G.load_state_dict(torch.load('%s_G.pth' % root))
D.load_state_dict(torch.load('%s_D.pth' % root))
G = G.cuda()
G_batch_size, dim_z, nclasses = 16, 128, 1000
z_ = torch.randn(G_batch_size, dim_z, requires_grad=False, device=device)
y_ = torch.randint(low=0, high=nclasses, 
                     size=(G_batch_size,), device=device, 
                     dtype=torch.int64, requires_grad=False)
torch.set_grad_enabled(False)  
i = 0
import h5py as h5
with h5.File('/home/s1580274/scratch/data/ILSVRC128_DCT.hdf5', 'r') as f:
  img = f['imgs'][i]

y = torch.tensor(0).long().cuda()
x = torch.tensor(img).cuda().unsqueeze(0)
#z = D(x.unsqueeze(0), y)[:,:128]
z_mu, z_ls = torch.split(D(x, y), 128, 1)
z_hat = z_mu + z_[[0]] * torch.exp(z_ls)
gz = G(z_hat, G.shared(y.long()).unsqueeze(0))
img = ((torch.from_numpy(img).float() / 255)
(gz /2 + 0.5) * 255.
# try zeroing out some coeffs?
i = 32
gz[:,i:64] = 0
gz[:,64 + i:128] = 0
gz[:,128 + i:] = 0                   
xx = utils.idct((gz /2 + 0.5) * 255.)
import torchvision
torchvision.utils.save_image(xx.cpu(), 'testGdct.jpg',nrow=G_batch_size,normalize=False)
                               
 
d = utils.get_data_loaders('I128_DCT', augment=False, batch_size=64, num_workers=1,
                     shuffle=False, load_in_mem=False, hdf5=True,
                     pin_memory=True) 
""" Test to figure out SVDN parameterizations """
import torch
import numpy as np
x = torch.randn(64,128)
u,s,v = torch.svd(x)
v = v.t()
w = v
2*torch.mm(torch.mm(w,w.t()) - torch.eye(64), w)
xx = torch.mm(u*s, v.t())
x = torch.randn(128,64)
u,s,v = torch.svd(x)
xx = torch.mm(u*s, v.t())
#torch.mean(torch.abs(xx-x))

import torch.nn
import torch.nn.init as init
s1, s2 = 128, 256
x = torch.randn(s1,s2)
_ = init.kaiming_uniform_(x)
_,s,_ = torch.svd(x,compute_uv=False)
x.norm(),s


x = np.random.randn(64,128)
u,s,v = np.linalg.svd(x,full_matrices=False)
xx = np.matmul(u * s, v)

w = torch.tensor(x)
ortho = 0.5 * torch.sum((torch.mm(w,w.t()) - torch.eye(s1))**2)
do_dw = torch.autograd.grad(ortho,w)[0] # autograd's result
myd = 2*torch.mm(torch.mm(w,w.t()) - torch.eye(s1), w)
# v dot v.t() is eye, v.t() dot v is NOT                               


""" Observe singular values from trained nets """
import torch
import model
import utils
import torch.nn as nn

device='cuda'
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'cross_replica': True, 'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 32, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'xavier',
          'G_init': 'xavier', 'D_nl': nn.ReLU(inplace=True), 'D_param': 'SVD_SN', 'G_param': 'SVD_SN',
          'n_classes':10}
fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_seed2_cr_Gshared'
fname = 'BigGAN_C100_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_seed2_cr_Gshared'
#fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_DSVD_Gattn0_Dattn0_seed0_cr_Gshared_ema'
root = '/exports/eddie/scratch/s1580274/weights/' + fname
G = model.Generator(**config)
D = model.Discriminator(**config)
G.load_state_dict(torch.load('%s_G.pth' % root))
D.load_state_dict(torch.load('%s_D.pth' % root))
x = torch.randn(4,3,32,32)
y = torch.tensor([0,1,2,3])
x.requires_grad=True
d = D(x, y)
g = torch.autograd.grad(d.mean(),x,create_graph=True)[0]
g2 = g**2
dd = torch.autograd.grad(torch.sum(g2),D.blocks[1][0].conv1.S, retain_graph=True)
torch.sum(g2).backward()
dd = torch.autograd.grad(torch.sum(g2),D.linear.S, retain_graph=True)


""" Tests to orthogonalize matrices"""
u = torch.randn(64,128)
u -= torch.mm(torch.mm(u, u.t()) - torch.eye(u.shape[0], device=u.device), u)

g = torch.autograd.grad(d,x,create_graph=True)


""" Test synchronous batchnorm """
import torch
import model
import utils
import torch.nn as nn
from sync_batchnorm import DataParallelWithCallback, SynchronizedBatchNorm1d, patch_replication_callback
utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_nl': nn.ReLU(inplace=False),'cross_replica':True}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
# Generate images
G=G.train()
G = nn.DataParallel(G)
patch_replication_callback(G)
# G = DataParallelWithCallback(G)
gz = torch.randn(batch_size, 128).to(device)
gz.requires_grad = True
gy = torch.arange(batch_size).to(device)
# with torch.no_grad():
  # x = G(gz, G.shared(gy))
  # x = nn.parallel.data_parallel(G, (gz, G.shared(gy))).cpu()
  # x = G(gz, G.module.shared(gy)).cpu()
x = G(gz, G.module.shared(gy))
l = x.sum()
dz = torch.autograd.grad(l, gz, retain_graph=True)[0]
dl = torch.autograd.grad(l, G.module.linear.weight)[0]

utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_nl': nn.ReLU(inplace=False),'cross_replica':False}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
# Generate images
G=G.train()
x0 = G(gz, G.shared(gy))
l0 = x0.sum()
dz0 = torch.autograd.grad(l0, gz, retain_graph=True)[0]
dl0 = torch.autograd.grad(l0, G.linear.weight)[0]
torch.mean(torch.abs(x0-x)), torch.max(torch.abs(x0-x))
torch.mean(torch.abs(dz0-dz)), torch.max(torch.abs(dz0-dz))
torch.mean(torch.abs(dl0-dl)), torch.max(torch.abs(dl0-dl))

utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_nl': nn.ReLU(inplace=False),'cross_replica':False, 'mybn': True}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
# Generate images
G=G.train()
x1 = G(gz, G.shared(gy))
l1 = x1.sum()
dz1 = torch.autograd.grad(l1, gz, retain_graph=True)[0]
dl1 = torch.autograd.grad(l1, G.linear.weight)[0]

torch.mean(torch.abs(x0-x)), torch.max(torch.abs(x0-x))
torch.mean(torch.abs(dz0-dz)), torch.max(torch.abs(dz0-dz))
torch.mean(torch.abs(dl0-dl)), torch.max(torch.abs(dl0-dl))


torch.mean(torch.abs(x1-x)), torch.max(torch.abs(x1-x))
torch.mean(torch.abs(dz1-dz)), torch.max(torch.abs(dz1-dz))
torch.mean(torch.abs(dl1-dl)), torch.max(torch.abs(dl1-dl))


torch.mean(torch.abs(x0-x1)), torch.max(torch.abs(x0-x1))
torch.mean(torch.abs(dz0-dz1)), torch.max(torch.abs(dz0-dz1))
torch.mean(torch.abs(dl0-dl1)), torch.max(torch.abs(dl0-dl1))

# gz = torch.randn(batch_size, 128).to(device)
# gy = torch.arange(batch_size).to(device)
# with torch.no_grad():
  # x0 = G(gz, G.shared(gy)).cpu()
  # x = nn.parallel.data_parallel(G, (gz, G.shared(gy))).cpu()
  # x = G(gz, G.module.shared(gy)).cpu()  
# xx = torch.load('true_x.pt').cpu()
# torch.mean(torch.abs(xx-x))
#xx = torch.load('syncbntest.pt')
# torch.save(x.data.cpu(),'syncbntest.pt')

""" Test cudnn batchnorm vs cpu batchnorm vs manual batchnorm """
import torch
import utils
import torch.nn as nn
from torch.nn import functional as F
utils.seed_rng(0)
eps = 1e-5
device='cuda'
x = torch.randn(4,128,64,64).to(device)*5.3 - 4.9
x.requires_grad=True
# x2 = F.relu(x, inplace=False)
x2 = x
y0 = F.batch_norm(x2, torch.zeros(128).to(device),torch.ones(128).to(device), None, None,True, 0.1, eps)
# Mean of x
m = torch.mean(x2, [0, 2, 3], keepdim=True)
# Mean of x squared
m2 = torch.mean(x2 ** 2, [0, 2, 3], keepdim=True)
# Calculate variance as mean of squared minus mean squared.
var = (m2 - m **2)
y1 = (x2 - m) * torch.rsqrt(var + eps)
# y2 = F.batch_norm(x2, m.squeeze().detach(), var.squeeze().detach(), None, None,False, 0.1, eps)
torch.mean(torch.abs(y0-y1))
# torch.mean(torch.abs(y0-y2))

# y0 = F.relu(y0)
# y1 = F.relu(y1)
go = torch.randn(y0.shape).to(device)
dx0 = torch.autograd.grad(y0.sum(), x, go, retain_graph=True)[0]
dx1 = torch.autograd.grad(y1.sum(), x, go)[0]
torch.mean(torch.abs(dx0-dx1))
""" Test inplace vs out-of-place relu """
import torch
import model
import utils
import torch.nn as nn
import copy
utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_activation': nn.ReLU(inplace=False),'cross_replica':False}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
G = G.eval()
gz = torch.randn(batch_size, 128).to(device)
gy = torch.arange(batch_size).to(device)
gz.requires_grad = True
x = G(gz, G.shared(gy))
l = x.sum()
l.backward(retain_graph=True)
g0 = copy.deepcopy(G.linear.weight.grad)
G.zero_grad()
for block in G.blocks:
  block[0].activation.inplace=True
  G.output_layer[1].inplace=True

l.backward(retain_graph=True)
g0b = copy.deepcopy(G.linear.weight.grad)
torch.mean(torch.abs(g0 - g0b))
G.zero_grad()
x2 = G(gz, G.shared(gy))
l2 = x2.sum()
l2.backward()
g2 = copy.deepcopy(G.linear.weight.grad)
torch.mean(torch.abs(g0 - g2))



# 
# Older tests
import torch
import model
import utils
import torch.nn as nn
import copy
utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_activation': nn.ReLU(inplace=False),'cross_replica':False}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
G = G.eval()
G = G.eval()
gz = torch.randn(batch_size, 128).to(device)
gy = torch.arange(batch_size).to(device)
gz.requires_grad = True
x = G(gz, G.shared(gy))
l = x.sum()
g0 = torch.autograd.grad(l, gz,retain_graph=True)[0]
g0b = torch.autograd.grad(l, gz,retain_graph=True)[0]
torch.mean(torch.abs(g0 - g0b))
for block in G.blocks:
  block[0].activation.inplace=True
  G.output_layer[1].inplace=True


x2 = G(gz, G.shared(gy))
x[0,0]-x2[0,0]
g1 = torch.autograd.grad(x.sum(), gz)[0]
g2 = torch.autograd.grad(x2.sum(), gz)[0]
torch.mean(torch.abs(g0 - g1))
torch.mean(torch.abs(g1 - g2))
torch.mean(torch.abs(g0 - g2))
loss = x.sum()
loss.backward()
g = gz.grad
gzg = torch.load('gzgrad_oop_cpu.pt')
# torch.save(gz.grad, 'gzgrad_oop_cpu.pt')
# gzg = torch.load('gzgrad_oop2.pt')

# torch.save(gz.grad.cpu(), 'gzgrad_oop2.pt')


""" Test inplace vs out-of-place relu for D"""
import torch
import model
import utils
import torch.nn as nn
import copy
utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '0', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'D_activation': nn.ReLU(inplace=False),'cross_replica':False}
batch_size=16
device='cuda'
D = model.Discriminator(**config).to(device)
D = D.eval()
dx = torch.randn(batch_size, 3,128,128).to(device)
dy = torch.arange(batch_size).to(device)
dx.requires_grad = True
x = D(dx, dy)
l = x.sum()
l.backward(retain_graph=True)
g0 = copy.deepcopy(D.linear.weight.grad)
gx0 = copy.deepcopy(dx.grad)
for block in D.blocks:
  block[0].activation.inplace=True
  D.activation.inplace=True

D.zero_grad() 
dx.grad *= 0
x2 = D(dx, dy)
l2 = x2.sum()
l2.backward()
g2 = copy.deepcopy(D.linear.weight.grad)
gx2 = copy.deepcopy(dx.grad)
torch.mean(torch.abs(g0 - g2))
torch.mean(torch.abs(gx0 - gx2))


for block in D.blocks:
  block[0].activation.inplace=True
  D.activation.inplace=True

l.backward(retain_graph=True)

torch.mean(torch.abs(g0 - g0b))
D.zero_grad()


# gx1 = torch.autograd.grad(x2.sum(), dx)[0]


""" Test interp sheets """
""" Test inplace vs out-of-place relu """
import torch
import model
import utils
import torch.nn as nn
import copy
import torchvision
utils.seed_rng(0)
config = {'parallel': True, 'shuffle': True, 'batch_size': 64,
          'G_lr': 1e-4, 'D_lr': 1e-4,
          'G_ch': 64, 'dim_z': 128, 'shared_dim': 128, 'G_shared': True,
          'resolution': 128, 'G_attn': '64', 'D_attn': '0', 'D_ch': 64, 'D_init': 'N02',
          'G_init': 'N02', 'G_activation': nn.ReLU(inplace=True),'cross_replica':False}
batch_size=16
device='cuda'
G = model.Generator(**config).to(device)
gz = torch.randn(batch_size, 128).to(device)
gy = torch.arange(batch_size).to(device)
root = '/home/s1580274/scratch/weights/BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs128_nDa4_nGa4_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_Gattn64_Dattn64_Gshared_ema_SAGAN_bs128x4_ema'
d = torch.load('%s/state_dict.pth' % root)
G.load_state_dict(torch.load('%s/G_ema.pth' % root))
G = G.eval()
# with torch.no_grad():
  # gx = G(gz,G.shared(gy))

#torchvision.utils.save_image(gx.cpu(),'test.jpg', nrow=int(batch_size**0.5), normalize=True) 

fix_z = True
fix_y = True
parallel=True
num_per_sheet = 12
num_midpoints = 8
device='cuda'
num_classes=1000
# Prepare zs and ys
if fix_z: # If fix Z, only sample 1 z per row
  zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
  zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
else:
  zs = utils.interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
              torch.randn(num_per_sheet, 1, G.dim_z, device=device),
              num_midpoints).view(-1, G.dim_z)

if fix_y: # If fix y, only sample 1 z per row
  ys = utils.sample_1hot(num_per_sheet, num_classes)
  ys = G.shared(ys).view(num_per_sheet, 1, -1)
  ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
else:
  ys = utils.interp(G.shared(utils.sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                    G.shared(utils.sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
              num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)


with torch.no_grad():
  if parallel:
    out_ims = nn.parallel.data_parallel(G, (zs, ys))
  else:
    out_ims = G(zs, ys)

torchvision.utils.save_image(out_ims.cpu(),'test_fixYZ.jpg', nrow=num_midpoints+2,normalize=True)



interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
image_filename = '%s/%s/%d/interp%s%d.jpg' % (samples_root, experiment_name,
                                              folder_number, interp_style,
                                              sheet_number)
torchvision.utils.save_image(out_ims, image_filename,
                               nrow=samples_per_class, normalize=True)  
samples_root = '/home/s1580274/scratch/samples'
experiment_name = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs128_nDa4_nGa4_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_Gattn64_Dattn64_Gshared_ema_SAGAN_bs128x4_ema'
 
""" Test fp32 accumulation for mixed-precision batchnorm/BN"""
import torch
device='cuda'
size = (128,64,128,128)
x=torch.randn(size).to(device)
xh = x.half()
f = torch.mean
y = f(x,[0,2])
yh = f(xh, [0,2]) # Y half precision
ym = f(xh.float(), [0,2]) # Y mixed precision
torch.mean(torch.abs(y-yh.float())), torch.mean(torch.abs(y.half()-yh)), torch.mean(torch.abs(y-ym))
y2 = f(x **2,[0,2])
y2h = f(xh **2, [0,2]) # Y half precision
y2m = f(xh.float() **2, [0,2]) # Y mixed precision
torch.mean(torch.abs(y2-y2h.float())), torch.mean(torch.abs(y2.half()-y2h)), torch.mean(torch.abs(y2-y2m))




var = y2 - y**2
varh = y2h - yh**2
varm = y2m - ym**2
varm2 =y2m.half() - yh ** 2
torch.mean(torch.abs(var-varh.float())), torch.mean(torch.abs(var.half()-varh)), torch.mean(torch.abs(var-varm)), torch.mean(torch.abs(var-varm2.float()))