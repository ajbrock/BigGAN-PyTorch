""" Convert imagenet to HDF5
    How to use this: specify an image size on line 25, and a root folder on line 46
    then run it as python make_imagenet_hdf5.py. No cmdline args 
    to-do: prettify this script? """
import os
import sys
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils


batch_size = 256

# Get dataset
kwargs = {'num_workers': 16, 'pin_memory': False, 'drop_last': False}
image_size = 64
train_loader = utils.get_data_loaders(dataset='I%d' % image_size,batch_size=batch_size,
                                      shuffle=False,
                                      dataset_root = '/home/s1580274/scratch/data/', **kwargs)[0]
   


# Chunk sizes to try: 5, 20, 100, 500,  already tried 1000, 
# Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
# 1 / None                   20/s
# 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
# 500 / LZF                                         8/s                   56GB                  23min
# 1000 / None                78/s
# 5000 / None                81/s
# auto:(125,1,16,32) / None                         11/s                  61GB                

chunk_size = 500 # batch_size # 77-78 it/s/worker with chunk_size of 1000, 20 it/s with a chunk_size of 1
compression= None#'lzf' No compression

print('Starting to load I%i into an HDF5 file with chunk size %i and compression %s...' % (image_size, chunk_size, compression))

root = '/home/s1580274/scratch/data/'
# root = '/home/abrock/imagenet/'

for i,(x,y) in enumerate(tqdm(train_loader)):
  x = (255 * ((x + 1) / 2.0)).byte().numpy()
  y = y.numpy()
  if i==0:
    with h5.File(root + 'ILSVRC%i.hdf5' % image_size, 'w') as f:
      print('Producing dataset of len %d' % len(train_loader.dataset))
      imgs_dset = f.create_dataset('imgs', x.shape,dtype='uint8', maxshape=(len(train_loader.dataset), 3, image_size, image_size),
                                   chunks=(chunk_size, 3, image_size, image_size), compression=compression) 
      print('Image chunks chosen as ' + str(imgs_dset.chunks))
      imgs_dset[...] = x
      labels_dset = f.create_dataset('labels', y.shape, dtype='int64', maxshape=(len(train_loader.dataset),), chunks=(chunk_size,), compression=compression)
      print('Label chunks chosen as ' + str(labels_dset.chunks))
      labels_dset[...] = y
  else:
    with h5.File(root + 'ILSVRC%i.hdf5' % image_size, 'a') as f:
      f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
      f['imgs'][-x.shape[0]:] = x
      f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
      f['labels'][-y.shape[0]:] = y
  
    
    
 

# Test h5py
# import numpy as np
# import h5py as h5

# f = h5.File('test1.hdf5','w')
# x = np.zeros((100,3,32,32))
# dset=f.create_dataset('dset1',(100,3,32,32),dtype='uint8',maxshape=(None, 3,32,32)) 
# f.flush()
# f.close()

# f = h5.File('test1.hdf5','a')
# f['dset1'].resize(f['dset1'].shape[0]+100, axis=0)

# with h5.File('test1.hdf5', "a") as f:
    
    
# class ImageFolder_forHDF5(dset.ImageFolder):
                

    # def __getitem__(self, index):
        # """
        # Args:
            # index (int): Index

        # Returns:
            # tuple: (image, target) where target is class_index of the target class.
        # """
        # if self.load_in_mem:
            # img = self.data[index]
            # target = self.labels[index]
        # else:
            # path, target = self.imgs[index]
            # img = self.loader(str(path))
            # if self.transform is not None:
                # img = self.transform(img)
        
        # if self.target_transform is not None:
            # target = self.target_transform(target)
        
        # print(img.size(), target)
        # return img, int(target)