# BigGAN-PyTorch
The author's authorized and officially unofficial PyTorch BigGAN implementation.
[Dogball? Dogball!](header_image.jpg?raw=true "Header")


This repo contains code for replicating experiments from [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) by Andrew Brock, Jeff Donahue, and Karen Simonyan.


## How to use this code:
You will need:

- [PyTorch](https://pytorch.org/), version 1.0
- tqdm, scipy, and h5py
- The ImageNet training set


Optionally, first, edit and run "make_imagenet_hdf5.py" to produce an hdf5 file containing a pre-processed ImageNet at 128x128 pixel resolution for faster I/O.

Next, run the "calculate_inception_moments.py" script to produce the moment files needed to calculate FID, e.g.

```sh
python calculate_inception_moments.py --dataset I128_hdf5 --parallel
```

Now, in order to run experiments on CIFAR or ImageNet, edit the desired .sh files to point towards the root directory where you want your weights/data/samples contained (or edit the logs_root, weights_root, samples_root, and dataset_root arguments individually).

The main script for training on ImageNet at 128x128 is currently "launch_I128_A.sh." This is a script that runs the SA-GAN baseline (no self-attention) with the slight modification of using shared embeddings (helps memory consumption since I run these on shared servers). 

Training scripts will output logs with training metrics and test metrics, will save multiple copies of the model weights/optimizer weights and will produce samples every time it saves weights.

If you wish to resume interrupted training, run the same launch script but with the `--load_weights` and `--resume` arguments added.

## To-do:
- Debug *everything*
- Cleanup make_imagenet_hdf5.py
- Flesh out this readme
- Writeup design doc
- Write acks 
- Determine if init style is correct? Have we always used N(0,0.02) on the embeddings?