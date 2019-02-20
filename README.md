# BigGAN-PyTorch
The author's authorized and officially unofficial PyTorch BigGAN implementation.

![Dogball? Dogball!](imgs/header_image.jpg?raw=true "Header")


This repo contains code for replicating experiments from [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) by Andrew Brock, Jeff Donahue, and Karen Simonyan.

This code is by Andy Brock and Alex Andonian.

## How to use this code:
You will need:

- [PyTorch](https://pytorch.org/), version 1.0
- tqdm, scipy, and h5py
- The ImageNet training set

First, you (optionally) need to prepare a pre-processed HDF5 version of your target dataset for faster I/O, and the Inception moments needed calculate FID. This can be done by modifying and running

```sh
sh scripts/prepare_data.sh
```

Which by default assumes your ImageNet training set is downloaded into the root folder "data" in this directory, and will prepare the cached HDF5 at 128x128 pixel resolution.

Now, in order to run experiments on CIFAR or ImageNet, edit the desired .sh files to point towards the root directory where you want your weights/data/samples contained (or edit the logs_root, weights_root, samples_root, and dataset_root arguments individually).

There are also scripts to run SA-GAN and SN-GAN on ImageNet. The SA-GAN code assumes you have 4xTitanX (or equivalent in terms of GPU RAM) and will run with a batch size of 128 and 2 gradient accumulations.

The main script for training on ImageNet at 128x128 is currently "launch_I128_A.sh." This is a script that runs the SA-GAN baseline (no self-attention) with the slight modification of using shared embeddings (helps memory consumption since I run these on shared servers). 

Training scripts will output logs with training metrics and test metrics, will save multiple copies of the model weights/optimizer weights and will produce samples and interpolations every time it saves weights.

If you wish to resume interrupted training, run the same launch script but with the `--resume` arguments added.

See the docs folder for more detailed markdown files describing the rest of this codebase, or check the comments in the code.

After training, one can use `sample.py` to produce additional samples and interpolations, test with different truncation values, batch sizes, number of standing stat accumulations, etc. See a `sample_BigGAN_A.sh` script for an example.

## Using your own dataset
You will need to modify utils.py

-using your own training function: either modify train_fns.GAN_training_function or add a new train fn and modify the train = whichtrainfn line in train.py

## Saved info to be integrated into docs later
See [This directory](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) for ImageNet labels.

## To-do:
- Debug *everything*
- Flesh out this readme
- Writeup design doc
- Write acks

## Acknowledgments
Thanks to Google for the generous cloud credit donations.
Progress bar [originally from](https://github.com/Lasagne/Recipes/tree/master/papers/densenet) Jan Schlüter
TensorFlow Inception Score code from [OpenAI's Improved-GAN](https://github.com/openai/improved-gan)

