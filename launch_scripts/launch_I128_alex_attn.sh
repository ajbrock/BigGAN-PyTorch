#!/bin/bash
# launch.sh
python train.py \
--model model_alex2 \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 256 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--G_ortho 0.0 \
--G_shared \
--G_init xavier --D_init xavier \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--weights_suffix alex0attn \
--base_root /home/s1580274/scratch