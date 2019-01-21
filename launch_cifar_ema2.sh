#!/bin/bash
# launch.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --hdf5 --parallel --shuffle --batch_size 64 --G_shared --hier --cross_replica
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# --hdf5 --parallel --shuffle  --batch_size 128 --G_shared --hier \
# --cross_replica --num_G_accumulations 8 --num_D_accumulations 8 \
# --test_every 1250 --save_every 200 --seed $SEED

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# --hdf5 --parallel --shuffle  --batch_size 64 --G_shared --hier \
# --cross_replica --num_G_accumulations 8 --num_D_accumulations 8 \
# --test_every 1250 --save_every 200 --seed $SEED
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# --hdf5 --parallel --shuffle  --batch_size 64 --G_shared --hier \
# --cross_replica --num_G_accumulations 8 --num_D_accumulations 8 \
# --test_every 10 --save_every 10 --seed 6969 --load_weights 


CUDA_VISIBLE_DEVICES=0,1 python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 --no_fid \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--base_root /home/s1580274/scratch \
--weights_suffix BPT