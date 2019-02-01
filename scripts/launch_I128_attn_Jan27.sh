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


CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--G_eval_mode --no_pin_memory --split_D \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 192 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--G_ortho 0.0 \
--G_shared \
--G_init xavier --D_init xavier \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--weights_suffix jan27attn \
--base_root /home/s1580274/scratch