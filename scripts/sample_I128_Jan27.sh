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


CUDA_VISIBLE_DEVICES=0,1,2,3 python sample.py \
--G_eval_mode --G_batch_size 512 \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_nl relu --D_nl relu \
--G_ortho 0.0 \
--G_shared \
--G_init xavier --D_init xavier \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--load_weights --resume \
--weights_suffix jan27 \
--sample_npz --sample_num_npz 50000 --sample_sheets --sample_sheet_folder_num -1 --sample_random --sample_inception_metrics \
--base_root /home/s1580274/scratch