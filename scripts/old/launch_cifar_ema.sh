CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--shuffle --batch_size 64 \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 \
--dataset C10 --no_fid \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init xavier --D_init xavier \
--ema --use_ema --ema_start 1000 \
--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \