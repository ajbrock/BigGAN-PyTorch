#!/bin/bash
python ../../calculate_inception_moments.py --dataset Kinetics400 --num_workers 1 --data_root /home/shared/cs_vision/train_frames_12fps_128_center_cropped_h5/compact.h5 \
--batch_size 256 --shuffle
