#! /bin/sh

nohup python -u train.py --name videoColorization \
                         --use_D \
                         --batch_size 16 \
                         --num_threads 16 \
                         --gpu_ids 0,1 \
                         --display_port 19894 >>videoColorization.out &