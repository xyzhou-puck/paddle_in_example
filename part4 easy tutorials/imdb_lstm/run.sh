#!/bin/bash

set -ex
export CUDA_VISIBLE_DEVICES=0

python train.py \
        --num_layers 1 \
        --hidden_size 128 \
        --src_vocab_size 5147 \
        --batch_size 32 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --use_gpu True

