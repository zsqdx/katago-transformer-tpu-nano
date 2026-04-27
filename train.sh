#!/bin/bash
# Multi-GPU DDP: uncomment MULTI_GPUS below, batch-size is per-GPU
MULTI_GPUS="--multi-gpus 0,1,2,3,4,5,6,7"
# MULTI_GPUS=""

python3 -u train.py \
    --traindir ../data/train/nano_test \
    --datadir ../data/shuffleddata/kata1_trainingdata_25q4_2601 \
    --pos-len 19 \
    --batch-size 1024 \
    --model-kind b12c768 \
    --lr 2e-4 \
    --max-training-samples 300000000 \
    --symmetry-type xyt \
    --print-every 1 \
    --save-every-samples 1000000 \
    --val-every-samples 1000000 \
    --warmup-samples 2000000 \
    --enable-history-matrices \
    ${MULTI_GPUS}
