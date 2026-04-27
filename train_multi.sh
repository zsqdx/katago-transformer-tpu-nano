#!/bin/bash
# Multi-node multi-GPU training using torchrun.
#
# Usage:
#   Single-node 8-GPU:
#     bash train_multi.sh
#
#   Multi-node (2 nodes, 8 GPUs each):
#     Node 0: MASTER_ADDR=<node0_ip> NNODES=2 bash train_multi.sh 0
#     Node 1: MASTER_ADDR=<node0_ip> NNODES=2 bash train_multi.sh 1
#
# Environment variables (all have defaults):
#   NNODES            - Number of nodes (default: 1)
#   NPROC_PER_NODE    - GPUs per node (default: 8)
#   MASTER_ADDR       - IP of node 0 (default: 127.0.0.1)
#   MASTER_PORT       - Communication port (default: 23456)

NODE_RANK=${1:-0}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-23456}

torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    train.py \
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
    --enable-history-matrices
