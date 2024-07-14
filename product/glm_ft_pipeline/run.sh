#! /bin/bash
export CUDA_VISIBLE_DEVICES='0'
torchrun --nproc_per_node 1 run_train.py 
