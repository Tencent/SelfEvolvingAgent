#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 -m qwen.eval.test_refcoco
# Note: You should directly modify the pruning settings in test_refcoco.py
