#!/bin/bash

CKPT="Qwen2.5-VL-7B-Instruct"

python -m qwen.eval.model_vqa_loader \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/pope/llava_pope_test.jsonl \
    --image-folder ../data/eval/pope/val2014 \
    --answers-file ../data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

python llava/eval/eval_pope.py \
    --annotation-dir ../data/eval/pope/coco \
    --question-file ../data/eval/pope/llava_pope_test.jsonl \
    --result-file ../data/eval/pope/answers/$CKPT.jsonl
