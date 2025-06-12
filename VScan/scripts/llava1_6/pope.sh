#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ../data/model/llava-v1.6-vicuna-7b \
    --question-file ../data/eval/pope/llava_pope_test.jsonl \
    --image-folder ../data/eval/pope/val2014 \
    --answers-file ../data/eval/pope/answers/llava-v1.6-vicuna-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ../data/eval/pope/coco \
    --question-file ../data/eval/pope/llava_pope_test.jsonl \
    --result-file ../data/eval/pope/answers/llava-v1.6-vicuna-7b.jsonl
