#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ../data/model/llava-v1.6-vicuna-7b \
    --question-file ../data/eval/mmbench/$SPLIT.tsv \
    --answers-file ../data/eval/mmbench/answers/$SPLIT/llava-v1.6-vicuna-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ../data/eval/mmbench/$SPLIT.tsv \
    --result-dir ../data/eval/mmbench/answers/$SPLIT \
    --upload-dir ../data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.6-vicuna-7b
