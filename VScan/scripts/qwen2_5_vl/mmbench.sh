#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m qwen.eval.S \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/mmbench/$SPLIT.tsv \
    --answers-file ../data/eval/mmbench/answers/$SPLIT/Qwen2.5-VL-7B-Instruct.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ../data/eval/mmbench/$SPLIT.tsv \
    --result-dir ../data/eval/mmbench/answers/$SPLIT \
    --upload-dir ../data/eval/mmbench/answers_upload/$SPLIT \
    --experiment Qwen2.5-VL-7B-Instruct
