#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ../data/model/llava-v1.5-7b \
    --question-file ../data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ../data/eval/scienceqa/images/test \
    --answers-file ../data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ../data/eval/scienceqa \
    --result-file ../data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --output-file ../data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl \
    --output-result ../data/eval/scienceqa/answers/llava-v1.5-7b_result.json
