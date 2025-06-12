#!/bin/bash

python -m qwen.eval.model_vqa_science \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ../data/eval/scienceqa/images/test \
    --answers-file ../data/eval/scienceqa/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

python llava/eval/eval_science_qa.py \
    --base-dir ../data/eval/scienceqa \
    --result-file ../data/eval/scienceqa/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --output-file ../data/eval/scienceqa/answers/Qwen2.5-VL-7B-Instruct_output.jsonl \
    --output-result ../data/eval/scienceqa/answers/Qwen2.5-VL-7B-Instruct_result.json
