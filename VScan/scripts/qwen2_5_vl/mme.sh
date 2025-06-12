#!/bin/bash

python -m qwen.eval.model_vqa_loader \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/MME/llava_mme.jsonl \
    --image-folder ../data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ../data/eval/MME/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

cd ../data/eval/MME

python convert_answer_to_mme.py --experiment Qwen2.5-VL-7B-Instruct

cd eval_tool

python calculation.py --results_dir answers/Qwen2.5-VL-7B-Instruct
