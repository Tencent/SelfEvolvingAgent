#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ../data/model/llava-v1.6-vicuna-7b \
    --question-file ../data/eval/MME/llava_mme.jsonl \
    --image-folder ../data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ../data/eval/MME/answers/llava-v1.6-vicuna-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

cd ../data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.6-vicuna-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.6-vicuna-7b
