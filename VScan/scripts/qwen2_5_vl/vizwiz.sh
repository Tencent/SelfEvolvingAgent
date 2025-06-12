#!/bin/bash

python -m qwen.eval.model_vqa_loader \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/vizwiz/llava_test.jsonl \
    --image-folder ../data/eval/vizwiz/test \
    --answers-file ../data/eval/vizwiz/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ../data/eval/vizwiz/llava_test.jsonl \
    --result-file ../data/eval/vizwiz/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --result-upload-file ../data/eval/vizwiz/answers_upload/Qwen2.5-VL-7B-Instruct.json
