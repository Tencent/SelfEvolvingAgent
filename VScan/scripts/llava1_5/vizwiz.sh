#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ../data/model/llava-v1.5-7b \
    --question-file ../data/eval/vizwiz/llava_test.jsonl \
    --image-folder ../data/eval/vizwiz/test \
    --answers-file ../data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ../data/eval/vizwiz/llava_test.jsonl \
    --result-file ../data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file ../data/eval/vizwiz/answers_upload/llava-v1.5-7b.json
