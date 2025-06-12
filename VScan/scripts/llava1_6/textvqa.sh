#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ../data/model/llava-v1.6-vicuna-7b \
    --question-file ../data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ../data/eval/textvqa/train_images \
    --answers-file ../data/eval/textvqa/answers/llava-v1.6-vicuna-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ../data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ../data/eval/textvqa/answers/llava-v1.6-vicuna-7b.jsonl
