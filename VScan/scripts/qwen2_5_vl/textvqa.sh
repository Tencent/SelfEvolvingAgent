#!/bin/bash

python -m qwen.eval.model_vqa_loader \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ../data/eval/textvqa/train_images \
    --answers-file ../data/eval/textvqa/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167 

python -m llava.eval.eval_textvqa \
    --annotation-file ../data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ../data/eval/textvqa/answers/Qwen2.5-VL-7B-Instruct.jsonl
