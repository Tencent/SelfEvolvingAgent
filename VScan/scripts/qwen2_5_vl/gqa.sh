#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="Qwen2.5-VL-7B-Instruct"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="../data/eval/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m qwen.eval.model_vqa_loader \
        --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
        --question-file ../data/eval/gqa/$SPLIT.jsonl \
        --image-folder ../data/eval/gqa/images \
        --answers-file ../data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --layer-list '[14]' \
        --image-token-ratio-list '[0.333]' \
        --image-token-ratio 0.167 &
done

wait

output_file=../data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ../data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
