import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionSdpaAttention, Qwen2_5_VisionPatchEmbed, Qwen2_5_VLModel
from qwen.model.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration_X, Qwen2_5_VisionTransformerPretrainedModel_X, Qwen2_5_VLVisionBlock_X, Qwen2_5_VLVisionSdpaAttention_X, Qwen2_5_VisionPatchEmbed_X, Qwen2_5_VLModel_X
from qwen_vl_utils import process_vision_info

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Change to custom forward function
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    
    model.model.layer_list = eval(args.layer_list)
    model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
    model.image_token_ratio = args.image_token_ratio

    # default processor
    # processor = AutoProcessor.from_pretrained(model_path)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = output_text[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)
