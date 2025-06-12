import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionSdpaAttention, Qwen2_5_VisionPatchEmbed, Qwen2_5_VLModel, Qwen2_5_VLVisionFlashAttention2
from qwen.model.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration_X, Qwen2_5_VisionTransformerPretrainedModel_X, Qwen2_5_VLVisionBlock_X, Qwen2_5_VLVisionSdpaAttention_X, Qwen2_5_VisionPatchEmbed_X, Qwen2_5_VLModel_X, Qwen2_5_VLVisionFlashAttention2_X
from qwen_vl_utils import process_vision_info

from PIL import Image
import math

from io import BytesIO
import base64

all_options = ['A', 'B', 'C', 'D']

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

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

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Change to custom forward function
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune
    Qwen2_5_VLVisionFlashAttention2.forward = Qwen2_5_VLVisionFlashAttention2_X.forward

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", attn_implementation="flash_attention_2", device_map="auto"
    )
    
    model.model.layer_list = eval(args.layer_list)
    model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
    model.image_token_ratio = args.image_token_ratio
    
    min_pixels = 1024*28*28
    max_pixels = 16384*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # ratio = torch.ones(28)
    # ratio[0] = model.image_token_ratio
    # for i in range(len(model.model.layer_list)):
    #     ratio[model.model.layer_list[i]] = model.model.image_token_ratio_list[i]
    # ratio = ratio.cumprod(0)
    
    
    
    # print("================Settings=================")
    # # Print raye as percent
    # print("Retention Rate at Visual Encoder:", model.image_token_ratio * 100, "%")
    # for i in range(len(model.model.layer_list)):
    #     print("Retention Rate at LLM Layer",  model.model.layer_list[i], ":", model.model.image_token_ratio_list[i] * 100, "%")
    # avg_keep_ratio = torch.mean(ratio).item()
    # print("=========================================")  
    # print("Average Retention Rate:", avg_keep_ratio * 100, "%")
    # print("=========================================") 


    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                    
                    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": qs},
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
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)
