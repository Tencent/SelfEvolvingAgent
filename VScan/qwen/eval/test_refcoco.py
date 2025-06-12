import torch
import torch.distributed as dist
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import math
import numpy as np
import datetime

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionSdpaAttention, Qwen2_5_VisionPatchEmbed, Qwen2_5_VLModel
from qwen.model.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration_X, Qwen2_5_VisionTransformerPretrainedModel_X, Qwen2_5_VLVisionBlock_X, Qwen2_5_VLVisionSdpaAttention_X, Qwen2_5_VisionPatchEmbed_X, Qwen2_5_VLModel_X
from qwen_vl_utils import process_vision_info


def round_by_factor(number: int, factor: int) -> int:
    """ 返回最接近 number 的且能被 factor 整除的整数 """
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """ 返回大于等于 number 的且能被 factor 整除的整数 """
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """ 返回小于等于 number 的且能被 factor 整除的整数 """
    return math.floor(number / factor) * factor

def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280, max_long_side=8192):
    """ 缩放后图片满足以下条件:
        1. 长宽能被 factor 整除
        2. pixels 总数被限制在 [min_pixels, max_pixels] 内
        3. 最长边限制在 max_long_side 内
        4. 保证其长宽比基本不变
    """
    if height < 2 or width < 2:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio must be smaller than 100, got {height} / {width}')

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def remove_leading_articles(text):
    # 使用正则表达式去除开头的 'a', 'an', 'the'，并忽略大小写
    result = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)
    return result

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def gather_results(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    # answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    # content_answer_match = re.search(answer_tag_pattern, content)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    matches = re.findall(bbox_pattern, content)
    bboxes = []
    for match in matches:
        bbox = list(map(float, match))
        bboxes.append(bbox)
    if len(bboxes) == 0:
        bboxes.append([0, 0, 0, 0])
    return bboxes, False


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union


if __name__ == "__main__":
    rank, world_size = get_rank_and_world_size()
    if world_size >= 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))
        print(f'Setting up for local_rank:{local_rank}, rank:{rank}, world_size:{world_size}')
    steps = 0
    print("Steps: ", steps)
    MODEL_PATH = "../data/model/Qwen2.5-VL-7B-Instruct"
    MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
    print('MODEL_PATH:', MODEL_PATH)
    print('MODEL_NAME:', MODEL_NAME)
    OUTPUT_PATH="./logs/rec_results_{DATASET}_{MODEL_NAME}_{STEPS}.json"
    LOG_ROOT = f"./logs/{MODEL_NAME}"
    if not os.path.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT, exist_ok=True)
    BSZ=1


    DATA_ROOT = "../data/eval/RefCOCO/rec_jsons_processed" # the refcoco evaluation set is from the VLM-R1 repo
    TEST_DATASETS = ['refcoco_val', 'refcoco_testA', 'refcoco_testB', 'refcocop_val', 'refcocop_testA', 'refcocop_testB', 'refcocog_val', 'refcocog_test']

    IMAGE_ROOT = "../data/eval"

    random.seed(42)

    # Change to custom forward function
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).eval().cuda()
    
    # Specify pruning settings here
    layer_list = '[14]'
    image_token_ratio_list = '[0.67]'
    image_token_ratio = 0.6

    model.model.layer_list = eval(layer_list)
    model.model.image_token_ratio_list = eval(image_token_ratio_list)
    model.image_token_ratio = image_token_ratio
    
    ratio = torch.ones(28)
    ratio[0] = model.image_token_ratio
    for i in range(len(model.model.layer_list)):
        ratio[model.model.layer_list[i]] = model.model.image_token_ratio_list[i]
    ratio = ratio.cumprod(0)
    
    
    
    print("================Settings=================")
    # Print raye as percent
    print("Retention Rate at Visual Encoder:", model.image_token_ratio * 100, "%")
    for i in range(len(model.model.layer_list)):
        print("Retention Rate at LLM Layer",  model.model.layer_list[i], ":", model.model.image_token_ratio_list[i] * 100, "%")
    avg_keep_ratio = torch.mean(ratio).item()
    print("=========================================")  
    print("Average Retention Rate:", avg_keep_ratio * 100, "%")
    print("=========================================") 

    processor = AutoProcessor.from_pretrained(MODEL_PATH)


    for ds in TEST_DATASETS:
        ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        print(f"Processing {ds} with a total of {len(data)} samples")
        random.shuffle(data)
        QUESTION_TEMPLATE = "{Question}"
        # data = data[:sample_num]
        messages = []

        for x in data:
            image_path = os.path.join(IMAGE_ROOT, x['image'])
            description = remove_leading_articles(x['normal_caption'])
            problem = f"Locate {description} in this image and output the bbox coordinates in JSON format."
            message = [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{image_path}"
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=problem)
                        }
                    ]
                }
            ]
            messages.append(message)

        rank, world_size = get_rank_and_world_size()
        start_idx = rank * (len(messages) // world_size)
        end_idx = (rank+1) * (len(messages) // world_size)
        if rank == world_size - 1:
            end_idx = len(messages)
        
        messages_per_device = messages[start_idx:end_idx]
        data_per_device = data[start_idx:end_idx]
        outputs_per_device = []
        for idx in tqdm(range(0, len(messages_per_device), BSZ)):
            with open("/apdcephfs_us/share_300814644/user/cexzhang/data/eval/RefCOCO/selected_ours.txt", "a") as f:
                f.write(f"{data[idx]['image']}\n")
                f.write(f"{data[idx]['normal_caption']}\n")
            batch_messages = messages_per_device[idx:idx + BSZ]
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            outputs_per_device.extend(batch_output_text)
        
        saved_outputs_per_device = []
        for input_data, output_data in zip(data_per_device, outputs_per_device):
            saved_outputs_per_device.append({
                "data": input_data,
                "output": output_data
            })
        save_path_per_device = os.path.join(LOG_ROOT, f"{MODEL_NAME}_{steps}_{ds}_{rank}.json")
        with open(save_path_per_device, "w") as f:
            json.dump(saved_outputs_per_device, f, ensure_ascii=False)
        
        if world_size > 0:
            dist.barrier()

        if rank == 0:
            final_output = []
            correct_number = 0
            all_outputs = []
            for i in range(world_size):
                save_path_per_device = os.path.join(LOG_ROOT, f"{MODEL_NAME}_{steps}_{ds}_{i}.json")
                outputs_i = json.load(open(save_path_per_device, "r"))
                all_outputs.extend(outputs_i)
            for output in all_outputs:
                input_example = output['data']
                model_output = output['output']
                original_output = model_output
                ground_truth = input_example['solution']
                ground_truth_normalized = input_example['normalized_solution']
                model_answers, normalized = extract_bbox_answer(original_output)
                img_w = input_example['width']
                img_h = input_example['height']
                patch_size = 14
                merge_base = 2
                pixels_per_token = patch_size * patch_size * merge_base * merge_base
                resized_h, resized_w = smart_resize(img_h, img_w, factor=28, min_pixels=pixels_per_token, max_pixels=pixels_per_token * 1280, max_long_side=50000)

                model_answers = np.array(model_answers).reshape(-1, 4) / np.array([resized_w, resized_h, resized_w, resized_h]) * np.array([img_w, img_h, img_w, img_h])
                model_answers = model_answers.tolist()

                for model_answer in model_answers:
                    # Count correct answers
                    correct = 0
                    if model_answer is not None:
                        if not normalized and iou(model_answer, ground_truth) > 0.5:
                            correct = 1
                        elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
                            correct = 1
                    correct_number += correct
                    if correct == 1:
                        break
                
                # Create a result dictionary for this example
                result = {
                    'question': input_example['problem'],
                    'image': input_example['image'],
                    'ground_truth': ground_truth,
                    'model_output': original_output,
                    'extracted_answer': model_answer,
                    'correct': correct
                }
                final_output.append(result)
            
            accuracy = correct_number / len(data) * 100
            print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

            # Save results to a JSON file
            # output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
            output_path = os.path.join(LOG_ROOT, f"{MODEL_NAME}_{steps}_{ds}_all_original.json")
            with open(output_path, "w") as f:
                json.dump({
                    'accuracy': accuracy,
                    'results': final_output
                }, f, indent=2)

            print(f"Results saved to {output_path}")
            print("-"*100)
