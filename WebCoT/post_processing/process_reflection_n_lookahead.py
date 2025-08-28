import time
import pandas as pd
import requests
import json
import os
from copy import deepcopy
import random
from itertools import chain

import sys
sys.path.append("../cognitive_kernel_v2/backend")
sys.path.append("../cognitive_kernel_v2/evaluation")

from ck_utils import test_system_prompt, extract_string, start_docker, stop_docker, extract_string
from eval_webvoyager import parse_history
from evaluator.simpleqa_evaluator import evaluate_simple_qa, ChatGPTConnection, get_final_results
from utils import read_jsonl, save_jsonl, gather_multiple_runs_jsonl, load_jsonl_folder

os.environ['LLAMA_VERSION'] = '3.3'
import re

def extract_info_from_action(generated_action):
    '''
        This function is adapted from  call_web.py::extract_info_from_action
        Extract structured action from generated_action
    '''
    if (
        generated_action.lower().startswith("goback")
        or generated_action.lower().startswith("restart")
        or generated_action.lower().startswith("wait")
        or generated_action.lower().startswith("stop")
    ):
        val = None
        if generated_action.lower().startswith("goback"):
            action_name = "goback"
        elif generated_action.lower().startswith("restart"):
            action_name = "restart"
        elif generated_action.lower().startswith("wait"):
            action_name = "wait"
        elif generated_action.lower().startswith("stop"):
            action_name = "stop"
            pattern = r"stop\s+(\[?[^\]]+\]?)"
    
            # Search for the pattern in the input string
            match = re.search(pattern, generated_action)
            
            # If a match is found, return the extracted string
            if match:
                val = match.group(1)
            else:
                val = None
        return (
                action_name,
                None,
                val,
                None
            )

    # zero-shot llama 3.3 additional parsing rules
    if os.environ.get('LLAMA_VERSION') == '3.3': 
        # print("llama3.3 parsing")
        pattern = r"(\w+)\s+\[?(\d+)\]?\s+\[?(.+?)\]?\s+\[?press_enter_after=(\d+)\]?"
        match = re.match(pattern, generated_action)
        if match:
            action_name, target_id, action_value, enter_marker = match.groups()
            return (action_name, target_id, action_value, enter_marker)
        else:
            # match the end first
            if generated_action.split()[-1] == '1':
                enter_marker = '1'
                generated_action = generated_action[:-2]
                pattern = r"(\w+)\s+\[?(\d+)\]?\s+\[?(.+)\]?"
                match = re.match(pattern, generated_action)
                if match:
                    action_name, target_id, action_value = match.groups()
                    return (action_name, target_id, action_value, enter_marker)
                else:
                    return (None, None, None, None)


    action_pattern = r"(\w+) \[(\d+)\](?: \[([^\]]*)\])?( \[1\])?"
    match = re.search(action_pattern, generated_action)
    if match:
        action_name, target_id, action_value, enter_marker = match.groups()
        need_enter = enter_marker is not None
        if (action_name == 'type' and action_value is not None) or action_name == 'click':
            return (
                action_name,
                target_id,
                action_value if action_value else None,
                need_enter,
            )
    else:
        scroll_pattern = r"(\w+) \[([^\d\]]*)\]( \[1\])?"
        scroll_match = re.search(scroll_pattern, generated_action)
        if scroll_match:
            action_name, action_value, enter_marker = scroll_match.groups()
            action_value = action_value.replace('direction=', '')
            return action_name, None, action_value, enter_marker is not None
    ## this is a hack to gpt-based backbone because it do not always generate the correct `type` action
    if os.environ.get('MODEL_NAME', "ck") != "ck":
        if generated_action.startswith("type"):
            action_name = 'type'
            
            if generated_action.endswith(' 1'):
                generated_action = generated_action[:-2]
                need_enter = True
            elif generated_action.endswith('[1]'):
                generated_action = generated_action[:-3]
                need_enter = True
            
            need_enter = True
            
            pattern = r'type\s+(\[?\d+\]?)\s(.+)'
            # r'(type)\s+(\[6\])\s+(.+)'
            # #+(.+?)\s+
            
            match = re.match(pattern, generated_action)

            target_id, action_value = match.groups()

            target_id = target_id.strip('[]')
            action_value = action_value.strip('[]')

            return (
                action_name,
                target_id,
                action_value,
                need_enter,
            )
        
    if generated_action.startswith("type") or generated_action.startswith("click"):
        backup_pattern = r"(\w+)\s*(\[\d+\]|\d+)?\s*([^\[]+)?\s*(\[press_enter_after=\d\]|\[1\]|1)?"
        backup_match = re.search(backup_pattern, generated_action)
        if backup_match:
            action_name, target_id, action_value, enter_marker = backup_match.groups()
            if isinstance(target_id, str) and target_id.startswith("["):
                target_id = target_id[1:-1]
            need_enter = enter_marker is not None
            return action_name, target_id, action_value, need_enter
    elif generated_action.startswith("scroll"):
        action_name = "scroll"
        if len(generated_action.split(" ")) > 1:
            action_value = generated_action.split(" ")[1]
        else:
            action_value = None

        # new fix:
        action_value = action_value.replace('direction=', '')
        # print ('BACKUP SCROLL:', action_value)
        return action_name, None, action_value, False
    return None, None, None, False



def prune_actions(raw_action_code):
    try:
        thought = raw_action_code.split("Action:")[0].replace("Thought:", "").strip()
    except:
        thought = None
    # pattern = rf"```(.*?)```"
    # match = re.search(pattern, raw_action_code)
    # generated_action = match.group(1)
    action_part = raw_action_code.split("Action:")[1]
    start = action_part.find("```")
    end = action_part.rfind("```")

    if start != -1 and end != -1 and start != end:
        generated_action = action_part[start + 3:end].strip()
    else:
        print("No matching triple backticks found or only one set found.")
        return None
    return generated_action


def sanity_check_action(s):

    try:
        extract_info_from_action(prune_actions(s))
        return 1
    except:
        print("error", s)
        return 0
    
    
def extract_url(text):
    pattern = r"Please interact with (.*) and get the answer."
    matches = re.findall(pattern, text)[-1]
    return matches

def extract_task(text):
    pattern = r"Now given a task: (.*) Please interact with .* and get the answer."
    matches = re.findall(pattern, text)[-1]
    return matches

def extract_thought(text):
    pattern = r"Thought:([\s\S]*)Action:[\s\S]*"
    matches = re.findall(pattern, text)[-1]
    return matches

def extract_action(text):
    pattern = r"Thought:[\s\S]*Action:([\s\S]*)"
    matches = re.findall(pattern, text)[-1]
    return matches


data = json.load(open("./all_outer_inner_messages_w_minda_trajwt_v3_wo_loop_aligned.json"))
success_index = json.load(open("./data/success_index.json", "r"))
sft_data = []
token_cnt_list = []
for idx in success_index:
    item = data[2][idx]
    original_item = data[1][idx]

    # outer loop
    sft_data.append(data[0][idx])

    # if no long-cot is produced, use the original inner_history
    if item is None or item['corresponding trajectory'] is None or item['corresponding trajectory'] == '[No Loop Here]':
        sft_data.extend(data[1][idx])
        continue
    
    print("success idx", idx)
    
    # get the objective from data[1]
    for k in range(len(data[1][idx])):
        if data[1][idx][k][-2]['content'] != "<|im_omitted|>":
            obj = data[1][idx][k][-2]['content'].split('OBJECTIVE: ')[1]

    for j in range(len(item['corresponding trajectory']['traj_list'])):
        all_format_correct = 1
        traj = item['corresponding trajectory']['traj_list'][j]
        ori_traj = original_item[j]
        for i in range(2, len(traj)):
            if i % 2 == 0:
                if_last_act =  (i + 2 >= len(traj)) # if it is the last action
                s = traj[i]['content']
                ori_s = ori_traj[i]['content']
                ### process s:
                if 'Insights on Current Action' in s:
                    s = s.replace('Insights on Current Action', "Insights")
                
                if len(s.split("Action:")[:-1]) >= 2:
                    s = "Step:".join(s.split("Action:")[:-1]) + "Action:" + s.split("Action:")[-1]
                    # print(s)
                s = s.replace('Action: Action:', 'Action:').replace('Thought: Thought:', "Thought:")
                    
                traj[i]['content'] = f'Thought:{extract_thought(s)}Action:{extract_action(s)}'
                all_format_correct = all_format_correct * sanity_check_action(s)
            else:
                assert traj[i]['content'] == '<|im_omitted|>' or i + 2 >= len(traj)

            # add objective

        # print(traj[-2]['content'])
        try:
            assert traj[-2]['content'] != '<|im_omitted|>'
        except:
            print("error idx", idx)
            continue
            
        traj[-2]['content'] = traj[-2]['content'] + " \nOBJECTIVE: " + obj
        print(traj[-2:])
        input()
            
        if all_format_correct == 1:
            sft_data.append(traj)
            assert all([sanity_check_action(traj[i]['content']) for i in range(2, len(traj)) if i % 2 == 0])

            token_count = sum([len(turn['content'].split()) for turn in traj])
            token_cnt_list.append(token_count)


import random
random.shuffle(sft_data)
save_jsonl([{"messages": item} for item in sft_data], "./webcot_reflection_n_lookahead.jsonl")
