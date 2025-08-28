import argparse
import os
import json
import time
import re
import base64

from openai import OpenAI

import sys
sys.path.append("../cognitive_kernel_v2/backend")
sys.path.append("../cognitive_kernel_v2/evaluation")

from utils import read_jsonl, save_jsonl, load_jsonl_folder
# from ck_utils import 


import json

def parse_history(res):
    res = json.loads(res)

    outer_loop_history = res['messages']

    inner_loop_history = []

    for i in range(len(res['other_logs'])):
        raw_data = json.loads(res['other_logs'][i]['raw_data'])
        raw_data_clean = []
        for item in raw_data:
            item = {
                "role": item["role"],
                "content": item["content"]
            }
            raw_data_clean.append(item)
        inner_loop_history.append(raw_data_clean)

    ### get all accessibility trees

    histories = [inner_loop_history[i][-2]['content'].split("OBJECTIVE:")[0]  for i in range(len(inner_loop_history))]
    simulation_logs = res['simulation_logs']
    return outer_loop_history, inner_loop_history, histories, simulation_logs


SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Webpage Accessibility Tree: This is a representation of the web page showing the result or intermediate state of performing a web task. It serves as proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the webpage when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT = """TASK: <task>
Result Response: <answer>"""

def auto_eval_by_gpt4o(openai_client, messages, accessibility_tree_history):

    task_info = messages[1]['content']
    ans_info = messages[-2]["content"] + messages[-1]["content"]
    #print(ans_info)
    #input()

    # max_screenshot_id = max([int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f])
    # final_screenshot = f'screenshot{max_screenshot_id}.png'
    # b64_img = encode_image(os.path.join(process_dir, final_screenshot))


    user_prompt_tmp = USER_PROMPT.replace('<task>', task_info)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', ans_info)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_prompt_tmp}
            ]
            + [{'type': 'text', 'text': accessibility_tree_history[i]} for i in range(len(accessibility_tree_history))]
            + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]
    while True:
        try:
            print('Calling gpt4o API to get the auto evaluation......')
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=1000, seed=42, temperature=0
            )
            print('Prompt Tokens:', openai_response.usage.prompt_tokens, ';',
                  'Completion Tokens:', openai_response.usage.completion_tokens)
            print('Cost:', openai_response.usage.prompt_tokens/1000000 * 2.5
                  + openai_response.usage.completion_tokens / 1000000 * 10)

            print('API call complete...')
            break
        except Exception as e:
            print(e)
            if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                #break
                return 'None'
            
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                #exit(0)
                return 'None'
            else:
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content
    print_message = messages[1]

    # print_message[1]['content'][1]['image_url'] = {"url": "data:image/png;base64, b64_img"}
    print(print_message)
    print(gpt_4v_res)

    auto_eval_res = 0 if 'NOT SUCCESS' in gpt_4v_res else 1
    if 'SUCCESS' not in gpt_4v_res:
        auto_eval_res = 'None'
    print('Auto_eval_res:', auto_eval_res)
    print()
    return auto_eval_res


def auto_eval_by_vllm(openai_client, messages, accessibility_tree_history):

    task_info = messages[1]['content']
    ans_info = messages[-1]["content"]

    # max_screenshot_id = max([int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f])
    # final_screenshot = f'screenshot{max_screenshot_id}.png'
    # b64_img = encode_image(os.path.join(process_dir, final_screenshot))


    user_prompt_tmp = USER_PROMPT.replace('<task>', task_info)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', ans_info)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_prompt_tmp}
            ]
            + [{'type': 'text', 'text': accessibility_tree_history[i]} for i in range(len(accessibility_tree_history))]
            + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]
    while True:
        try:
            # print('Calling gpt4o API to get the auto evaluation......')
            openai_response = openai_client.get_response(
                messages=messages, 
                temperature=0,
                do_print=False,
            )

            break
        except Exception as e:
            print(e)
            if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                break
                
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                exit(0)
                #return 'None'
            else:
                time.sleep(10)
    gpt_4v_res = openai_response

    # print(gpt_4v_res)

    auto_eval_res = 0 if 'NOT SUCCESS' in gpt_4v_res else 1
    if 'SUCCESS' not in gpt_4v_res:
        auto_eval_res = 'None' # None
    print('Auto_eval_res:', auto_eval_res)
    print()
    return auto_eval_res



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

def extract_info_from_action(generated_action):
    # zero-shot llama 3.3 additional parsing rules
    if True: #os.environ.get('LLAMA_VERSION') == '3.3': 
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
                action_name, target_id, action_value = match.groups()
                return (action_name, target_id, action_value, enter_marker)


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
        print ('BACKUP SCROLL:', action_value)
        return action_name, None, action_value, False
    return None, None, None, False


def find_target_element_info(current_accessbility_tree, target_id, action_name):
    if target_id is None:
        return None, None, None

    if action_name == "type":
        tree_to_check = current_accessbility_tree.split("\n")[int(target_id) - 1 :]
        for i, line in enumerate(tree_to_check):
            if f"[{target_id}]" in line and ("combobox" in line or "box" not in line):
                num_tabs = len(line) - len(line.lstrip("\t"))
                for j in range(i + 1, len(tree_to_check)):
                    curr_num_tabs = len(tree_to_check[j]) - len(
                        tree_to_check[j].lstrip("\t")
                    )
                    if curr_num_tabs <= num_tabs:
                        break
                    if "textbox" in tree_to_check[j] or "searchbox" in tree_to_check[j]:
                        target_element_id = tree_to_check[j].split("]")[0].strip()[1:]
                        print(
                            "CATCHED ONE MISSED TYPE ACTION, changing the type action to",
                            target_element_id,
                        )
                        target_id = target_element_id
    target_pattern = r"\[" + re.escape(target_id) + r"\] ([a-z]+) '(.*)'"
    matches = re.finditer(target_pattern, current_accessbility_tree, re.IGNORECASE)
    for match in matches:
        target_element_type, target_element_name = match.groups()
        return target_id, target_element_type, target_element_name
    return target_id, None, None


def extract_action_for_web(current_accessbility_tree, raw_action_code, expanded_part):
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
    if (
        generated_action.lower().startswith("goback")
        or generated_action.lower().startswith("restart")
        or generated_action.lower().startswith("stop")
    ):
        if generated_action.lower().startswith("goback"):
            action_name = "goback"
        elif generated_action.lower().startswith("restart"):
            action_name = "restart"
        elif generated_action.lower().startswith("stop"):
            action_name = "stop"
        return (
            {
                "action_name": action_name,
                "target_id": None,
                "action_value": None,
                "need_enter": None,
                "target_element_type": None,
                "target_element_name": None,
            },
            generated_action,
            thought,
        )
    else:
        action_name, target_id, action_value, need_enter = extract_info_from_action(
            generated_action
        )
        if action_name == "type":
            need_enter = '1'
        target_id, target_element_type, target_element_name = find_target_element_info(
            current_accessbility_tree, target_id, action_name
        )
        if expanded_part and int(target_id) in expanded_part:
            expand_target_id, expand_target_type, expand_target_name = expanded_part[int(target_id)]
            print ("Expanded target found", expand_target_id, expand_target_type, expand_target_name, target_element_name)
            return ({
                "action_name": 'select',
                "target_id": expand_target_id,
                "action_value": target_element_name,
                "need_enter": None,
                "target_element_type": expand_target_type,
                "target_element_name": expand_target_name,
            },
            generated_action,
            thought,
            )
        return (
            {
                "action_name": action_name,
                "target_id": target_id,
                "action_value": action_value,
                "need_enter": need_enter,
                "target_element_type": target_element_type,
                "target_element_name": target_element_name,
            },
            generated_action,
            thought,
        )


from copy import deepcopy
def align_sim_log(inner_loop_history, simu_log):
    aligned_res = []
    def convert_act_set(simu):
        output_set = set()
        for s in simu:
            act = s['Action']
            output_set.add(act)
        
        return output_set
    
    simu_log_ptr = 0
    simu_log_set = convert_act_set(simu_log[simu_log_ptr]) if len(simu_log) > 0 else set()
    simu_log_ptr += 1
    for idx, hist in enumerate(inner_loop_history):
        action = hist[-1]['content']
        obs = hist[-2]
        if action in simu_log_set:
            # print(action, simu_log[simu_log_ptr - 1])
            aligned_res.append((obs, action, simu_log[simu_log_ptr - 1]))
            if idx < len(inner_loop_history) - 1 and simu_log_ptr < len(simu_log):
                simu_log_set = convert_act_set(simu_log[simu_log_ptr])
                simu_log_ptr += 1
                
        else:
            #print(action, "No Branching")
            aligned_res.append((obs, action, None))
        #print('-' * 30)
            
    assert simu_log_ptr == len(simu_log) or len(simu_log) == 0
    return aligned_res
    
#align_sim_log(inner_loop_history, json.loads(simu_log['raw_data']))

import argparse
import os
import json
import time
import re
import base64
from openai import OpenAI
import json
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

client = AzureOpenAI(
    azure_endpoint = "", 
    api_key="",  
    api_version="2024-02-01"
    )

def omit_obs_think(message_list):
    res_list = []
    for idx, message in enumerate(message_list):
        if message['role'] == 'user' and idx < len(message_list) - 2:
            res_list.append(
                {
                    'role': 'user',
                    'content': '<|im_omitted|>'
                }
            )
        elif message['role'] == 'assistant' and idx < len(message_list) - 1:
            res_list.append(
                {
                    'role': 'assistant',
                    'content': message['content'].split("</think>\n")[-1]
                }
            )
        else:
            res_list.append(message)
    
    return res_list

def auto_eval_by_gpt4o(openai_client, messages):
    while True:
        try:
            print('Calling gpt4o API to get the auto evaluation......')
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, seed=42, temperature=0
            )
            print('Prompt Tokens:', openai_response.usage.prompt_tokens, ';',
                  'Completion Tokens:', openai_response.usage.completion_tokens)
            print('Cost:', openai_response.usage.prompt_tokens/1000000 * 2.5
                  + openai_response.usage.completion_tokens / 1000000 * 10)

            print('API call complete...')
            break
        except Exception as e:
            print(e)
            
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                #exit(0)
                return 'None'
            else:
                try:
                    if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                        return 'None'
                except:
                    pass
                
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content
    # print_message = messages[1]

    # print_message[1]['content'][1]['image_url'] = {"url": "data:image/png;base64, b64_img"}
    #print(print_message)
    #print(gpt_4v_res)

    return gpt_4v_res


from tqdm import tqdm
backtracking_data = []
for obs_act_list in tqdm(backtracking_list):
    message_list = [
            {
                'role': 'system',
                'content': original_system_message
            }
        ]
    for obs, action, _ in obs_act_list[:-1]:
        message_list.append(obs)
        message_list.append(
            {
                'role': 'assistant',
                'content': action
            }
        )
    
    other_act_msg = f"\nPreviously, the action \"{obs_act_list[-2][1]}\" has been attempted, and this action will not lead to the task completion. "
    other_act_msg += "Please provide an action for going back to the last observation following the aforementioned format. Give your brief reason why this action cannot help to complete the task."
    other_act_msg += f"\nLast Observation: {obs_act_list[-2][0]}"
    prev_user_message = (
    obs_act_list[-1][0]['content'] + other_act_msg
    )
    messages = omit_obs_think(deepcopy(message_list))
    messages.append(
        {
            'role': 'user',
            'content': prev_user_message
        }
    )
    res = auto_eval_by_gpt4o(client, messages)
    message_list.append({
            'role': 'user',
            'content': obs_act_list[-1][0]['content']
        })
    message_list.append({
            'role': 'assistant',
            'content': res
        })
    backtracking_data.append(omit_obs_think(deepcopy(message_list)))
    #break

#print('before sft len:', len(sft_data))
#random.shuffle(backtracking_data)
#save_jsonl([{"messages": item} for item in backtracking_data], f"/apdcephfs/default125428/apdcephfs_cq11/share_1603164/user/tianqfang/AFM/sft_data/minda_sft_data_ablation/backtracking_cot_v3_aligned.jsonl")

import sys
import glob

import argparse
import os
import json
import time
import re
import base64
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy
import json
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

client = AzureOpenAI(
    azure_endpoint = "", 
    api_key="",  
    api_version="2024-02-01"
    )


def omit_obs_think(message_list):
    res_list = []
    for idx, message in enumerate(message_list):
        if message['role'] == 'user' and idx < len(message_list) - 2:
            res_list.append(
                {
                    'role': 'user',
                    'content': '<|im_omitted|>'
                }
            )
        elif message['role'] == 'assistant' and idx < len(message_list) - 1:
            res_list.append(
                {
                    'role': 'assistant',
                    'content': message['content'].split("</think>\n")[-1]
                }
            )
        else:
            res_list.append(message)
    
    return res_list


def auto_eval_by_gpt4o(openai_client, messages):
    while True:
        try:
            print('Calling gpt4o API to get the auto evaluation......')
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, seed=42, temperature=0
            )
            print('Prompt Tokens:', openai_response.usage.prompt_tokens, ';',
                  'Completion Tokens:', openai_response.usage.completion_tokens)
            print('Cost:', openai_response.usage.prompt_tokens/1000000 * 2.5
                  + openai_response.usage.completion_tokens / 1000000 * 10)

            print('API call complete...')
            break
        except Exception as e:
            print(e)
            
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                #exit(0)
                return 'None'
            else:
                try:
                    if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                        return 'None'
                except:
                    pass
                
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content
    # print_message = messages[1]

    # print_message[1]['content'][1]['image_url'] = {"url": "data:image/png;base64, b64_img"}
    #print(print_message)
    #print(gpt_4v_res)

    return gpt_4v_res


total_cnt = 0
backtrack_cnt = 0
backtracking_list = []
sampled_backtracking_data = []
for path in list(glob.glob("/data/workspace/webvoyager_backtracking/llama3.3-70b-backtracking-train-set-sampling-latest/*")) + list(glob.glob("/data/workspace/webvoyager_backtracking/llama3.3-70B-backtracking-train-set-sampling-googlesearch-latest/*")):
    try:
        with open(path, "r") as fin:
            js = json.loads(json.loads(fin.read()))[0]
    except:
        continue
    
    if "backtrack_messages" in js:
        backtracking_list.extend(js["backtrack_messages"])
        

def worker_func(message_list):
    message_list[0]['content'] = original_system_message
    other_act_msg = f"\nPreviously, the action \"{message_list[-3]['content']}\" has been attempted, and this action will not lead to the task completion. "
    other_act_msg += "Please provide an action for going back to the last observation following the aforementioned format. Give your brief reason why this action cannot help to complete the task."
    other_act_msg += f"\nLast Observation: {message_list[-4]['content']}"
    messages = deepcopy(message_list[:-1])
    messages.append(
        {
            'role': 'user',
            'content': message_list[-2]['content'] + other_act_msg
        }
    )
    
    res = auto_eval_by_gpt4o(client, messages)
    res_messages = deepcopy(message_list[:-1])
    #print(res_messages)
    res_messages.append({
            'role': 'assistant',
            'content': res
        })
    
    return omit_obs_think(deepcopy(res_messages))
    #backtracking_data.append(omit_obs_think(deepcopy(res_messages)))

with ThreadPoolExecutor(64) as executor:
    for res in tqdm(executor.map(worker_func, backtracking_list), total=len(backtracking_list)):
        sampled_backtracking_data.append(res)
    #backtracking_data.append(worker_func(backtracking_list[0]))



import sys
from copy import deepcopy
sys.path.append("./cognitive_kernel_v2/backend")
sys.path.append("./cognitive_kernel_v2/evaluation")
from utils import read_jsonl, save_jsonl, load_jsonl_folder

with open("./data/long_cot_rej_sampling_v3_wo_loop_aligned.jsonl") as fin:
    llama3_data_list = [json.loads(line)['messages'] for line in fin]
    
zeroshot_success_query = set()
wo_loop_sys_prompt = 'You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere\'s the information you\'ll have:\nThe user\'s objective: This is the task you\'re trying to complete.\nThe current observation (web page\'s accessibility tree): This is a simplified representation of the webpage, providing key information. Optionally, you may be provided with a screenshot of the webpage. You should pay close attention to the screesnhot to make decisions.\nThe open tabs: These are the tabs you have open.\nThe previous actions: You can refer to the conversation history with the user to see the actions you have taken. It may be helpful to track your progress.\n\nThe actions you can perform are the following:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.\n`scroll [direction=down|up]`: Scroll the page up or down.\n`wait`: Wait for the page to load, with a duration of 5 seconds.\n`goback`: Navigate to the previously viewed page.\n`restart`: Navigate to the Google search homepage. When you can\'t find information in some websites, try starting over from Google search.\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.\n2. You should only issue one action at a time.\n3. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.\n4. Issue stop action when you think you have achieved the objective. Don\'t generate anything after stop.\n5. If you ever need to login, login with Google, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.\n\nYour reply should strictly follow the format:\nThought: {Your brief thoughts (briefly summarize the info that will help complete the task)} Action: ```{the next action you choose to take}```'
sft_data = []
MIX_RATIO = 1
loop_cnt = 0
print("len(llama3_data_list)", len(llama3_data_list))
if MIX_RATIO == 0:
    sampled_llama3_data_list = []
elif MIX_RATIO < 1:
    sampled_llama3_data_list = random.sample(llama3_data_list, int(len(llama3_data_list) * MIX_RATIO))
else:
    sampled_llama3_data_list = llama3_data_list

for data in sampled_llama3_data_list:
    if data[0]['content'] == wo_loop_sys_prompt:
        assert data[0]['role'] == 'system'
        data[0]['content'] = webdreamer_system_message
        
    sft_data.append(data)

print('before sft len:', len(sft_data))
'''for data in deepcopy(backtracking_data):
    data[-1]['content'] = data[-1]['content'].split("</think>")[-1].strip()
    sft_data.append(data)

for data in deepcopy(sampled_backtracking_data):
    data[-1]['content'] = data[-1]['content'].split("</think>")[-1].strip()
    sft_data.append(data)'''

sft_data.extend(backtracking_data)
sft_data.extend(sampled_backtracking_data)
print('after sft len:', len(sft_data))
import random
random.shuffle(sft_data)
save_jsonl([{"messages": item} for item in sft_data], f"./webcot_backtracking.jsonl")
