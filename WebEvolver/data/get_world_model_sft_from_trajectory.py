import time
import pandas as pd
import requests
import json
import os
from copy import deepcopy
import random
from itertools import chain

import sys
sys.path.append("cognitive_kernel_v2/backend")
sys.path.append("cognitive_kernel_v2/evaluation")

from ck_utils import test_system_prompt, extract_string, extract_string
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

def filter_wait_checker(messages_list):
    """
        remove all the step with "wait" action
        remove the message ending with the "wait" action.
    """
    pop_msg_ids = []
    for msg_id, messages in enumerate(messages_list):
        pop_msg_is = []
        for i, message in enumerate(messages):
            if "```wait```" in message['content'] and message['role'] == 'assistant':
                return True
    return False

def filter_wait(messages_list):
    """
        remove all the step with "wait" action
        remove the message ending with the "wait" action.
    """
    pop_msg_ids = []
    for msg_id, messages in enumerate(messages_list):
        pop_msg_is = []
        for i, message in enumerate(messages):
            if "```wait```" in message['content'] and message['role'] == 'assistant':
                if i == len(messages) - 1:
                    pop_msg_ids.append(msg_id)
                pop_msg_is.append(i)
                pop_msg_is.append(i-1)
                break
        pop_msg_is.sort(reverse=True)
        for pop_msg_i in pop_msg_is:
            messages.pop(pop_msg_i)
        
    for pop_msg_id in pop_msg_ids:
        messages_list.pop(pop_msg_id)
    return messages_list

def filter_login(messages_list):
    """
        count the number of actions which requires logging in.
    """
    cnt = 0
    for i, message in enumerate(messages_list[-1]):
        if "log in" in message['content'] and message['role'] == 'assistant':
            cnt += 1
            break
    return cnt

def filter_no_change(messages_list):
    """
        check the current accessibility v.s. the last one.
        If no change, then remove the message.
    """

    flag = False

    removed_msg_ids = []
    removed_msg_list_ids = []
    
    for msg_id, messages in enumerate(messages_list):
        if msg_id == 0:
            continue
        
        last_accessibility_tree = messages_list[msg_id - 1][-2]['content'].split('OBJECTIVE:')[0]
        current_accessibility_tree = messages_list[msg_id][-2]['content'].split('OBJECTIVE:')[0]

        
        if last_accessibility_tree == current_accessibility_tree or "The action you have chosen cannot be executed." in messages_list[msg_id][-2]['content']:
            # 1. remove this message
            # 2. remove the action and observation for future steps
            # print(msg_id)
            removed_msg_list_ids.append(msg_id - 1)
            removed_msg_ids.append(len(messages_list[msg_id]) - 3)
            removed_msg_ids.append(len(messages_list[msg_id]) - 2)
        
        # for ii in range(msg_id - 1, 0, -1):
        # if "The action you have chosen cannot be executed." in messages_list[msg_id][-2]['content']:
        #     removed_msg_list_ids.append(msg_id)
        #     removed_msg_ids.append(len(messages_list[msg_id]) - 1)
        #     removed_msg_ids.append(len(messages_list[msg_id]) - 2)

    removed_msg_ids = list(set(removed_msg_ids))
    removed_msg_list_ids = list(set(removed_msg_list_ids))
    # print(removed_msg_ids, removed_msg_list_ids)
    removed_msg_ids.sort(reverse=True)
    removed_msg_list_ids.sort(reverse=True)

    for removed_msg_id in removed_msg_list_ids:
        messages_list.pop(removed_msg_id)
    
    for msg_id, messages in enumerate(messages_list):
        for pop_id in removed_msg_ids:
            if pop_id < len(messages):
                messages.pop(pop_id)

    return messages_list

def filter_no_change_checker(messages_list):
    """
        check the current accessibility v.s. the last one.
        If no change, then remove the message.
    """

    flag = False
    for msg_id, messages in enumerate(messages_list):
        if msg_id == 0:
            continue
        
        last_accessibility_tree = messages_list[msg_id - 1][-2]['content'].split('OBJECTIVE:')[0]
        current_accessibility_tree = messages_list[msg_id][-2]['content'].split('OBJECTIVE:')[0]
        
        if last_accessibility_tree == current_accessibility_tree or "The action you have chosen cannot be executed." in messages_list[msg_id][-2]['content']:
            flag = True
    return flag

def contain_scroll(messages_list):
    """
        check the current accessibility v.s. the last one.
        If no change, then remove the message.
    """

    flag = False
    for msg_id, messages in enumerate(messages_list):
        if msg_id == 0:
            continue
        
        if "scroll" in messages_list[msg_id][-1]['content']:
            return True
    return False

def add_type_cot(action, type_id):
    thought = action.split("Action:")[0].strip()
    add_cot = f" We should type in messages to the element [{type_id}], which has a combobox.\n\n"
    
    return action.replace(thought, thought + add_cot)

def fix_type_number(s, action, number):
    # Split the string into lines
    lines = s.split('\n')

    # Find the line starting with "[8] combobox"
    combobox_line_index = None
    for i, line in enumerate(lines):
        if line.strip("\t").startswith(f"[{number}] combobox"):
            combobox_line_index = i
            break

    # If the line is found, proceed with the random number generation and swapping
    if combobox_line_index is not None:
        # Generate a random number between 3 and 10 (inclusive)
        new_id = random.randint(4, 10)
        
        # Find the line starting with the new_id
        new_id_line_index = None
        for i, line in enumerate(lines):
            if line.startswith(f"\t[{new_id}] "):
                new_id_line_index = i
                break
        
        # Swap the lines if the new_id line is found
        if new_id_line_index is not None:
            # Change the number in the combobox line
            combobox_line = lines[combobox_line_index].replace(f"[{number}]", f"[{new_id}]")
            
            # Swap the lines
            lines[combobox_line_index] = lines[new_id_line_index].replace(f"[{new_id}]", f"[{number}]")
            lines[new_id_line_index] = combobox_line

    # Join the lines back into a single string
    result = '\n'.join(lines)
    action = action.replace(f"type [{number}]", f"type [{new_id}]")
    if random.random() < 0.5:
        action = add_type_cot(action, new_id)
    return result, action

# Add additional chain of thought of combobox

def fix_click_31(s, action):
    # Split the string into lines
    result = s
    lines = s.split('\n')

    # Find the line starting with "[8] combobox"
    combobox_line_index = None
    for i, line in enumerate(lines):
        if line.startswith("\t[31]"):
            combobox_line_index = i
            break

    # If the line is found, proceed with the random number generation and swapping
    if combobox_line_index is not None:

        new_id = random.randint(27, 31)
        
        # Find the line starting with the new_id
        new_id_line_index = None
        for i, line in enumerate(lines):
            if line.startswith(f"\t[{new_id}] "):
                new_id_line_index = i
                break
        
        # Swap the lines if the new_id line is found
        if new_id_line_index is not None:
            # Change the number in the combobox line
            combobox_line = lines[combobox_line_index].replace("[31]", f"[{new_id}]")
            
            # Swap the lines
            lines[combobox_line_index] = lines[new_id_line_index].replace(f"[{new_id}]", "[31]")
            lines[new_id_line_index] = combobox_line

    # Join the lines back into a single string
        result = '\n'.join(lines)
        
        action = action.replace("click [31]", f"click [{new_id}]")
    return result, action

def get_no_change_single_step(messages_list):
    # def filter_no_change(messages_list):
    """
        check the current accessibility v.s. the last one.
        If no change, then remove the message.
    """

    flag = False

    removed_msg_ids = []
    removed_msg_list_ids = []

    # no-change messages
    no_change_messages = []
    
    for msg_id, messages in enumerate(messages_list):
        if msg_id == 0:
            continue
        
        last_accessibility_tree = messages_list[msg_id - 1][-2]['content'].split('OBJECTIVE:')[0]
        current_accessibility_tree = messages_list[msg_id][-2]['content'].split('OBJECTIVE:')[0]

        
        if last_accessibility_tree == current_accessibility_tree or "The action you have chosen cannot be executed." in messages_list[msg_id][-2]['content']:
            # 1. remove this message
            # 2. remove the action and observation for future steps
            # print(msg_id)
            removed_msg_list_ids.append(msg_id - 1)
            # removed_msg_ids.append(len(messages_list[msg_id]) - 3)
            # removed_msg_ids.append(len(messages_list[msg_id]) - 2)
            no_change_messages.append( [
                {"role": "user",
                 "content": last_accessibility_tree},
            ] )
        
        # for ii in range(msg_id - 1, 0, -1):
        # if "The action you have chosen cannot be executed." in messages_list[msg_id][-2]['content']:
        #     removed_msg_list_ids.append(msg_id)
        #     removed_msg_ids.append(len(messages_list[msg_id]) - 1)
        #     removed_msg_ids.append(len(messages_list[msg_id]) - 2)

    removed_msg_ids = list(set(removed_msg_ids))
    removed_msg_list_ids = list(set(removed_msg_list_ids))
    # print(removed_msg_ids, removed_msg_list_ids)
    removed_msg_ids.sort(reverse=True)
    removed_msg_list_ids.sort(reverse=True)

    for removed_msg_id in removed_msg_list_ids:
        messages_list.pop(removed_msg_id)
    
    for msg_id, messages in enumerate(messages_list):
        for pop_id in removed_msg_ids:
            if pop_id < len(messages):
                messages.pop(pop_id)

    return messages_list

def get_whole_planning_trajectory(inner_loop_history):
    planning_trajectory = inner_loop_history[-1]
    for i in range(len(inner_loop_history)-1):
        planning_trajectory[2*i + 1]['content']  = inner_loop_history[i][2*i+1]['content']
    return planning_trajectory

def get_successul_unsuccessful_steps(messages):
    unsuccessful_steps = []
    successful_steps = []
    for i in range(1, len(messages)-2, 2):
        prev_observation = messages[i]['content']
        action = messages[i+1]['content']
        next_observation = messages[i+2]['content']
        if 'The action you have chosen cannot be executed.' in next_observation:
            unsuccessful_steps.append(
                [prev_observation, action, next_observation]
            )
        elif prev_observation == next_observation:
            unsuccessful_steps.append(
                [prev_observation, action, "The action you have chose brings no changes to the webpage."]
            )
        else:
            successful_steps.append([prev_observation, action, next_observation])
    if "```stop" in messages[-1]['content']:
        successful_steps.append([ messages[-2]['content'], messages[-1]['content'], "successfully stopped." ])
    else:
        unsuccessful_steps.append([ messages[-2]['content'], messages[-1]['content'], "last action did not stopped." ])
    return successful_steps, unsuccessful_steps

if __name__ == "__main__":
    # zero-shot
    all_data = {
        'search': ['search'],
        'mind2web': ['mind2web'],
        'mind2web_human': ['mind2web_human'],
        "webvoyager_human": ['webvoyager_human'],
        "webvoyager_self_instruct": ['webvoyager_self_instruct_old'],
    }

    # iter 1
    # all_data = {
    #     'search': ['search'],
    #     'mind2web': ['mind2web'],
    #     'mind2web_human': ['mind2web_human', 'mind2web_human_2'],
    #     "webvoyager_human": ['webvoyager_human'],
    #     "webvoyager_self_instruct": ['webvoyager_self_instruct'],
    # }

    outer_messages = []
    inner_planning_messages = []
    history_action_observations = []

    successful_action_list = []
    unsuccessful_action_list = []

    for key, res_list in all_data.items():

        try:
            results_dict, _ = gather_multiple_runs_jsonl(res_list)
        except:
            results_dict = load_jsonl_folder(res_list[0])
            results_dict = {i:item for i, item in enumerate(results_dict)}

        for i, res in results_dict.items():
            remove_example_flag = False
            outer_loop_history, inner_loop_history_list, histories = parse_history(res)

            history_action_observation = []

            return_any_answer = False

            formatted_raw_planning_code = []

            # do not support log in now.
            if len(inner_loop_history_list) < 1 or filter_login(inner_loop_history_list) > 0:
                continue


            for turn_i, messages in enumerate(inner_loop_history_list):
                raw_planning_code = messages[-1]['content']
                try:
                    action = prune_actions(raw_planning_code)
                except:
                    action = None
                    print("error pruning action:", raw_planning_code)

                if action is None:
                    print("bad", raw_planning_code)
                    remove_example_flag = True
                    break

                action_type, target_id, value, press_enter  = extract_info_from_action(action)
                
                if action_type is None:
                    # discard this example
                    print("bad", raw_planning_code)
                    remove_example_flag = True
                    break
                

                # print(extracted_action)
                action_formatted = action_type
                if target_id:
                    action_formatted += f" [{target_id}]"
                if value:
                    value = value.strip('""[]')
                    action_formatted += f" [{value}]"
                if press_enter is not None and action_type == "type":
                    if press_enter == True:
                        press_enter = ' [1]'
                    elif press_enter == '1':
                        press_enter = ' [1]'
                    else:
                        press_enter = ' [0]'
                    action_formatted += press_enter

                # change the format
                if not action == action_formatted:
                    # print(action, "\t", action_formatted)
                    raw_planning_code = raw_planning_code.replace(action, action_formatted)
                    # print(raw_planning_code)
                    messages[-1]['content'] = raw_planning_code

                formatted_raw_planning_code.append(raw_planning_code)
                for i in range(2, len(messages), 2):
                    messages[i]['content'] = formatted_raw_planning_code[i // 2 - 1]

                if turn_i == len(inner_loop_history_list) - 1:
                    if action_type == 'stop' and value != "N/A":
                        return_any_answer = True

            if len(inner_loop_history_list) > 0:
                full_inner_loop_history = get_whole_planning_trajectory(deepcopy(inner_loop_history_list))

                #### World Modeling messages
                successful_steps, unsuccessful_steps = get_successul_unsuccessful_steps(full_inner_loop_history)
                # observation action pairs that were unsuccessful / successful 
                successful_action_list.append(successful_steps)
                

                #### filter out those unsuccessful actions
                #### due to format issues.
                selected_unsuccessful_steps = []
                for msg in unsuccessful_steps:
                    remove_flag = False
                    raw_planning_code = msg[1]
                    try:
                        action = prune_actions(raw_planning_code)
                    except:
                        action = None
                        print("bad", raw_planning_code)

                    if action is None:
                        print("bad", raw_planning_code)
                        remove_flag = True
                        continue

                    action_type, target_id, value, press_enter  = extract_info_from_action(action)
                    
                    if action_type is None:
                        # discard this example
                        print("bad", raw_planning_code)
                        remove_flag = True
                    if not remove_flag:
                        selected_unsuccessful_steps.append(msg)

                unsuccessful_action_list.append(unsuccessful_steps)

            inner_loop_history_list = filter_wait(filter_no_change(deepcopy(inner_loop_history_list)))

            for messages in inner_loop_history_list:
                observation = messages[-2]['content']
                raw_planning_code = messages[-1]['content']

                #####
                # since there are too many type [8] and click [31]
                # randomly  change [8] and [31] to nearby numbers
                #####
                # if "[8] combobox" in observation and "type [8]" in raw_planning_code:
                #     observation, raw_planning_code = fix_type_number(observation, raw_planning_code, 8)
                #     messages[-2]['content'] = observation
                #     messages[-1]['content'] = raw_planning_code
                #     print("type\n", raw_planning_code)
                # if "[31]" in observation and "click [31]" in raw_planning_code:
                #     observation, raw_planning_code = fix_click_31(observation, raw_planning_code)
                #     messages[-2]['content'] = observation
                #     messages[-1]['content'] = raw_planning_code
                #     print("31\n", raw_planning_code)


            if not remove_example_flag:
                if return_any_answer:
                    outer_messages.append(outer_loop_history[:4]) # this version, do not train what's after stop
                    inner_planning_messages.append(inner_loop_history_list)

                try:
                    history_action_observation = [{'role': 'system',
                                                    'content': world_model_system_prompt},
                                                {'role' : 'user', 
                                                'content': "Initial operation: " + outer_loop_history[3]['content'] + f"\n\nOBJECTIVE: {outer_loop_history[1]['content']}"}]
                    for msg in inner_loop_history_list:
                        if msg[-2]['content'].split('OBJECTIVE:')[0] != "<|im_omitted|>":
                            history_action_observation.append({"role": "assistant", "content": msg[-2]['content'].split('OBJECTIVE:')[0] })
                            history_action_observation.append({"role": "user", "content": "Action: ```" + prune_actions(msg[-1]['content'])+ f"```\nOBJECTIVE: {msg[-2]['content'].split('OBJECTIVE:')[1].strip()}" })
                    # history_action_observation.pop(len(history_action_observation) - 1)
                    if len(history_action_observation) > 1:
                        history_action_observations.append(history_action_observation)
                except:
                    pass

                # break

                # 3. using llama3.3 as a critic to filter.

                #### get history of observation-action chain.

    # inner_planning_messages

    ### v2
    sft_data = outer_messages + list(chain(*inner_planning_messages))
    import random
    random.shuffle(sft_data)
    sft_data = [{'messages':item} for item in sft_data]
    save_jsonl(sft_data, "sft_llama3.3_70b_openwebvoyager_v2.jsonl")
