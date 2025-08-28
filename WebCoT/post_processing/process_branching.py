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

llama3_wm_wv = load_jsonl_folder("./data/llama3.3-70B-wm-train-set-sampling-googlesearch")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70b-wm-train-set-sampling")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70B-wm-train-set-sampling-googlesearch-latest")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70b-wm-train-set-sampling-latest")

from openai import AzureOpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
client = AzureOpenAI(
            azure_endpoint = "", 
            api_key="",  
            api_version="2024-02-01"
        )

def worker(item):
    time.sleep(1)
    try:
        outer_loop_history, inner_loop_history, histories, _ = parse_history(item)
        output = auto_eval_by_gpt4o(client, outer_loop_history, histories)
        if type(output) == int or type(output) == str:
            return output
        else:
            return 'None'
    except Exception as e:
        print(e)
        return 'None'

gpt_4_res_list = []
with ThreadPoolExecutor(128) as executor:
    for res in tqdm(executor.map(worker, llama3_wm_wv), total=len(llama3_wm_wv)):
        gpt_4_res_list.append(True if res == 1 else False)


sft_data = []
llama3_wm_wv = load_jsonl_folder("./data/llama3.3-70B-wm-train-set-sampling-googlesearch")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70b-wm-train-set-sampling")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70B-wm-train-set-sampling-googlesearch-latest")
llama3_wm_wv += load_jsonl_folder("./data/llama3.3-70b-wm-train-set-sampling-latest")
gpt_additional_restriction_prompt = "When generating Action, please include [] for [id] and [content] and [1] for pressing enter."
webdreamer_system_message = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current observation (web page's accessibility tree): This is a simplified representation of the webpage, providing key information. Optionally, you may be provided with a screenshot of the webpage. You should pay close attention to the screesnhot to make decisions.
The open tabs: These are the tabs you have open.
The previous actions: You can refer to the conversation history with the user to see the actions you have taken. It may be helpful to track your progress.

The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the original homepage at first. When you can't find information in some websites, try starting over from the beginning.
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.

To be successful, it is very important to follow the following rules:
1. If you are uncertain about the next action, follow these steps: First, generate up to three of the most likely and valid actions based on the current observation. Then, for each of these possible actions, simulate and describe the expected future outcome in free text, detailing the next observation that would result from performing the action. Next, evaluate the correctness of each action by considering both the current observation and the simulated future results. Assign a numerical score from 0 to 1 to indicate the likelihood of correctness for each action: a score of 1.0 means "complete", 0.5 means "on track", and 0 means "incorrect". Provide your rationale for each score before assigning it. Finally, select and output the action with the highest score from the evaluated actions.
2. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.
3. You should only issue one action at a time.
4. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label.
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
6. If you ever need to login, login with Google, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.
{gpt_additional_restriction_prompt if False else ""}
Your reply should strictly follow the format:
<think>
1. Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}}
   Possible Step: {{One of the logical and valid actions to take based on the current observation.}}
   Simulated Output: {{A prediction of what the next observation or result will be after performing the action.}}
   Critic Evaluation: {{Your rationale on the effectiveness of the action as well as a score from 0 (poor performance) to 1 (excellent performance), judging the corresponding action's s effectiveness.}}
2. ... (continue with subsequent steps as needed in the same format)
</think> (Optional: You can choose to include the steps between `<think>` and `</think>` if necessary or skip them based on the task's complexity.) 
Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}} Action: ```{{The final action you choose to take in the process.}}```"""

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

def sanity_check_action(s):

    try:
        extract_info_from_action(prune_actions(s))
        return 1
    except:
        print("error", s)
        return 0
    
def sim_log2text(sim_log):
    res = ''
    for i, l in enumerate(sim_log):
        thought = l["Action"].split("Action: ")[0].replace("Thought: ", "").replace("\n", "").strip()
        act = l["Action"].split("Action: ")[1].replace("\n", "").strip()
        res += f'{i + 1}. Thought: {thought}\n'
        res += f'   Possible Step: {act}\n'
        res += f'   Simulated Output: {l["Simulated Output"]}\n'
        res += f'   Critic Evaluation: {l["Critic Evaluation"]}\n'
        
    return res

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
                    'role': 'user',
                    'content': message['content'].split("</think>\n")[-1]
                }
            )
        else:
            res_list.append(message)
    
    return res_list

import random
random.seed(0)
        
def add_type_cot(action, type_id):
    thought = action.split("Action:")[0].strip()
    add_cot = f" We should type in messages to the element [{type_id}], which has a combobox.\n\n"
    #action = action.replace("We should type in messages to the element [3], which has a combobox.", "")
    return action.replace(thought, thought + add_cot)

def fix_type_number(s, action, number):
    # Split the string into lines
    lines = s.split('\n')

    # Find the line starting with "[8] combobox"
    combobox_line_index = None
    for i, line in enumerate(lines):
        if line.strip("\t").startswith(f"[{number}] textbox"):
            combobox_line_index = i
            break

    # If the line is found, proceed with the random number generation and swapping
    if combobox_line_index is not None:
        # Generate a random number between 3 and 10 (inclusive)
        new_id = random.randint(2, 4)
        
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

def find_loop(traj_list):
    def _find_same_obs(traj_id, obs):
        for i in reversed(range(traj_id + 1, len(traj_list))):
            if obs == traj_list[i][0]:
                return i
            
        return -1
    
    
    for traj_id, (obs, _, _) in enumerate(traj_list):
        same_idx = _find_same_obs(traj_id, obs)
        if same_idx != -1:
            return traj_id, same_idx
        
    return -1, -1

id_list = []
action_list = []
inner_cnt = 0
success_cnt = 0
from tqdm import tqdm
with open("./long_cot_rej_sampling_v3_wo_loop_aligned.jsonl") as fin:
    llama3_data_list = [json.loads(line)['messages'] for line in fin]
    
zeroshot_success_query = set()
wo_loop_sys_prompt = 'You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere\'s the information you\'ll have:\nThe user\'s objective: This is the task you\'re trying to complete.\nThe current observation (web page\'s accessibility tree): This is a simplified representation of the webpage, providing key information. Optionally, you may be provided with a screenshot of the webpage. You should pay close attention to the screesnhot to make decisions.\nThe open tabs: These are the tabs you have open.\nThe previous actions: You can refer to the conversation history with the user to see the actions you have taken. It may be helpful to track your progress.\n\nThe actions you can perform are the following:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.\n`scroll [direction=down|up]`: Scroll the page up or down.\n`wait`: Wait for the page to load, with a duration of 5 seconds.\n`goback`: Navigate to the previously viewed page.\n`restart`: Navigate to the Google search homepage. When you can\'t find information in some websites, try starting over from Google search.\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.\n2. You should only issue one action at a time.\n3. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.\n4. Issue stop action when you think you have achieved the objective. Don\'t generate anything after stop.\n5. If you ever need to login, login with Google, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.\n\nYour reply should strictly follow the format:\nThought: {Your brief thoughts (briefly summarize the info that will help complete the task)} Action: ```{the next action you choose to take}```'

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
    if data[1]['content'].startswith("Now given a task:"):
        zeroshot_success_query.add(data[1]['content'].replace("www.google.com", "www.bing.com"))
    
    if data[0]['content'] == wo_loop_sys_prompt:
        assert data[0]['role'] == 'system'
        data[0]['content'] = webdreamer_system_message
        
    sft_data.append(data)

print('before sft len:', len(sft_data))

for wm_wv, gpt_4_res in tqdm(zip(llama3_wm_wv[0:], gpt_4_res_list)):
    if wm_wv is None: #or gpt_4_res == False:
        continue
    
    outer_loop_history, inner_loop_history, histories, simu_log = parse_history(wm_wv)
    if outer_loop_history[1]['content'] in zeroshot_success_query:
        continue
    
    zeroshot_success_query.add(outer_loop_history[1]['content'])
    
    if len(inner_loop_history) < 1 or filter_login(inner_loop_history) > 0:
        continue
    
    if inner_loop_history == [] or inner_loop_history[-1][-1]['content'] == 'Failed:500' \
        or inner_loop_history[-1][-1]['content'] == 'Failed:502':
        continue
    
    if "Lattice-Based Zero-Knowledge Proofs in Action: Applications to Electronic Voting" in inner_loop_history[-1][-1]['content'] or len(inner_loop_history[-1][-1]['content'].split("Action:")) == 1:
        continue
    
    parsed_res, _, _ = extract_action_for_web("", inner_loop_history[-1][-1]['content'], "")
    if parsed_res['action_name'] != 'stop':
        continue
    
    '''prev_len = len(inner_loop_history)
    prev_hist = deepcopy(inner_loop_history)
    inner_loop_history = filter_wait(filter_no_change(deepcopy(inner_loop_history)))
    if prev_len != len(inner_loop_history):
        print("wait or no change filtered")
        #continue'''
    
    sft_data.append(outer_loop_history[:-2])
    success_cnt += 1
    message_list = [
        {
            'role': 'system',
            'content': webdreamer_system_message
        }
    ]
    
    inner_loop_history_list = []
    aligned_sim_log_list = align_sim_log(deepcopy(inner_loop_history), json.loads(simu_log['raw_data']))
    loop_st, loop_nd = find_loop(aligned_sim_log_list)
    if loop_st != -1 and loop_nd != -1:
        print('len(sft_data), loop_st, loop_nd', len(sft_data), loop_st, loop_nd)
        while loop_st != -1 and loop_nd != -1:
            aligned_sim_log_list = aligned_sim_log_list[:loop_st] + aligned_sim_log_list[loop_nd:]
            loop_st, loop_nd = find_loop(aligned_sim_log_list)
            print('further cleaning: loop_st, loop_nd', loop_st, loop_nd)
        
        print("wo_loop_traj_list", aligned_sim_log_list)
        loop_cnt += 1
    
    for obs, action, aligned_log in aligned_sim_log_list:
        message_list.append(obs)
        if aligned_log is None:
            message_list.append(
                {
                    'role': 'assistant',
                    'content': action
                }
            )
        else:
            message_list.append(
                {
                    'role': 'assistant',
                    'content': f'<think>\n{sim_log2text(aligned_log)}\n</think>\n{action}'
                }
            )

        inner_loop_history_list.append(omit_obs_think(message_list))


    inner_loop_history_list = filter_wait(filter_no_change(deepcopy(inner_loop_history_list)))
    #all_format_correct = 1
    for messages in inner_loop_history_list:
        observation = messages[-2]['content']
        raw_planning_code = messages[-1]['content']

        #####
        # Fixing type [8] and click [31]
        #####
        if "[3] textbox" in observation and "type [3]" in raw_planning_code:
            observation = observation.replace("We should type in messages to the element [3], which has a combobox.", "")
            observation, raw_planning_code = fix_type_number(observation, raw_planning_code, 3)
            messages[-2]['content'] = observation
            messages[-1]['content'] = raw_planning_code
            #print('observation', observation)
            #print('raw_planning_code', raw_planning_code)
            #break
            
        assert sanity_check_action(raw_planning_code.split("</think>\n")[-1]) == 1
            
    for hist in inner_loop_history_list:
        tmp_action = hist[-1]['content']
        parsed_res, _, _ = extract_action_for_web("", tmp_action.split("</think>\n")[-1], "")
        action_list.append(parsed_res["action_name"])
        id_list.append(parsed_res["target_id"])
        #st_cnt += 1
        
    #print(len(inner_loop_history_list), st_cnt)
    inner_cnt += len(inner_loop_history_list)
    sft_data.extend(inner_loop_history_list)
    #break
    
print(inner_cnt, success_cnt)
print('after sft len:', len(sft_data))

#import random
random.shuffle(sft_data)
save_jsonl([{"messages": item} for item in sft_data], f"./webcot_branching.jsonl")
