import re
import os
import requests
from datetime import datetime
from database import (
    update_or_create_session,
    update_or_create_rawdata,
    update_or_create_annotation,
)
import json
import copy
from evaluator.openai_evaluator import SYSTEM_PROMPT_STEP_TEXT, USER_PROMPT_STEP_TEXT, parse_eval_output
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

WEB_IP = os.getenv("WEB_IP", "/web:3000")
WEBDREAMER_COT = os.environ.get("WEBDREAMER_COT", 'false')
INFERENCE_MODEL_NAME = os.environ.get("INFERENCE_MODEL_NAME", "DEFAULT")

def extract_info_from_action(generated_action):
    # zero-shot llama 3.3 additional parsing rules
    if os.environ.get('LLAMA_VERSION') == '3.3': 
        print("llama3.3 parsing")
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
        or generated_action.lower().startswith("wait")
        or generated_action.lower().startswith("stop")
    ):
        if generated_action.lower().startswith("goback"):
            action_name = "goback"
        elif generated_action.lower().startswith("restart"):
            action_name = "restart"
        elif generated_action.lower().startswith("wait"):
            action_name = "wait"
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


def get_browser(storage_state, geo_location):
    url = "http://web:3000/getBrowser"
    data = {"storageState": storage_state, "geoLocation": geo_location}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["browserId"]
        else:
            return None
    except requests.RequestException as e:
        print(f"Request Error: {e}")


def close_browser(browser_id):
    url = "http://web:3000/closeBrowser"
    data = {"browserId": browser_id}
    print(f"Closing browser {browser_id}")
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Closing browser succeeded")
            return None
        else:
            print("closing browser", response.status_code)
            return None
    except requests.RequestException as e:
        print(f"Request Error: {e}")

import aiohttp
def close_all_browser():
    url = "http://web:3000/closeAllBrowsers"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return None
        else:
            return None
    except requests.RequestException as e:
        print(f"Request Error: {e}")

# async def close_all_browser():
#     url = "http://web:3000/closeAllBrowsers"
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url) as response:
#                 if response.status == 200:
#                     return None
#                 else:
#                     return None
#     except aiohttp.ClientError as e:
#         print(f"Request Error: {e}")

def open_page(browser_id, target_url):
    url = "http://web:3000/openPage"
    data = {"browserId": browser_id, "url": target_url}

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return True, response.json()["pageId"]
        else:
            print(f"Open page Request failed with status code: {response.status_code}")
            return False, ""
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, ""


def get_accessibility_tree(browser_id, page_id, current_round=-1, timeout=30):
    if current_round > 0:
        url = "http://web:3000/getAccessibilityTree"
    else:
        url = "http://web:3000/getAccessibilityTreeOnly"
        # this does not require a round number to be used
        # for retrieving the screenshot.
    data = {
        "browserId": browser_id,
        "pageId": page_id,
        "currentRound": current_round,
    }

    try:
        response = requests.post(url, json=data, timeout=timeout)
        if response.status_code == 200:
            res_json = response.json()
            AccessibilityTree = res_json.get("yaml", [])
            curr_url = res_json.get("url", "")
            snapshot = res_json.get("snapshot", "")
            idx2element = res_json.get("treeIdxtoElement", {})
            return True, AccessibilityTree, curr_url, snapshot, idx2element
        else:
            print(
                f"Get accessibility tree Request failed with status code: {response.status_code}"
            )
            return False, None, None, None, None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, None, None, None, None


def perform_action(browser_id, page_id, action, timeout=30):
    print("ftqftq", browser_id, page_id, action)
    url = "http://web:3000/performAction"
    data = {
        "browserId": browser_id,
        "pageId": page_id,
        "actionName": action["action_name"],
        "targetId": action["target_id"],
        "targetElementType": action["target_element_type"],
        "targetElementName": action["target_element_name"],
        "actionValue": action["action_value"],
        "needEnter": action["need_enter"],
    }
    try:
        response = requests.post(url, json=data, timeout=timeout)
        if response.status_code == 200:
            response_data = response.json()
            current_url = response_data.get('currentUrl')
            return True, current_url
        else:
            print(
                f"Request failed with status code: {response.status_code} {response.text} on action {action['action_name']}"
            )
            return False, None
    except requests.RequestException as e:
        print(f"Request failed: {e} on action {action['action_name']}")
        return False, None

def is_annoying(current_accessbility_tree):
    if "See results closer to you?" in current_accessbility_tree and len(current_accessbility_tree.split("\n")) <= 10:
        return True
    return False

def get_skip_action(current_accessbility_tree):
    action_name, target_id, action_value, need_enter = extract_info_from_action(
        "click [5]"
    )
    target_id, target_element_type, target_element_name = find_target_element_info(
        current_accessbility_tree, target_id, action_name
    )
    return {
            "action_name": action_name,
            "target_id": target_id,
            "action_value": action_value,
            "need_enter": need_enter,
            "target_element_type": target_element_type,
            "target_element_name": target_element_name,
        }

gpt_additional_restriction_prompt = "When generating Action, please include [] for [id] and [content] and [1] for pressing enter."

system_message = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current observation (web page's accessibility tree): This is a simplified representation of the webpage, providing key information. Optionally, you may be provided with a screenshot of the webpage. You should pay close attention to the screesnhot to make decisions.
The open tabs: These are the tabs you have open.
The previous actions: You can refer to the conversation history with the user to see the actions you have taken. It may be helpful to track your progress.

The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`wait`: Wait for the page to load, with a duration of 5 seconds.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the Google search homepage. When you can't find information in some websites, try starting over from Google search.
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.
2. You should only issue one action at a time.
3. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
4. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
5. If you ever need to login, login with Google, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.
{gpt_additional_restriction_prompt if os.environ['MODEL_NAME']!= "ck" else ""}
Your reply should strictly follow the format:
Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}} Action: ```{{the next action you choose to take}}```"""

webdreamer_cot_system_message = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

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
{gpt_additional_restriction_prompt if os.environ['MODEL_NAME']!= "ck" else ""}
Your reply should strictly follow the format:
<think>
1. Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}}
   Possible Step: {{One of the logical and valid actions to take based on the current observation.}}
   Simulated Output: {{A prediction of what the next observation or result will be after performing the action.}}
   Critic Evaluation: {{Your rationale on the effectiveness of the action as well as a score from 0 (poor performance) to 1 (excellent performance), judging the corresponding action's s effectiveness.}}
2. ... (continue with subsequent steps as needed in the same format)
</think> (Optional: You can choose to include the steps between `<think>` and `</think>` if necessary or skip them based on the task's complexity.) 
Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}} Action: ```{{The final action you choose to take in the process.}}```"""


# 5. If you ever need to login, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.

def web_action(
        llm_connection,
        browser_id,
        target_url=None,
        raw_action_code=None,
        current_accessbility_tree='',
        page_id=0,
        expanded_part=None,
        downloaded_files=[],
        counter=0,
):
    '''
        Inputs:
            browser_id:
            target_url: 
                if raw_action_code is None, then just jump to target_url,
                or if the action forces the browser to a specific url.
            raw_action_code: model-generated planning code.
            page_id: TBA
            expanded_part: TBA
            downloaded_files: not used.
            counter: depth.
        Returns:
            Whether the action succeeded or not.
            Tuple: 
                page_id:
                get_accessibility_tree_succeed: 
                current_accessbility_tree: accessibility tree.
                step_url: current url
    '''
    
    if raw_action_code:

        print("raw_action_code: ", raw_action_code)
        
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, raw_action_code, expanded_part
        )

        print("In web_action extracted_action: ", extracted_action)

        action_succeed, target_url = perform_action(browser_id, page_id, extracted_action)
        
        if not action_succeed:
            return False, (page_id, False, current_accessbility_tree, None, None, None)
        else:
            get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                get_accessibility_tree(browser_id, page_id, counter)
            )
            if 'downloaded the following files:' in current_accessbility_tree:
                path_str = current_accessbility_tree.split('downloaded the following files:')[1].strip().split('OBJECTIVE:')[0]
                for path in path_str.split('\n'):
                    downloaded_files.append(path)#.split(':')[1].strip())
            if is_annoying(current_accessbility_tree):
                skip_this_action = get_skip_action(current_accessbility_tree)
                perform_action(browser_id, page_id, skip_this_action)
                get_accessibility_tree_succeed, current_accessbility_tree, step_url, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
            if "Cookie banner" in current_accessbility_tree:
                cookie_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "\nOBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                )
                popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                popup_messages.append({"role": "user", "content": cookie_message})
                next_action = llm_connection.get_response(popup_messages)
                extracted_action, action_string, extracted_thought = extract_action_for_web(
                    current_accessbility_tree, next_action, expanded_part
                )
                print ('Tring to close the pop-up window', next_action)
                perform_action(browser_id, page_id, extracted_action)

                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                print ('close the pop-up window attempted')
            current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
            # if current_accessbility_tree == prev_obs and prev_action == extracted_action:
            #     repeat_count += 1
            # else:
            #     repeat_count = 0
    else:
        _, page_id = open_page(browser_id, target_url)
        get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = get_accessibility_tree(
            browser_id, page_id, 0
        )

    return True, (page_id, get_accessibility_tree_succeed, current_accessbility_tree, step_url, expanded_part, idx2element)


gpt4_world_model_prompt  = """
You are a web server. You are given the current observed accessibility tree of the web page, and an action to perform. 
The expected output is a short description on what the next observation is, in the form of free text.

The definitions of the actions are as follows: The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the original home page and restart the action.
"""

world_model_evaluation_prompt = """
You are an evaluator of a web agent task, evaluating the correctness of the action, conditioned on the current observation and a simulated future result. 

You are given the task query, the current observed accessibility tree, the action performed, and a textual description of the simulated output after performing this action.

You are expected to give a numerical score (0 to 1) to indicate whether the simulated output is correct. The higher the score, the more likely the action is correct. 

Here are some example scores: complete (1.0), on track (0.5), or incorrect (0).

Output your rationale first and then the score.

Output format:

Thought: XXXX. Score: {a score from 0 to 1}.
"""

messages = [{"role": "system", "content": world_model_evaluation_prompt},
            {"role": "user", "content": "OBJECTIVE: JQuery selector for elements with specific class\n\nOBSERVATION: Tab 0 (current): Google\n\n[1] RootWebArea 'Google' focused: true\n\t[2] link 'Gmail '\n\t[3] link '搜尋圖片 '\n\t[4] button 'Google 應用程式' expanded: false\n\t[5] link '登入'\n\t[6] image '2024 年佳節'\n\t[7] combobox '搜尋' focused: true autocomplete: both hasPopup: listbox required: false expanded: false\n\t[8] button '分享'\n\nAction: type [7] [JQuery selector for elements with specific class] [1]\n\nSimulated output of the action: The next observation will show that the search query \"JQuery selector for elements with specific class\" has been typed into the search box (combobox) and the \"Enter\" key has been pressed. This will likely lead to a new page displaying search results related to that query."},
]

# The format of accessibility tree:

# Tab 0 (current): Google\n\n[1] RootWebArea 'Google' focused: true\n\t[2] link 'Gmail '\n\t[3] link '搜尋圖片 '\n\t[4] button 'Google 應用程式' expanded: false\n\t[5] link '登入'\n\t[6] image '2024 年佳節'\n\t[7] combobox '搜尋' focused: true autocomplete: both hasPopup: listbox required: false expanded: false\n\t[8] button '分享'

# The format of action:

# type [7] [JQuery selector for elements with specific class] [1]

import numpy as np
def world_model_imagination_selection_gpt(next_actions, objective, current_messages, llm_connection, world_model_connection, critic_model_connection):

    try:
        current_observation = current_messages[-1]['content'].split('OBSERVATION:')[1].strip().split('OBJECTIVE:')[0].strip()
    except:
        current_observation = current_messages[-1]['content']


    # print("WM::current_observation: ", current_observation)
    predictions = []
    predictions_thought = []
    simulated_outputs = []
    for current_action in next_actions:
        # if the action is stop, then there's no need to simulate
        if '```stop' in current_action:
            wm_eval_messages = [{"role": "system", "content": world_model_evaluation_prompt},
                    {"role": "user", "content": f"OBJECTIVE: {objective}\n\nOBSERVATION: {current_observation}\n\nAction: {current_action}. The web agent task is stopped and the current results are returned. Evaluate whether the task should be stopped and returned with the current value."},
                ]
        else:
            wm_messages = [{"role": "system", "content": gpt4_world_model_prompt},
                    {"role": "user", "content": f"OBSERVATION: {current_observation}\n\nAction: {current_action}"},
            ]
            
            simulated_output = world_model_connection.get_response(wm_messages)
            simulated_outputs.append(simulated_output)


            wm_eval_messages = [{"role": "system", "content": world_model_evaluation_prompt},
                    {"role": "user", "content": f"OBJECTIVE: {objective}\n\nOBSERVATION: {current_observation}\n\nAction: {current_action}\n\Simulated output of the action: {simulated_output}"},
                ]
        simulated_output = critic_model_connection.get_response(wm_eval_messages)
        predictions_thought.append(simulated_output)

        try:
            simulated_output = float(simulated_output.split('Score:')[1].strip().strip('.'))
        except:
            simulated_output = 0


        predictions.append(simulated_output)
        # evaluate
    
    # return the action with the highest score.
    print("WM::predictions", predictions, predictions_thought)
    return next_actions[np.argmax(predictions)]

# Using world model to synthesize trajectories

gpt_additional_restriction_prompt = ""
# "When generating Action, please include [] for [id] and [content] and [1] for pressing enter."

policy_model_system_message = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

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
`restart`: Navigate to the Google search homepage. When you can't find information in some websites, try starting over from Google search.
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation. For example, you should NOT type into buttons or click on statictext.
2. You should only issue one action at a time.
3. STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
4. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
5. If you ever need to login, login with Google, use cognitivekernel@gmail.com as the email and IsAwesome as the password. mafinder189@gmail.com is the recovery email, enter it when required. Try to skip any follow-up questions that may appear after logging in.
{gpt_additional_restriction_prompt}
Your reply should strictly follow the format:
Thought: {{Your brief thoughts (briefly summarize the info that will help complete the task)}} Action: ```{{the next action you choose to take}}```"""

world_model_system_prompt = """You are an intelligent assistant designed to interact with web pages through an accessibility tree. Your task is to predict the accessibility tree of the next web page based on the given starting accessibility tree and a specified action.
The format of accessibility tree:

Tab 0 (current): Google\n\n[1] RootWebArea 'Google' focused: true\n\t[2] link 'Gmail '\n\t[3] link '搜尋圖片 '\n\t[4] button 'Google 應用程式' expanded: false\n\t[5] link '登入'\n\t[6] image '2024 年佳節'\n\t[7] combobox '搜尋' focused: true autocomplete: both hasPopup: listbox required: false expanded: false\n\t[8] button '分享'

The format of action:

type [7] [JQuery selector for elements with specific class] [1]

which indicates typing "JQuery selector for elements with specific class" into the field with id 7, corresponding to the combobox (search box) on the Google homepage.

The definitions of the actions are as follows: The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the Google search homepage. When you can't find information in some websites, try starting over from Google search.
"""

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

from copy import deepcopy

def get_multi_step_wm_simulation(initial_policy, next_action, all_messages,
                                 world_model_connection, llm_connection, search_depth=1):
    """
        Give history and the current action, predict and sample the next n observation (n=search_depth).
    """

    output_simulations = []

    ### parse history

    if 'restart' in next_action:
        # the original web page
        return [all_messages[1]['content'].split('OBSERVATION:')[1].strip()]

    query = initial_policy.split('query="')[1].split('",')[0]

    all_messages.append({"role": "assistant", "content": next_action})

    # print("in get_multi_step_wm_simulation", all_messages)

    depth = 0
    while depth < search_depth:

        depth += 1
        # print(depth)

        world_model_message = [{'role': 'system',
                    'content': world_model_system_prompt},
                    {"role": "user", "content": f"Initial operation: {initial_policy}\n\nOBJECTIVE: {query} "}]\
                            + deepcopy(all_messages[1:])
        # swap the roles of user and assistant

        for j in range(1, len(world_model_message), 2):
            world_model_message[j]['role'] = 'user'
            if j >= 3:
                world_model_message[j]['content'] = prune_actions(world_model_message[j]['content']) + "\nOBJECTIVE: " + query

            # if j - 1 > 0:
            #     world_model_message[j - 1]['role'] = 'assistant'

        for j in range(0, len(all_messages[1:]) - 2, 2):
            world_model_message[j + 2]['role'] = 'user'
            world_model_message[j + 2]['content'] = 'The previous observations are omitted.'
        

        next_accessibility_tree = world_model_connection.get_response(world_model_message, temperature=0.7, do_print=False)

        
        
        if 'OBSERVATION:' in next_accessibility_tree:
            current_accessbility_tree = next_accessibility_tree.split('OBSERVATION:')[1].strip()
        elif '[1] RootWebArea' in next_accessibility_tree:
            current_accessbility_tree = '[1] RootWebArea' + next_accessibility_tree.split('[1] RootWebArea')[1].strip()
        else:
            # next_accessibility_tree = world_model_connection.get_response(world_model_message, temperature=0.7, do_print=False)
            
            # if 'OBSERVATION:' in next_accessibility_tree:
            #     current_accessbility_tree = next_accessibility_tree.split('OBSERVATION:')[1].strip()
            # elif '[1] RootWebArea' in next_accessibility_tree:
            #     current_accessbility_tree = '[1] RootWebArea' + next_accessibility_tree.split('[1] RootWebArea')[1].strip()
            # else:
            #     print("error", next_accessibility_tree)
            current_accessbility_tree = next_accessibility_tree

        print("WM::multi-step", depth, next_action, current_accessbility_tree)
        output_simulations.append(current_accessbility_tree)
        
        if depth >= search_depth:
            return output_simulations

        current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "\nOBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message})
        all_messages.append({"role": "user", "content": current_user_message})
        next_action = llm_connection.get_response(current_messages, temperature=0.7, do_print=False)

        current_messages.append({"role": "assistant", "content": next_action})
        all_messages.append({"role": "assistant", "content": next_action})

        try:
            next_action = prune_actions(next_action)
        except:
            # stop?
            print("action parsing error", next_action)
            return output_simulations
            
        if 'stop' in next_action:
            return output_simulations
        elif 'restart' in next_action:
            # the original web page
            return [all_messages[1]['content'].split('OBSERVATION:')[1].strip()]
    return []


def world_model_imagination_selection_ours(initial_policy, next_actions, objective, all_messages, 
                                           llm_connection, world_model_connection, critic_model_connection,
                                           search_depth=1):


    try:
        current_observation = all_messages[-1]['content'].split('OBSERVATION:')[1].strip().split('OBJECTIVE:')[0].strip()
    except:
        current_observation = all_messages[-1]['content']


    # print("WM::current_observation: ", current_observation)
    # predictions = []
    # predictions_thought = []
    # simulated_outputs = []

    def worker_func(item):
        i, current_action = item
        
        if '```stop' in current_action:
            wm_eval_messages = [{"role": "system", "content": world_model_evaluation_prompt},
                    {"role": "user", "content": f"OBJECTIVE: {objective}\n\nOBSERVATION: {current_observation}\n\nAction: {current_action}. The web agent task is stopped and the current results are returned. Evaluate whether the task should be stopped and returned with the current value."},
                ]
            simulated_output = ["stopped"]
        else:
            try:
                simulated_output = get_multi_step_wm_simulation(initial_policy, current_action, copy.deepcopy(all_messages),
                                world_model_connection[i], llm_connection, search_depth=search_depth)
            except:
                simulated_output = [""]
            # simulated_output.append(simulated_output)

            if len(simulated_output) == 0:
                simulated_output = [""]
            elif len(simulated_output) == 1:
                # depth = 1 or only one step
                simulated_output = simulated_output[0]
            else:
                simulated_output = "\n\n".join([f"Step {_+1}: {simulated_output[_]}." for _ in range(len(simulated_output))])
            wm_eval_messages = [{"role": "system", "content": world_model_evaluation_prompt},
                    {"role": "user", "content": f"OBJECTIVE: {objective}\n\nOBSERVATION: {current_observation}\n\nAction: {current_action}\n\Simulated output of the action: {simulated_output}"},
            ]
        
        critic_output = critic_model_connection.get_response(wm_eval_messages)
        # predictions_thought.append(simulated_output)

        try:
            score = float(critic_output.split('Score:')[1].strip().strip('.'))
        except:
            score = 0

        return (simulated_output, critic_output, score)
        # predictions.append(score)


    start = time.time()
    res_list = []
    with ThreadPoolExecutor(len(next_actions)) as executor:
        for res in tqdm(executor.map(worker_func, list(enumerate(next_actions)) ), total=len(next_actions)):
            res_list.append(res)

    # res_list = [worker_func(item) for item in enumerate(next_actions)]

    simulated_outputs = [res[0] for res in res_list]
    predictions_thought = [res[1] for res in res_list]
    predictions = [res[2] for res in res_list]

    end = time.time()
    print("WM: wm prompting time", end - start )

    # for current_action in next_actions:
        # if the action is stop, then there's no need to simulate
        
        # evaluate
    
    # return the action with the highest score.
    print("WM::simulated_outputs", simulated_outputs)
    print("WM::predictions", predictions, predictions_thought)
    return next_actions[np.argmax(predictions)]

def call_web(
    llm_connection,
    query,
    target_url,
    session_id,
    message_id,
    username,
    max_steps=12,
    world_model_search=None,
    search_depth=1,
    world_model_connection=None,
    critic_model_connection=None,
    storage_state=None,
    geo_location=None,
    yield_full_message=False,
):
    """Makes an asynchronous GET request to a target URL with a query parameter.

    Args:
        query (str): The query parameter value to be sent with the request.
        target_url (str): The target URL to which the request is made.

    Returns:
        dict: The JSON response from the target URL, if successful.
    """
    if not world_model_search in ['worldmodel', 'webdreamer']:
        world_model_search = None
    
    max_steps = int(os.environ.get("MAX_STEPS", max_steps))
    # Use aiohttp.ClientSession for making the HTTP request
    browser_id = get_browser(storage_state, geo_location)
    if browser_id is None:
        yield None

    if WEBDREAMER_COT == 'true':
        all_messages = [{"role": "system", "name": "head", "content": webdreamer_cot_system_message}]
    else:
        all_messages = [{"role": "system", "name": "head", "content": system_message}]

    _, page_id = open_page(browser_id, target_url)
    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = get_accessibility_tree(
        browser_id, page_id, 0
    )
    counter = 0
    repeat_count = 0
    prev_obs = current_accessbility_tree
    prev_action = None
    expanded_part = None
    returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] goto {target_url}\n"
    yield returned_info
    action_error = False
    response_error = False
    downloaded_files = []
    while get_accessibility_tree_succeed:

        if counter > max_steps:
            break
        if action_error:
            current_user_message = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
            action_error = False
        elif response_error:
            current_user_message = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            response_error = False
        else:
            if repeat_count >= 2:
                print ('Browsing is getting stuck, stop here')
                break
                # current_user_message = (
                # "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: Close any pop-up window in the current page, if there is no close button you can accept the pop-up." 
                # )
                # popup_messages = []
                # for tmp_message in all_messages:
                #     if tmp_message["role"] == "user":
                #         popup_messages.append({"role": "user", "content": "<|im_omitted|>"})
                #     else:
                #         popup_messages.append(tmp_message)
                # popup_messages.append({"role": "user", "content": current_user_message})
                # next_action = llm_connection.get_response(popup_messages)
                # extracted_action, action_string, extracted_thought = extract_action_for_web(
                #     current_accessbility_tree, next_action
                # )
                # print ('Tring to close the pop-up window')
                # action(browser_id, page_id, extracted_action)

                # get_accessibility_tree_succeed, current_accessbility_tree, step_url = (
                #     get_accessibility_tree(browser_id, page_id, counter)
                # )
                # print ('close the pop-up window attempted')
            current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "\nOBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message, 'step_url': step_url, 'idx2element': idx2element})
        all_messages.append({"role": "user", "content": current_user_message})


        print("WM inference type:", world_model_search)

        if world_model_search is not None and counter >= 2:
            # the first a few steps does not require wm sampling.

            next_actions = [llm_connection.get_response(current_messages)]

            try:

                # for _ in range(2):
                tmp_messages = copy.deepcopy(current_messages)
                try:
                    first_action = next_actions[0].split("Action:")[1].strip().strip("```")
                except:
                    first_action = next_actions[0].split("Action:")[1].strip()
                tmp_messages[-1]['content'] += f"\nPlease generate actions different from { first_action }. "
                next_actions.append(llm_connection.get_response(tmp_messages, temperature=1))
            except:
                pass

            try:
                tmp_messages = copy.deepcopy(current_messages)
                try:
                    second_action = next_actions[1].split("Action:")[1].strip().strip("```")
                except:
                    second_action = next_actions[1].split("Action:")[1].strip()
                tmp_messages[-1]['content'] += f"\nPlease generate actions different from { first_action } and {second_action}. "
                next_actions.append(llm_connection.get_response(tmp_messages, temperature=1))
            except:
                pass

            initial_policy = f"CallWeb(query=\"{query}\", target_url=\"{target_url}\")"

            
            pruned_actions = [prune_actions(a) for a in next_actions]
            def get_unique_indices(B):
                unique_indices = []
                seen = set()

                for index, item in enumerate(B):
                    if item not in seen:
                        seen.add(item)
                        unique_indices.append(index)  # Store the index from A

                return unique_indices
            
            unique_indices = get_unique_indices(pruned_actions)
            next_actions = [next_actions[i] for i in unique_indices]
            print("WM::next_actions", next_actions)
            if len(next_actions) == 1:
                next_action = next_actions[0]
            else:
                if world_model_search == "webdreamer":
                    next_action = world_model_imagination_selection_gpt(next_actions, query, copy.deepcopy(current_messages), llm_connection, world_model_connection, world_model_connection)
                elif world_model_search == "worldmodel":
                    next_action = world_model_imagination_selection_ours(initial_policy, next_actions, query, copy.deepcopy(all_messages), llm_connection, world_model_connection, critic_model_connection, search_depth=search_depth)
        else:
            next_action = llm_connection.get_response(current_messages)
        print(next_action)
        current_messages.append({"role": "assistant", "content": next_action})
        if yield_full_message:
            yield current_messages
        update_or_create_rawdata(
            session_id=session_id,
            message_id=f"{message_id}@@web@@{counter+1}",
            username=username,
            messages_in_train_format=current_messages,
            updated_time=datetime.now().isoformat(),
        )
        print("RESPONSE", next_action)
        all_messages.append({"role": "assistant", "content": next_action})
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, next_action, expanded_part
        )
        print("EXTRACTED ACTION", extracted_action)
        # TODO: "check chunked node and content"
        counter += 1
        if extracted_action["action_name"] is None:
            get_accessibility_tree_succeed = True
            if extracted_thought is not None:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
            else:
                response_error = True
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {action_string}\n"
                )
            yield returned_info
        else:
            if extracted_action["action_name"] == "stop":
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
                if len(downloaded_files) == 0:
                    returned_info = f"[/WEB] {action_string}"
                else:
                    returned_info = f"[/WEB] {action_string}\nDownloaded file paths:\n"
                    for path in downloaded_files:
                        returned_info += f"{path}\n"
                yield returned_info
                break
            
            if extracted_action["action_name"] == "restart":
                extracted_action["action_value"] = target_url
            action_succeed, _ = perform_action(browser_id, page_id, extracted_action)

            if not action_succeed:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
            else:
                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                if 'downloaded the following files:' in current_accessbility_tree:
                    path_str = current_accessbility_tree.split('downloaded the following files:')[1].strip().split('OBJECTIVE:')[0]
                    for path in path_str.split('\n'):
                        downloaded_files.append(path)#.split(':')[1].strip())
                if is_annoying(current_accessbility_tree):
                    skip_this_action = get_skip_action(current_accessbility_tree)
                    perform_action(browser_id, page_id, skip_this_action)
                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                if "Cookie banner" in current_accessbility_tree:
                    cookie_message = (
                    "OBSERVATION:\n" + current_accessbility_tree + "\nOBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                    )
                    popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                    popup_messages.append({"role": "user", "content": cookie_message})
                    next_action = llm_connection.get_response(popup_messages)
                    extracted_action, action_string, extracted_thought = extract_action_for_web(
                        current_accessbility_tree, next_action, expanded_part
                    )
                    print ('Tring to close the pop-up window', next_action)
                    perform_action(browser_id, page_id, extracted_action)

                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                    print ('close the pop-up window attempted')
                current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
                if current_accessbility_tree == prev_obs and prev_action == extracted_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                prev_obs = current_accessbility_tree
                prev_action = extracted_action
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                )
                yield returned_info
    close_browser(browser_id)


def call_web_sample(
    llm_connection,
    query,
    target_url,
    session_id,
    message_id,
    username,
    max_steps=12,
    storage_state=None,
    geo_location=None,
    yield_full_message=False,
    search_from_step=1,
    search_scale=5,
    search_id=0,
    feedback=None,
    critic_connection=None,
):
    """Makes an asynchronous GET request to a target URL with a query parameter.

    Args:
        query (str): The query parameter value to be sent with the request.
        target_url (str): The target URL to which the request is made.

    Returns:
        dict: The JSON response from the target URL, if successful.
    """
    # Use aiohttp.ClientSession for making the HTTP request
    browser_id = get_browser(storage_state, geo_location)
    if browser_id is None:
        yield None

    all_messages = [{"role": "system", "name": "head", "content": system_message}]

    _, page_id = open_page(browser_id, target_url)
    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = get_accessibility_tree(
        browser_id, page_id, 0
    )
    counter = 0
    repeat_count = 0
    prev_obs = current_accessbility_tree
    prev_action = None
    expanded_part = None
    returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] goto {target_url}\n"
    yield returned_info
    action_error = False
    response_error = False
    downloaded_files = []
    while get_accessibility_tree_succeed:

        if counter > max_steps:
            break
        if action_error:
            current_user_message = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
            action_error = False
        elif response_error:
            current_user_message = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            response_error = False
        else:
            if repeat_count >= 2:
                print ('Browsing is getting stuck, stop here')
                break
            current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message, 'step_url': step_url, 'idx2element': idx2element})
        all_messages.append({"role": "user", "content": current_user_message})
        
            
        if counter + 1 > search_from_step:
            next_action = llm_connection.get_response_search_test(current_messages, temperature=0.2, seed=search_id)
            print("In call_web_sample, sampling: ", search_id, next_action)
        else:
            next_action = llm_connection.get_response(current_messages)

        

        ## evaluate the action based on 
        if feedback == "gpt":
            prompt = f"""TASK: {query}
Accessibility Tree: {current_accessbility_tree}

The rationale of the latest action : {next_action.split("Action:")[0]}

The latest action : {next_action.split("Action:")[1]}
"""
            print("evaluator prompt", prompt)
            eval_output = critic_connection.get_response([
                {"role": "system", 
                "content": SYSTEM_PROMPT_STEP_TEXT},
                {"role": "user", 
                "content": prompt}
            ])
            eval_result = parse_eval_output(eval_output)
            print("gpt-4o-eval", eval_result)

            if eval_result['EVALUATION'] == 'POOR':
                # regenerate the action
                msgs = copy.deepcopy(current_messages)
                msgs[-1]['content'] += eval_result['FEEDBACK']
                next_action = llm_connection.get_response_search_test(msgs)
                print("new action after feedback", next_action)

        current_messages.append({"role": "assistant", "content": next_action})

        if yield_full_message:
            yield current_messages
        update_or_create_rawdata(
            session_id=session_id,
            message_id=f"{message_id}@@web@@{counter+1}",
            username=username,
            messages_in_train_format=current_messages,
            updated_time=datetime.now().isoformat(),
        )
        print("RESPONSE", next_action)
        all_messages.append({"role": "assistant", "content": next_action})
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, next_action, expanded_part
        )
        print("EXTRACTED ACTION", extracted_action)
        # TODO: "check chunked node and content"
        counter += 1
        if extracted_action["action_name"] is None:
            get_accessibility_tree_succeed = True
            if extracted_thought is not None:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
            else:
                response_error = True
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {action_string}\n"
                )
            yield returned_info
        else:
            if extracted_action["action_name"] == "stop":
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
                if len(downloaded_files) == 0:
                    returned_info = f"[/WEB] {action_string}"
                else:
                    returned_info = f"[/WEB] {action_string}\nDownloaded file paths:\n"
                    for path in downloaded_files:
                        returned_info += f"{path}\n"
                yield returned_info
                break

            action_succeed, _ = perform_action(browser_id, page_id, extracted_action)

            if not action_succeed:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
            else:
                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                if 'downloaded the following files:' in current_accessbility_tree:
                    path_str = current_accessbility_tree.split('downloaded the following files:')[1].strip().split('OBJECTIVE:')[0]
                    for path in path_str.split('\n'):
                        downloaded_files.append(path)#.split(':')[1].strip())
                if is_annoying(current_accessbility_tree):
                    skip_this_action = get_skip_action(current_accessbility_tree)
                    perform_action(browser_id, page_id, skip_this_action)
                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                if "Cookie banner" in current_accessbility_tree:
                    cookie_message = (
                    "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                    )
                    popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                    popup_messages.append({"role": "user", "content": cookie_message})
                    next_action = llm_connection.get_response(popup_messages)
                    extracted_action, action_string, extracted_thought = extract_action_for_web(
                        current_accessbility_tree, next_action, expanded_part
                    )
                    print ('Tring to close the pop-up window', next_action)
                    perform_action(browser_id, page_id, extracted_action)

                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                    print ('close the pop-up window attempted')
                current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
                if current_accessbility_tree == prev_obs and prev_action == extracted_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                prev_obs = current_accessbility_tree
                prev_action = extracted_action
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                )
                yield returned_info
    close_browser(browser_id)



def call_web_search_test(
    llm_connection,
    query,
    target_url,
    session_id,
    message_id,
    username,
    max_steps=12,
    storage_state=None,
    geo_location=None,
    yield_full_message=False,
    search_from_step=1,
    search_scale=5,
    search_id=0,
):
    """Makes an asynchronous GET request to a target URL with a query parameter.

    Args:
        query (str): The query parameter value to be sent with the request.
        target_url (str): The target URL to which the request is made.

    Returns:
        dict: The JSON response from the target URL, if successful.
    """
    # Use aiohttp.ClientSession for making the HTTP request
    browser_id = get_browser(storage_state, geo_location)
    if browser_id is None:
        yield None

    all_messages = [{"role": "system", "name": "head", "content": system_message}]

    _, page_id = open_page(browser_id, target_url)
    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = get_accessibility_tree(
        browser_id, page_id, 0
    )
    counter = 0
    repeat_count = 0
    prev_obs = current_accessbility_tree
    prev_action = None
    expanded_part = None
    returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] goto {target_url}\n"
    yield returned_info
    action_error = False
    response_error = False
    downloaded_files = []
    while get_accessibility_tree_succeed:

        if counter > max_steps:
            break
        if action_error:
            current_user_message = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
            action_error = False
        elif response_error:
            current_user_message = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            response_error = False
        else:
            if repeat_count >= 2:
                print ('Browsing is getting stuck, stop here')
                break
            current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message, 'step_url': step_url, 'idx2element': idx2element})
        all_messages.append({"role": "user", "content": current_user_message})
        
            
        if counter + 1 == search_from_step:
            initial_thoughts = llm_connection.get_response_search_test(current_messages, additional_stop_token="```", logprobs=-1)
            print("in call_web_search_test initial_thoughts", counter, initial_thoughts)

            response = llm_connection.get_response_search_test(current_messages, initial_thoughts+"```", logprobs = search_scale, additional_stop_token=" [")
            print("in call_web_search_test response", counter, response)

            # select different actions

            logprobs = json.loads(response[0])['choices'][0]['logprobs']

            print("in call_web_search_test logprobs", counter, logprobs)

            this_action = list(logprobs['top_logprobs'][0].items())[search_id][0]

            print("in call_web_search_test this_action", counter, this_action)

            def process_click_sampling(logprobs, this_action):
                if this_action == 'click':
                    # click several more things:
                    return ["".join(['click', ' [', click_item, ']', '```', '<|im_end|>']) for click_item in list(logprobs['top_logprobs'][2].keys())]
            if this_action == "stop":
                action_prompt = initial_thoughts+"```"+this_action
                # print("action_prompt", action_prompt)
                next_action = action_prompt + llm_connection.get_response(current_messages, additional_prompt=action_prompt)
                # print("next_action", next_action)
            else:
                # next_action = initial_thoughts+"```" + process_sampling(logprobs, this_action)[search_id]
                action_prompt = initial_thoughts+"```"+this_action
                response_new = llm_connection.get_response_search_test(current_messages, additional_prompt=action_prompt)
                logprobs = json.loads(response_new[0])['choices'][0]['logprobs']
                click_item = list(logprobs['top_logprobs'][1].keys())[0]
                next_action = action_prompt + ''.join([' [', click_item, ']', '```', '<|im_end|>'])
                print("in call_web_search_test logprobs", logprobs)
                # exit(0)
        else:
            next_action = llm_connection.get_response(current_messages)

        print("searched action", counter, next_action)
        current_messages.append({"role": "assistant", "content": next_action})
        if yield_full_message:
            yield current_messages
        update_or_create_rawdata(
            session_id=session_id,
            message_id=f"{message_id}@@web@@{counter+1}",
            username=username,
            messages_in_train_format=current_messages,
            updated_time=datetime.now().isoformat(),
        )
        print("RESPONSE", next_action)
        all_messages.append({"role": "assistant", "content": next_action})
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, next_action, expanded_part
        )
        print("EXTRACTED ACTION", extracted_action)
        # TODO: "check chunked node and content"
        counter += 1
        if extracted_action["action_name"] is None:
            get_accessibility_tree_succeed = True
            if extracted_thought is not None:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
            else:
                response_error = True
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {action_string}\n"
                )
            yield returned_info
        else:
            if extracted_action["action_name"] == "stop":
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
                if len(downloaded_files) == 0:
                    returned_info = f"[/WEB] {action_string}"
                else:
                    returned_info = f"[/WEB] {action_string}\nDownloaded file paths:\n"
                    for path in downloaded_files:
                        returned_info += f"{path}\n"
                yield returned_info
                break

            action_succeed, _ = perform_action(browser_id, page_id, extracted_action)

            if not action_succeed:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
            else:
                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                if 'downloaded the following files:' in current_accessbility_tree:
                    path_str = current_accessbility_tree.split('downloaded the following files:')[1].strip().split('OBJECTIVE:')[0]
                    for path in path_str.split('\n'):
                        downloaded_files.append(path)#.split(':')[1].strip())
                if is_annoying(current_accessbility_tree):
                    skip_this_action = get_skip_action(current_accessbility_tree)
                    perform_action(browser_id, page_id, skip_this_action)
                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                if "Cookie banner" in current_accessbility_tree:
                    cookie_message = (
                    "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                    )
                    popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                    popup_messages.append({"role": "user", "content": cookie_message})
                    next_action = llm_connection.get_response(popup_messages)
                    extracted_action, action_string, extracted_thought = extract_action_for_web(
                        current_accessbility_tree, next_action, expanded_part
                    )
                    print ('Tring to close the pop-up window', next_action)
                    perform_action(browser_id, page_id, extracted_action)

                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                    print ('close the pop-up window attempted')
                current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
                if current_accessbility_tree == prev_obs and prev_action == extracted_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                prev_obs = current_accessbility_tree
                prev_action = extracted_action
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                )
                yield returned_info
    close_browser(browser_id)


def find_node_with_children(node, target_role, target_name):
    # Check if the current node matches the target role and name
    if node.get('role') == target_role and node.get('name') == target_name:
        return node.get('children', None)
    
    # If the node has children, recursively search through them
    children = node.get('children', [])
    for child in children:
        result = find_node_with_children(child, target_role, target_name)
        if result is not None:
            return result
    
    # If no matching node is found, return None
    return None

def check_if_menu_is_expanded(accessibility_tree, snapshot):
    node_to_expand = {}
    lines = accessibility_tree.split("\n")
    for i, line in enumerate(lines):
        if 'hasPopup: menu' in line and 'expanded: true' in line:
            num_tabs = len(line) - len(line.lstrip("\t"))
            next_tabs = len(lines[i+1]) - len(lines[i+1].lstrip("\t"))
            if next_tabs <= num_tabs:
                #node_to_expand.append(line)
                target_pattern = r"\[(\d+)\] ([a-z]+) '(.*)'"
                matches = re.finditer(target_pattern, line, re.IGNORECASE)
                target_id = None
                target_element_type = None
                target_element_name = None
                for match in matches:
                    target_id, target_element_type, target_element_name = match.groups()
                    break
                if target_element_type is not None:
                    #print ('Menu expand Target element:', target_element_type, target_element_name)
                    children = find_node_with_children(snapshot, target_element_type, target_element_name)
                    #print ('Children:', children)
                    if children is not None:
                        node_to_expand[i] = (num_tabs+1, children, target_id, target_element_type, target_element_name)
    print ('Nodes to expand:', node_to_expand)
    new_lines = []
    curr = 1
    if len(node_to_expand) == 0:
        return accessibility_tree, None
    expanded_part = {}
    for i, line in enumerate(lines):
        if not line.strip().startswith('['):
            new_lines.append(line)
            continue
        #print (line.split('] '))
        num_tabs = len(line) - len(line.lstrip("\t"))
        content = line.split('] ')[1]
        new_lines.append('\t'*num_tabs+f"[{curr}] {content}")
        curr += 1
        if i in node_to_expand:
            for child in node_to_expand[i][1]:
                child_content = f"{child.get('role', '')} '{child.get('name', '')}' " + ' '.join([f"{k}: {v}" for k, v in child.items() if k not in ['role', 'name']])
                tabs = '\t'*node_to_expand[i][0]
                new_lines.append(f"{tabs}[{curr}] {child_content}")
                expanded_part[curr] = (node_to_expand[i][2], node_to_expand[i][3], node_to_expand[i][4])
                curr += 1
    #print ('NEW tree after expanding menu:', '\n'.join(new_lines))
    print ('expanded_part:', expanded_part)
    return '\n'.join(new_lines), expanded_part

def call_webcanvas(
    llm_connection,
    query,
    target_url,
    session_id,
    message_id,
    username,
    max_steps=12,
    storage_state=None,
    geo_location=None,
    yield_full_message=False,
):
    """Makes an asynchronous GET request to a target URL with a query parameter.

    Args:
        query (str): The query parameter value to be sent with the request.
        target_url (str): The target URL to which the request is made.

    Returns:
        dict: The JSON response from the target URL, if successful.
    """
    # Use aiohttp.ClientSession for making the HTTP request
    browser_id = get_browser(None, None)
    if browser_id is None:
        yield None

    evaluate_steps = storage_state
    all_messages = [{"role": "system", "name": "head", "content": system_message}]

    _, page_id = open_page(browser_id, target_url)
    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = get_accessibility_tree(
        browser_id, page_id, 0
    )
    # tree = HTMLTree()
    # client = OpenAI(base_url="https://gptproxy.llmpaas.woa.com/v1", api_key='ZbTr1SazB0NOqLz4KQJTdyiP6xgiPhvw')
    # try:
    #     _, html_content, _ = get_html_content(browser_id, page_id, 0) 
    #     tree.fetch_html_content(html_content)
    #     dom_tree, content_to_id = tree.build_dom_tree()
    #     #print ('DOM TREE:', dom_tree)
    # except:
    #     dom_tree = None
    #     content_to_id = {}
    #     print ('Failed to get the DOM tree')
    counter = 0
    repeat_count = 0
    prev_obs = current_accessbility_tree
    prev_action = None
    expanded_part = None
    returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] goto {target_url}\n"
    yield returned_info
    action_error = False
    response_error = False
    while get_accessibility_tree_succeed:

        if counter > max_steps:
            break
        if action_error:
            current_user_message = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
            action_error = False
        elif response_error:
            current_user_message = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            response_error = False
        else:
            if repeat_count >= 2:
                print ('Browsing is getting stuck, stop here')
                break
                # current_user_message = (
                # "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: Close any pop-up window in the current page, if there is no close button you can accept the pop-up." 
                # )
                # popup_messages = []
                # for tmp_message in all_messages:
                #     if tmp_message["role"] == "user":
                #         popup_messages.append({"role": "user", "content": "<|im_omitted|>"})
                #     else:
                #         popup_messages.append(tmp_message)
                # popup_messages.append({"role": "user", "content": current_user_message})
                # next_action = llm_connection.get_response(popup_messages)
                # extracted_action, action_string, extracted_thought = extract_action_for_web(
                #     current_accessbility_tree, next_action
                # )
                # print ('Tring to close the pop-up window')
                # action(browser_id, page_id, extracted_action)

                # get_accessibility_tree_succeed, current_accessbility_tree, step_url = (
                #     get_accessibility_tree(browser_id, page_id, counter)
                # )
                # print ('close the pop-up window attempted')
            current_user_message = (
                "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: " + query
            )
        current_messages = list()
        for tmp_message in all_messages:
            if tmp_message["role"] == "user":
                current_messages.append({"role": "user", "content": "<|im_omitted|>"})
            else:
                current_messages.append(tmp_message)
        current_messages.append({"role": "user", "content": current_user_message, 'step_url': step_url, 'evaluate_steps': evaluate_steps})
        all_messages.append({"role": "user", "content": current_user_message})
        next_action = llm_connection.get_response(current_messages)
        print(next_action)
        current_messages.append({"role": "assistant", "content": next_action})
        if yield_full_message:
            yield current_messages
        update_or_create_rawdata(
            session_id=session_id,
            message_id=f"{message_id}@@web@@{counter+1}",
            username=username,
            messages_in_train_format=current_messages,
            updated_time=datetime.now().isoformat(),
        )
        print("RESPONSE", next_action)
        all_messages.append({"role": "assistant", "content": next_action})
        extracted_action, action_string, extracted_thought = extract_action_for_web(
            current_accessbility_tree, next_action, expanded_part
        )
        print("EXTRACTED ACTION", extracted_action)
        # if extracted_action["target_element_name"] in content_to_id:
        #     mapped_id = content_to_id[extracted_action["target_element_name"]]
        # else:
        #     messages = [{'role': 'system', 'content': 'You are given an web page DOM element tree and a target accessibility element the user want to interact with, select one DOM element from the tree that best matches the target accessibility element. You should only give out the chose DOM element id and DO NOT SAY ANYTHING ELSE.'}, {'role': 'user', 'content': f"DOM element tree:\n{dom_tree}\nTarget accessibility element: {extracted_action['target_element_type']}: {extracted_action['target_element_name']}"}]
        #     try:
        #         openai_response = client.chat.completions.create(
        #             model='gpt-4', messages=messages, max_tokens=10, seed=42, temperature=0
        #         )
        #         mapped_id = openai_response.choices[0].message.content
        #         mapped_id = int(mapped_id)
        #     except:
        #         mapped_id = None
        #         #print ('GPT failed to find a mapped_id')
        #     #print ('GPT4 chosen mapped ID:', mapped_id)
        # if mapped_id is not None:
        #     path = tree.get_selector_and_xpath(tree.nodeDict[mapped_id])
        #     if extracted_action['action_name'] == 'click':
        #         element_value = tree.get_element_value(tree.nodeDict[mapped_id])
        #     elif extracted_action['action_name'] == 'type':
        #         element_value = extracted_action['action_value']
        #     else:
        #         element_value = None
        # else:
        #     path = None
        #     element_value = None
        
        #print ('MAPPED ID:', mapped_id, path, element_value)
        # evaluate_steps, match_result = step_evaluate(url=step_url, page_content=html_content, evaluate_steps=evaluate_steps, input_path=path, element_value=element_value)
        # total_step_score = sum([step['score'] for step in evaluate_steps])
        # print ('Total step score:', total_step_score, 'out of', len(evaluate_steps))
        # TODO: "check chunked node and content"
        counter += 1
        if extracted_action["action_name"] is None:
            get_accessibility_tree_succeed = True
            if extracted_thought is not None:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
            else:
                response_error = True
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {action_string}\n"
                )
            yield returned_info
        else:
            if extracted_action["action_name"] == "stop":
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
                returned_info = f"[/WEB] {action_string}"
                yield returned_info
                break

            action_succeed, _ = perform_action(browser_id, page_id, extracted_action)

            if not action_succeed:
                action_error = True
                returned_info = f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                yield returned_info
            else:

                get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                    get_accessibility_tree(browser_id, page_id, counter)
                )
                if is_annoying(current_accessbility_tree):
                    skip_this_action = get_skip_action(current_accessbility_tree)
                    perform_action(browser_id, page_id, skip_this_action)
                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                if "Cookie banner" in current_accessbility_tree:
                    cookie_message = (
                    "OBSERVATION:\n" + current_accessbility_tree + "OBJECTIVE: There is a cookie banner on the page, please accept the cookie banner." 
                    )
                    popup_messages = [{"role": "system", "name": "head", "content": system_message}]
                    popup_messages.append({"role": "user", "content": cookie_message})
                    next_action = llm_connection.get_response(popup_messages)
                    extracted_action, action_string, extracted_thought = extract_action_for_web(
                        current_accessbility_tree, next_action, expanded_part
                    )
                    print ('Tring to close the pop-up window', next_action)
                    perform_action(browser_id, page_id, extracted_action)

                    get_accessibility_tree_succeed, current_accessbility_tree, step_url, snapshot, idx2element = (
                        get_accessibility_tree(browser_id, page_id, counter)
                    )
                    print ('close the pop-up window attempted')
                # try:
                #     _, html_content, _ = get_html_content(browser_id, page_id, counter) 
                #     tree.fetch_html_content(html_content)
                #     dom_tree, content_to_id = tree.build_dom_tree()
                # except:
                #     dom_tree = None
                #     content_to_id = {}
                #     print ('Failed to get the DOM tree')
                #print (len(snapshot), snapshot)
                current_accessbility_tree, expanded_part = check_if_menu_is_expanded(current_accessbility_tree, snapshot)
                #print ('DOM TREE:', dom_tree)
                if current_accessbility_tree == prev_obs and prev_action == extracted_action:
                    repeat_count += 1
                else:
                    repeat_count = 0
                prev_obs = current_accessbility_tree
                prev_action = extracted_action
                returned_info = (
                    f"[WEB] [{browser_id}] [{page_id}] [{counter}] {extracted_thought}\n"
                )
                yield returned_info
    close_browser(browser_id)
