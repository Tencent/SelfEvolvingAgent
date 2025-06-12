import re
import time
import pandas as pd
import numpy as np
import requests
import json
import os
from copy import deepcopy
import random
from itertools import chain
from tqdm import tqdm
import argparse

import sys
sys.path.append("backend")
sys.path.append("evaluation")

from cognitive_kernel.base_model_connection import BaseModelConnection
from ck_utils import test_system_prompt, extract_string, start_docker, stop_docker, extract_string
from eval_webvoyager import parse_history
from evaluator.simpleqa_evaluator import evaluate_simple_qa, ChatGPTConnection, get_final_results
from utils import read_jsonl, save_jsonl, gather_multiple_runs_jsonl, load_jsonl_folder

os.environ['LLAMA_VERSION'] = '3.3'
os.environ['INFERENCE_SERVER_ENGINE'] = 'vLLM'


world_model_rationale_prompt = """Please generate rationale for why the webpage is generated after performing the given action from the original web page.

An example of the overall objective of the web task:

Search for "JQuery selector for elements with specific class" and summarize it.

An example of the previous webpage accessibility tree:

Tab 0 (current): Google\n\n[1] RootWebArea 'Google' focused: true\n\t[2] link 'Gmail '\n\t[3] link '搜尋圖片 '\n\t[4] button 'Google 應用程式' expanded: false\n\t[5] link '登入'\n\t[6] image '2024 年佳節'\n\t[7] combobox '搜尋' focused: true autocomplete: both hasPopup: listbox required: false expanded: false\n\t[8] button '分享'

An example of action:

type [7] [JQuery selector for elements with specific class] [1]

which indicates typing "JQuery selector for elements with specific class" into the field with id 7, corresponding to the combobox (search box) on the Google homepage.

The definitions of the actions are as follows: The actions you can perform are the following:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.
`scroll [direction=down|up]`: Scroll the page up or down.
`goback`: Navigate to the previously viewed page.
`restart`: Navigate to the Google search homepage. When you can't find information in some websites, try starting over from Google search.

An example of the next webpage accessibility tree:

'OBSERVATION:\nTab 0 (current): Google\n\n[1] RootWebArea \'JQuery selector for elements with specific class - Google 搜尋\' focused: true\n\t[2] heading \'無障礙功能連結\'\n\t[3] link \'跳至主內容\'\n\t[4] link \'無障礙功能說明\'\n\t[5] link \'無障礙功能意見\'\n\t[6] link \'2024 年佳節\'\n\t[7] combobox \'搜尋\' autocomplete: both hasPopup: listbox required: false expanded: false\n\t\t[8] StaticText \'JQuery selector for elements with specific class\'\n\t[9] button \' 清除\'\n\t[10] button \'語音搜尋\'\n\t[11] button \'以圖搜尋\'\n\t[12] button \'搜尋 \'\n\t[13] button \'設定\'\n\t[14] button \'Google 應用程式\' expanded: false\n\t[15] link \'登入\'\n\t[16] navigation \'篩選器和主題\'\n\t\t[17] link \'全部\' disabled: true\n\t\t[18] link \'圖片\'\n\t\t[19] link \'購物\'\n\t\t[20] link \'影片\'\n\t\t[21] link \'新聞\'\n\t\t[22] link \'書籍\'\n\t\t[23] link \'網頁\'\n\t\t[24] button \'更多\' hasPopup: menu expanded: false\n\t\t[25] button \'工具\' expanded: false\n\t[26] StaticText \'Looking for results in English?\'\n\t[27] link \'Change to English\'\n\t[28] link \'繼續使用 繁體中文\'\n\t[29] link \'語言設定\'\n\t[30] heading \'搜尋結果\'\n\t[31] heading \'網上的精選簡介\'\n\t[32] StaticText \'In jQuery, the class and ID selectors are the same as in CSS. If you want to select elements with a certain class, \'\n\t[33] StaticText \'use a dot ( . )\'\n\t[34] StaticText \'and the class name\'\n\t[35] StaticText \'. If you want to select elements with a certain ID, use the hash symbol ( # ) and the ID name.\'\n\t[36] StaticText \'2020年1月28日\'\n\t[37] link \' jQuery Selectors Explained: Class Selectors, ID Selectors, and ... freeCodeCamp https://www.freecodecamp.org › news › jquery-selectors\'\n\t[38] button \'About this result\'\n\t[39] link \'關於精選訊息摘錄\'\n\t[40] button \'意見反映\'\n\t[41] link \' jquery find element by specific class when element has ... Stack Overflow https://stackoverflow.com › questions\'\n\t[42] StaticText \'·\'\n\t[43] link \'翻譯這個網頁\'\n\t[44] button \'關於此結果\'\n\t[45] StaticText \'2011年9月28日\'\n\t[46] StaticText \' — \'\n\t[47] StaticText \'You can \'\n\t[48] StaticText \'select elements\'\n\t[49] StaticText \' with multiple \'\n\t[50] StaticText \'classes\'\n\t[51] StaticText \' like so: $("\'\n\t[52] StaticText \'.firstClass.anotherClass"); Simply chain the next \'\n\t[53] StaticText \' onto the first one, without a space.\'\n\t[54] link \'5 個答案\'\n\t[55] StaticText \'·\'\n\t[56] StaticText \'最佳解答:\xa0\'\n\t[57] StaticText \'You can combine selectors like this $(".alert-box.warn, .alert-box.dead"); Or if you ...\'\n\t[58] link \'Jquery element+class selector performance - Stack Overflow\'\n\t[59] StaticText \'2012年7月28日\'\n\t[60] link \'jquery selector for div with class - javascript - Stack Overflow\'\n\t[61] StaticText \'2012年5月11日\'\n\t[62] link \'Jquery Selector for Element type and Class name?\'\n\t[63] StaticText \'2011年8月24日\'\n\t[64] link \'jQuery: Is it possible to select elements with only one class ...\'\n\t[65] StaticText \'2011年12月14日\'\n\t[66] link \'stackoverflow.com 的其他相關資訊\'\n\t[67] StaticText \'jQuery API Documentation\'\n\t[68] StaticText \'https://api.jquery.com\'\n\t[69] StaticText \' › class-selector\'\n\t[70] StaticText \'·\'\n\t[71] link \'翻譯這個網頁\'\n\t[72] button \'關於此結果\'\nOBJECTIVE: JQuery selector for elements with specific class'

An example of the rationale:

The action performed was to type "JQuery selector for elements with specific class" into the Google search box. The expected result is that the Google search page will display relevant information about JQuery selectors for elements with specific classes. The observation indicates that the search results are displayed, and there are links to relevant articles, such as "jQuery Selectors Explained: Class Selectors, ID Selectors, and ..." and "jquery find element by specific class when element has ...". The text on the page also provides information about JQuery selectors, including the use of a dot (.) and the class name to select elements with a certain class.
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

def parse_planning_result(llm_generation):
    parts = llm_generation.split("```")
    planning_code = parts[1] if len(parts) > 1 else ""
    planning_code = planning_code.replace("python", "")
    
    # pattern = r'target_url="([^,]+)\)"'
    pattern = r'target_url=["\']([^"]+)["\']'
    # Use re.search to find the match
    match = re.search(pattern, llm_generation)
    # # Check if a match is found and extract the value
    if match:
        target_url = match.group(1)
    else:
        target_url = ""

    pattern = r'query=["\']([^"]+)["\'],'
    # Use re.search to find the match
    match = re.search(pattern, llm_generation)
    # # Check if a match is found and extract the value
    if match:
        query = match.group(1)
    else:
        query = ""

    return planning_code, query, target_url

def get_rationale_world_model(objective, current_accessibility_tree, action_performed, next_accessibilit_tree):
    return vllm_connection.get_response([{"role":"system", "content": world_model_rationale_prompt}, 
                              {"role": "user", "content": f"""
The objective of the web task:
{objective}
Previous webpage accessibility tree:
{current_accessibility_tree}
Action performed:
{action_performed}
Next webpage accessibility tree after performing the action:
{next_accessibilit_tree}
Explain the rationale that the next webpage accessibility tree is expected to be after performing the action.
"""}])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Postprocess trajectory data")
    parser.add_argument(
        "--history_action_observation_path", type=str, nargs='+', help="List of files to the trajectories"
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_service", type=str, default="IP and port of the vllm service for the LLM. xxx.xxx.xxx.xxx:yyyy")
    args = parser.parse_args()


    ## convert trajectory data to history_action_observations pairs.

    history_action_observations = read_jsonl(args.history_action_observation_path)

    history_action_observations_single = []
    for item in history_action_observations:
        history_action_observations_single.append( [item[1], item[2]] )
        for i in range(3, len(item)-1, 2):
            history_action_observations_single.append( [item[i-1], item[i], item[i+1]] )


    ##

    vllm_connection = BaseModelConnection(args.model_service)

    # filter
    first_step_dict = {}
    for i, item in enumerate(history_action_observations_single):
        if 'CallWeb' in item[1]['content']:
            _, query, target_url = parse_planning_result(item[1]['content'])
            if not target_url in first_step_dict:
                first_step_dict[target_url] = [i]
            else:
                first_step_dict[target_url].append(i)

    action_dict = {}
    for i, item in enumerate(history_action_observations_single):
        if not 'CallWeb' in item[1]['content']:
            action = ss(item[2]['content'])
            action_first_two = " ".join(action.split()[:2])
            if not action_first_two in action_dict:
                action_dict[action_first_two] = [i]
            else:
                action_dict[action_first_two].append(i)
                

    

    selected_idx = list(set(chain(*[np.random.choice(val, 20) for key, val in first_step_dict.items()]))) + \
            list(set(chain(*[np.random.choice(val, 50) for key, val in action_dict.items()])))

    history_action_observations_single_filter_first_step = []
    for i, item in enumerate(history_action_observations_single):
        if i in selected_idx:
            history_action_observations_single_filter_first_step.append(item)
        
    ### Add chain-of-thought of deriving the next action.
    ### Generating Rationale for next webpage generation task.

    history_action_observations_single_filter_first_step_rationale = deepcopy(history_action_observations_single_filter_first_step)
    for item in tqdm(history_action_observations_single_filter_first_step_rationale):
        if "```python\nCallWeb" in item[1]['content']:
            # this is the inital action
            objective = item[1]['content'].split('OBJECTIVE:')[1]
            _, query, target_url = parse_planning_result(item[1]['content'])
            rationale = f"The goal is to first open the web page {target_url}, which displays the homepage of it."
            item[-1]['content'] = f"THOUGHT: {rationale}\n" + item[-1]['content']
            
        else:
            objective = item[2]['content'].split('OBJECTIVE:')[1]
            action = ss(item[2]['content'])
            current_accessibility_tree = item[1]['content']
            next_accessibilit_tree = item[3]['content']
            rationale = get_rationale_world_model(objective, current_accessibility_tree, action, next_accessibilit_tree)
            item[-1]['content'] = f"THOUGHT: {rationale}\n" + item[-1]['content']
        print("RATIONALE", rationale)

    save_jsonl(history_action_observations_single_filter_first_step_rationale, args.output_path)
