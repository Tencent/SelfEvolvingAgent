# Using world model to synthesize trajectories


import json
import os
import sys
from itertools import chain
import argparse
sys.path.append("backend")
sys.path.append("evaluation")

from cognitive_kernel.base_model_connection import BaseModelConnection
from utils import read_jsonl, save_jsonl, gather_multiple_runs_jsonl, load_jsonl_folder
from ck_utils import test_system_prompt
from copy import deepcopy

os.environ['INFERENCE_SERVER_ENGINE'] = 'vLLM'

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


web_system_message = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

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

def synthetic_call_web(query):
    tmp_messages = [
        {'role':'system', 'name':'head', 'content': test_system_prompt}, 
        {'role':'user', 'content': query}
    ]

    initial_policy = policy_model_connection.get_response(tmp_messages, temperature=0.7, do_print=False)

    initial_state_message = [
        {"role": "system", "content": world_model_system_prompt}, 
        {"role": "user", "content": f"Initial operation: {initial_policy}\n\nOBJECTIVE: {query} "},
    ]

    initial_state = world_model_connection.get_response(initial_state_message, do_print=False)

    current_accessbility_tree = initial_state


    all_messages = [{"role": "system", "name": "head", "content": web_system_message}]

    depth = 0
    while depth < 7:

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
        next_action = policy_model_connection.get_response(current_messages, temperature=0.7, do_print=False)
        if depth == 6:
            print(next_action)
        current_messages.append({"role": "assistant", "content": next_action})
        all_messages.append({"role": "assistant", "content": next_action})


        depth += 1
        # print(depth)

        try:
            next_action = prune_actions(next_action)
        except:
            # stop?
            print("action parsing error", next_action)
            break

        if 'stop' in next_action or depth > 6:
            break
        

        world_model_message = [{'role': 'system',
                    'content': world_model_system_prompt},
                    {"role": "user", "content": f"Initial operation: {initial_policy}\n\nOBJECTIVE: {query} "}]\
                            + deepcopy(all_messages[1:])
        # swap the roles of user and assistant

        for j in range(1, len(world_model_message), 2, ):
            world_model_message[j]['role'] = 'user'
            if j >= 3:
                world_model_message[j]['content'] = prune_actions(world_model_message[j]['content']) + "\nOBJECTIVE: " + query
            if j - 1 > 0:
                world_model_message[j - 1]['role'] = 'assistant'

        for j in range(0, len(all_messages[1:]) - 2, 2):
            world_model_message[j + 2]['role'] = 'user'
            world_model_message[j + 2]['content'] = 'The previous observations are omitted.'
        
        if depth == 6:

            world_model_message[-1]['content'] += "\n\n Please generate a next webpage containining the information that is directly needed for answering the query."
            print("depth 6", world_model_message)
        
        next_accessibility_tree = world_model_connection.get_response(world_model_message, temperature=0.7, do_print=False)
        
        if 'OBSERVATION:' in next_accessibility_tree:
            current_accessbility_tree = next_accessibility_tree.split('OBSERVATION:')[1].strip()
        elif '[1] RootWebArea' in next_accessibility_tree:
            current_accessbility_tree = '[1] RootWebArea' + next_accessibility_tree.split('[1] RootWebArea')[1].strip()
        else:
            print("error", next_accessibility_tree)
            break

    return all_messages


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Postprocess trajectory data")
    parser.add_argument(
        "--input_query", type=str, nargs='+', help="List of files to the trajectories"
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--world_model_service", type=str, default="IP and port of the vllm service for world model. xxx.xxx.xxx.xxx:yyyy")
    parser.add_argument("--policy_model_service", type=str, default="IP and port of the vllm service for the agent policy model. xxx.xxx.xxx.xxx:yyyy")
    args = parser.parse_args()


    queries = list(chain(*[json.load(open(file_path, "r")) for file_path in args.input_query]))


    policy_model_connection = BaseModelConnection(args.world_model_service) 
    world_model_connection = BaseModelConnection(args.policy_model_service)


    synthetic_queries_m2w = []

    from tqdm import tqdm
    for query in tqdm(queries):
        synthetic_queries_m2w.append(synthetic_call_web(query))

    save_jsonl(synthetic_queries_m2w, args.output_path)
