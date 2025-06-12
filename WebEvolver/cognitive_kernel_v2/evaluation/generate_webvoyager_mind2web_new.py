import time
import requests
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
from openai import OpenAIError, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append("../backend")
sys.path.append("../evaluation")

from utils import read_jsonl, save_jsonl, load_jsonl_folder
from ck_utils import GAIA_system_prompt_new, extract_string, start_docker, stop_docker

# Searvice URL (Remember to change this if needed)
url = "http://0.0.0.0:8080/api/inference_api"

def get_ck_response(query, world_model_type=None, world_model_search_depth=1):
    tmp_messages = [{'role':'system', 'name':'head', 'content': GAIA_system_prompt_new}, {'role':'user', 'content': query}]

    # response = requests.post(url, data=json.dumps({"messages": tmp_messages, "full_info": True}), headers={"Content-Type": "application/json"})

    try:
        response = requests.post(url, data=json.dumps({"messages": tmp_messages, "full_info": True, "world_model_type": world_model_type, "world_model_search_depth": world_model_search_depth}), headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.text
        else:
            print("error code", response.status_code)
            return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error: {query}")
        return None
    except Exception as e:
        print("other error", e)
        return None
    
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
    return outer_loop_history, inner_loop_history, histories


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
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content
    print_message = messages[1]

    # print_message[1]['content'][1]['image_url'] = {"url": "data:image/png;base64, b64_img"}
    print(print_message)
    print(gpt_4v_res)

    auto_eval_res = 0 if 'NOT SUCCESS' in gpt_4v_res else 1
    if 'SUCCESS' not in gpt_4v_res:
        auto_eval_res = None
    print('Auto_eval_res:', auto_eval_res)
    print()
    return auto_eval_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the LLM on the GAIA dataset.")
    parser.add_argument("--query_path", type=str, default="../../data/webvoyager/WebVoyager_data.jsonl")
    parser.add_argument("--output_path", type=str, default="./output_traj/webvoyager/gpt-4o")
    parser.add_argument("--azure_endpoint", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_version", type=str, default="2024-02-01")
    parser.add_argument("--world_model_type", type=str, default=None)
    parser.add_argument("--world_model_search_depth", type=int, default=1)

    start_docker()
    args = parser.parse_args()
    
    if args.query_path.endswith(".jsonl"):
        input_query = read_jsonl(args.query_path)

    elif args.query_path.endswith(".csv"):
        input_query = pd.read_csv(args.query_path, encoding="latin-1").head(100)
        input_query = input_query.to_dict(orient="records")
        input_query = [
            {('ques' if k == 'problem' else k): v for k, v in row.items()}
            for row in input_query
        ]


    
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)
    # results = load_jsonl_folder(save_dir)
    # print(len(results))
    results = []
    import glob
    existing_ids = glob.glob(save_dir + "/*.jsonl")
    ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in existing_ids]

    for i in range(len(input_query)):
        if i in ids:
            continue
        question = input_query[i]['ques']
        if 'webvoyager' in args.query_path.lower():
            web = input_query[i]['web']
            if not input_query[i]['web_name'] in ['Apple', 'ArXiv', 'BBC News', 'Coursera', 'ESPN', 'GitHub', 'Google Map', 'Huggingface', 'Wolfram Alpha']:
                continue


        query = "Now given a task: " + question + " Please interact with "  + web

        print(query)

        result = get_ck_response(query, args.world_model_type, args.world_model_search_depth)
        if result is None:
            time.sleep(10)
            result = get_ck_response(query, args.world_model_type, args.world_model_search_depth)
            # retry for once
        save_jsonl([result], save_dir + "/" + str(i) + ".jsonl")
        results.append(result)
        time.sleep(3)
        if i % 10 == 0:
            stop_docker()
            start_docker()
            time.sleep(20)

    ### Evaluation