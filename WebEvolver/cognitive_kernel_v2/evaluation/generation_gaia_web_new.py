import time
import requests
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
from openai import OpenAIError, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

# import sys
# sys.path.append("../backend")
# sys.path.append("../evaluation")

from utils import read_jsonl, save_jsonl, load_jsonl_folder
from ck_utils import GAIA_system_prompt_new, extract_string, start_docker, stop_docker

# Searvice URL (Remember to change this if needed)
url = "http://localhost:8080/api/inference_api_gaia"

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

def compare_results(ground, generation):
    messages = [
        {'role': 'system', 'content': "You are given a ground answer and a generated answer. Please determine whether the generated answer is correct based on the provided ground answer. Please enter 1 for yes and 0 for no. The output format is Score: [0 or 1]."},
        {
            'role': 'user',
            'content': f"Ground answer: {ground}\nGenerated answer: {generation}"
        }
    ]
    jailbreak = False
    while True:
        try:
            print('Calling gpt4o API to get the auto evaluation......')
            openai_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=1000, seed=42, temperature=0
            )
            break
        except Exception as e:
            print(e)
            try:
                if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                    jailbreak = True
                    break
            except:
                pass
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    if jailbreak:
        return 0
    gpt_4v_res = openai_response.choices[0].message.content
    return gpt_4v_res

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the LLM on the GAIA dataset.")
    parser.add_argument("--query_path", type=str, default="../../data/GAIA_web/GAIA_web.jsonl")
    parser.add_argument("--output_path", type=str, default="../GAIA_web/llama-3.3-rej-sampling")
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
    results = load_jsonl_folder(save_dir)

    for i in range(len(input_query)):
        if i < len(results):
            continue
        question = input_query[i]['ques']
        if 'webwalker' in args.query_path:
            web = input_query[i]['root_url']
        else:
            web = "www.bing.com" # force using the search engine to be bing.com

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

    client = AzureOpenAI(
                azure_endpoint = args.azure_endpoint, 
                api_key=args.api_key,  
                api_version=args.api_version
            )

    def work_func_new(item):
        i, item = item
        # outer_loop_history, inner_loop_history, histories = parse_history(item)
        try:
            outer_loop_history, _ = json.loads(item)
        except:
            return 0

        if 'Final answer' in input_query[i]:
            ground_answer = input_query[i]['Final answer']
        elif 'answer' in input_query[i]:
            ground_answer = input_query[i]['answer']
        try:
            return compare_results(ground_answer, outer_loop_history[-1]['content'])
        except:
            return 0

    res_list = []
    with ThreadPoolExecutor(5) as executor:
        for res in tqdm(executor.map(work_func_new, enumerate(results)), total=len(results)):
            res_list.append(res)
    res_list = [item if item == 0 else int(item.split('Score:')[1].strip()) for item in res_list]

    import numpy as np

    print(f"Accuracy: {np.mean(res_list) * 100:.2f}")
    if 'gaia' in args.query_path.lower():
        print(f"Level 1: {np.mean(res_list[:26]) * 100:.2f}, Level 2: {np.mean(res_list[26:]) * 100:.2f}")
    elif 'webwalker' in args.query_path.lower():
        en_index = [item['info']['lang'] == 'en' for item in input_query]
        print(f"english acc: {np.mean([res for e, res in zip(en_index, res_list) if e]) * 100:.2f}")