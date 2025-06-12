import time
import pandas as pd
import requests
import json
import os
import argparse
from itertools import chain


# import sys
# sys.path.append("../backend")
# sys.path.append("../evaluation")

from evaluation import utils, ck_utils
from evaluation.utils import read_jsonl, save_jsonl
from evaluation.ck_utils import test_system_prompt


# Searvice URL (Remember to change this if needed)
url = "http://0.0.0.0:8080/api/inference_api"
# url = "http://localhost:8080/api/inference_api" # if on Windows

def get_ck_response(query):
    tmp_messages = [{'role':'system', 'name':'head', 'content': test_system_prompt}, {'role':'user', 'content': query}]

    # response = requests.post(url, data=json.dumps({"messages": tmp_messages, "full_info": True}), headers={"Content-Type": "application/json"})

    try:
        response = response = requests.post(url, data=json.dumps({"messages": tmp_messages, "full_info": True}), headers={"Content-Type": "application/json"})
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', nargs='+', help='Path to the query file')
    parser.add_argument('--output_path', type=str, default="evaluation/openwebvoyager/llama3.3", help='Path to the output file')
    args = parser.parse_args()

    openwebvoyager_queries = list(chain(*[json.load(open(path)) for path in args.query_path]))

    openwebvoyager_search_results = []
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(openwebvoyager_queries)):

        query = openwebvoyager_queries[i]

        # print(query)

        result = get_ck_response(query)
        if result is None:
            time.sleep(10)
            result = get_ck_response(query)
            # retry for once
        save_jsonl([result], save_dir + "/" + str(i) + ".jsonl")
        openwebvoyager_search_results.append(result)
        time.sleep(3)
        # if i % 10 == 0:
        #     stop_docker()
        #     time.sleep(20)