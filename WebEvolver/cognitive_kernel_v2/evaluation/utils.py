import json
import os
def read_jsonl(file_path):
    """
    Reads and parses all lines of a .jsonl file.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        list: A list of parsed JSON objects.
    """
    json_objects = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Ensure the line is not empty
                    json_objects.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return json_objects


def save_jsonl(data, file_name):
    """
    Save a list of dictionaries to a .jsonl file.

    Parameters:
    data (list): A list of dictionaries to be saved.
    file_name (str): The name of the .jsonl file to save the data to.
    """
    with open(file_name, 'w') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')


def load_jsonl_folder(folder_path, dictionary=False):
    jsonl_data = []
    
    # Get a sorted list of all .jsonl files in the folder
    jsonl_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jsonl')], key=lambda x:int(x.split('.jsonl')[0]))
    
    for jsonl_file in jsonl_files:
        jsonl_file_path = os.path.join(folder_path, jsonl_file)
        
        # Open and read the .jsonl file
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if dictionary:
                        jsonl_data.append((int(jsonl_file.split('.jsonl')[0]), data))
                    else:
                        jsonl_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {jsonl_file_path}: {e}")
    
    return jsonl_data


#### this function aggregates the results of multiple runs

def gather_multiple_runs_jsonl(folders):
    # assert len(folders) > 1
    final_results = dict(load_jsonl_folder(folders[0], dictionary=True))
    test_eval_results = json.load(open(os.path.join(folders[0], 'eval_results.json')))

    for folder in folders[1:]:
        jsonl_data = dict(load_jsonl_folder(folder, dictionary=True))
        eval_results = json.load(open(os.path.join(folder, 'eval_results.json')))
        keys = sorted(list(jsonl_data.keys()))
        for i in range(len(keys)):
            if eval_results[i] == 1:
                final_results[keys[i]] = jsonl_data[keys[i]]
                test_eval_results[keys[i]] = 1
    return final_results, test_eval_results

#####
'''
final_results, test_eval_results = gather_multiple_runs_jsonl(['./evaluation/openwebvoyager/llama3.3/search',
'./evaluation/openwebvoyager/llama3.3/search_2'])
final_results, test_eval_results = gather_multiple_runs_jsonl(['./evaluation/openwebvoyager/llama3.3/webvoyager_human',
'./evaluation/openwebvoyager/llama3.3/webvoyager_human_2'])
'''