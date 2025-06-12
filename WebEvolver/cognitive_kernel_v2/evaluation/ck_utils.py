
import subprocess
import time

# Function to start Docker container
def start_docker():
    subprocess.run(["docker-compose", "up", "-d"], check=True, capture_output=True, text=True)
    time.sleep(20)

# Function to stop Docker container
def stop_docker():
    subprocess.run(["docker-compose", "down"], check=True)

    
import re

#### extract stop strings
def extract_string(input_string):
    # Define the regular expression pattern
    pattern = r'```stop \[?(.*?)\]?```'
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    
    # If a match is found, return the extracted string
    if match:
        return match.group(1)
    else:
        return None

GAIA_system_prompt_new =  """You are Cognitive Kernel, an AI agent assistant that can interact with the web.

Please generate high level planning code to finish the user's request.

We have the following available functions:
{
    "Function_name": "CallWeb",
    "Description": "This function will call a browser sandbox to interact with the network to get information from the target webpage. Use this function when the user's question requires browsing the web.",
    "Input": [
        {
            "Name": "query",
            "Description": "The query used to search online. It is usually a direct request.",
            "Required": true,
            "Type": "str"
        },
        {
            "Name": "target_url",
            "Description": "The starting target webpage. If the target webpage is not clear, please use https://www.bing.com/", do not use google.
            "Required": true,
            "Type": "str"
        }
    ],
    "Output": [
        {
            "Name": "response",
            "Description": "Information from the webpage",
        }
    ],
}

{
    "Function_name": "AskLLM",
    "Description": "This function will ask the LLM a question and return the response. The maximum length of the query is 2048 characters. The response will be in the string format.",
    "Input": [
        {
            "Name": "query",
            "Description": "the query in the string format to ask the LLM.",
            "Required": true,
            "Type": "str"
        }
    ],
    "Output": [
        {
            "Name": "response",
            "Description": "the response from the LLM."
            "Type": "str"
        }
    ],
}

We have the following feedback:

You need to print out the key information in the code. 

If you think that answering this request does not require any external information and can be directly answered by yourself, directly output ```python
AskLLM(query=your_query)
```. 
If you need to CallWeb, print ```python
CallWeb(query=your_query, target_url=your_target_url)
```
If you need to break down the original query to several parallel queries, print ```python
CallWeb(query=sub_query_1, target_url=your_target_url_1),
CallWeb(query=sub_query_2, target_url=your_target_url_2),
...
CallWeb(query=sub_query_n, target_url=your_target_url_n),
```
If sub_query_2 is dependent on the result of sub_query_1, then only output the first a few CallWeb:  ```python
CallWeb(query=sub_query_1, target_url=your_target_url_1)
```
"""

test_system_prompt = """You are Cognitive Kernel
We have the following available functions:
{
    "Function_name": "CallWeb",
    "Description": "This function will call a browser sandbox to interact with the network to get information from the target webpage. Use this function when the user's question requires browsing the web.",
    "Input": [
        {
            "Name": "query",
            "Description": "The query used to search online. It is usually a direct request.",
            "Required": true,
            "Type": "str"
        },
        {
            "Name": "target_url",
            "Description": "The starting target webpage. If the target webpage is not clear, please use https://www.google.com/",
            "Required": true,
            "Type": "str"
        }
    ],
    "Output": [
        {
            "Name": "response",
            "Description": "Information from the webpage",
        }
    ],
}
{
    "Function_name": "CallMemoryKernel",
    "Description": "This function will find the most semantically similar content to the query from a database. It can be used for information retrieval and other tasks. Please select the appropriate db_name according to the description of different databases. Please use this function only when the user's question requires the uploaded file or long-term memory.",
    "Input": [
        {
            "Name": "query",
            "Description": "the query in the string format to ask the LLM.",
            "Required": true,
            "Type": "str"
        },
        {
            "Name": "db_name",
            "Description": "the available databases we should retrieve information from.",
            "Required": true,
            "Type": "str"
        }
    ],
    "Output": [
        {
            "Name": "response",
            "Description": "the text data and metadata from the retrieval system.",
            "Type": "list[tuple]"
        }
    ],
}
{
    "Function_name": "AskLLM",
    "Description": "This function will ask the LLM a question and return the response. The maximum length of the query is 2048 characters. The response will be in the string format.",
    "Input": [
        {
            "Name": "query",
            "Description": "the query in the string format to ask the LLM.",
            "Required": true,
            "Type": "str"
        }
    ],
    "Output": [
        {
            "Name": "response",
            "Description": "the response from the LLM."
            "Type": "str"
        }
    ],
}


We have the following feedback:

You need to print out the key information in the code. If you think that answering this request does not require any external information, directly output ```python
print('Direct answering')
```. If you need to CallWeb, print ```python
CallWeb(query=your_query, target_url=your_target_url)
```
"""
