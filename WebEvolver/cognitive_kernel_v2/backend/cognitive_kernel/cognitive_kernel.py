import json
import os
import io
import sys
import copy
import time
import asyncio
import nest_asyncio
import requests
import PyPDF2
import fitz
import docx2txt
import re
import uuid
import shutil
from datetime import datetime
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from cognitive_kernel.base_model_connection import (
    BaseModelConnection,
    ChatGPTConnection,
)
from functools import partial
from cognitive_kernel.call_web import call_web, call_webcanvas, call_web_search_test, call_web_sample
from concurrent.futures import ThreadPoolExecutor
from cognitive_kernel.code_executor import ExecutorManager
from database import (
    update_or_create_rawdata,
    get_annotations_for_evaluation,
    get_rawdata_by_session_id,
)
import copy
from evaluator.openai_evaluator import SYSTEM_PROMPT_STEP_TEXT, SYSTEM_PROMPT_START_TEXT, USER_PROMPT_STEP_TEXT, parse_eval_output

GAIA_follow_up_prompt = """
You are Cognitive Kernel, an AI agent assistant that can interact with the web.

You will be given previously attempted actions and their results, and based on that, please generate high-level planning code to finish the user's request.

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

Please determine the next action to perform:

If the returned information is sufficient to answer the user's question, respond with the answer based on the feedback. 
If the returned information is insufficient but the web actions have been completed, respond using your own knowledge.
printing  ```python
print("Final answer\n\nTHOUGHT: chain-of-thought\n\nANSWER: THE_FINAL_ANSWER")
```

If the returned information is valid for asking a further query (for example, when sub_query_2 is dependent on the result of sub_query_1), then print ```python
CallWeb(query=sub_query_2, target_url=your_target_url_2),
...
CallWeb(query=sub_query_n, target_url=your_target_url_n),
```
If sub_query_3 is dependent on the result of sub_query_2, then only output the first a few CallWeb:  ```python
CallWeb(query=sub_query_2, target_url=your_target_url_1)
```

If the returned information is not sufficient, or the previous action failed to return any information, you can decide to 

1) retry the previous action by printing the same CallWeb query, together with some feedback and suggestions to the web agent that performing the CallWeb action, by printing ```python
CallWeb(query=sub_query_2, target_url=your_target_url_2, feedback=your_feedback)
```
2) modify the CallWeb action by either changing the query or the target_url.
3) if the same sub_query cannot be finished for 2 times, then in the 3rd time, try to answer the sub_query using your internal knowledge, by printing  ```python
AskLLM(query=sub_query_1)
```, where sub_query_1 is the query from last step.
"""

FILE_LOCATIONS = "/app/UploadedFiles"
GLOBAL_DB_LOCATIONS = "/app/Database_global"
DB_LOCATIONS = "/app/Database_local"
CHARACTER_POOL_PATH = "/app/Character_pool"
CUSTOMIZED_CHARACTER_POOL_PATH = "/app/Customized_character_pool"
TOP_K_SENTENCES = 1000

ACTIVATE_KE = os.environ.get("ACTIVATE_KE", "True").lower() in ("true", "1", "t")
ACTIVATE_HISTORY = os.environ.get("ACTIVATE_HISTORY", "True").lower() in (
    "true",
    "1",
    "t",
)
ACTIVATE_SHORT_FEEDBACK = os.environ.get("ACTIVATE_SHORT_FEEDBACK", "True").lower() in (
    "true",
    "1",
    "t",
)
HISTORY_SUMMARY_LENGTH = os.environ.get("HISTORY_SUMMARY_LENGTH", 3)

from cognitive_kernel.memory_kernel.knowledge_engine import (
    KnowledgeEngine,
    KnowledgeEngineConfig,
)

print("ACTIVATE_KE:", ACTIVATE_KE)
print("ACTIVATE_HISTORY:", ACTIVATE_HISTORY)


async def generate_stream(current_connection, messages):
    try:
        full_response = ""
        async for message in current_connection.get_response_stream(messages):
            yield message
    except AttributeError:
        full_response = current_connection.get_response(messages)
        yield full_response


def load_function_prompt(path, function_name):
    """This function will load the function prompt from the description.txt file
    Args:
        path (str): location of the available functions
        function_name (str): name of the function
    Returns:
        str: the detailed description of the function
    """
    with open(path + "/" + function_name + "/description.txt", "r") as f:
        function_prompt = f.read()
    return function_prompt


def load_function_implementation(path, function_name):
    """This function will load the function implementation from the implementation.py file
    Args:
        path (str): location of the available functions
        function_name (str): name of the function
    Returns:
        str: the detailed implementation of the function
    """
    with open(os.path.join(path, function_name, "implementation.py"), "r") as f:
        function_implementation = f.read()
    return function_implementation


def load_function_example(path, function_name):
    """This function will load the function examples from the examplel.jsons file
    Args:
        path (str): location of the available functions
        function_name (str): location of the available functions

    Returns:
        list: examples of using the target function
    """
    function_exmaples = list()
    with open(os.path.join(path, function_name, "examples.jsonl"), "r") as f:
        for line in f:
            tmp_example = json.loads(line)
            tmp_example[-2]["content"] = (
                "Please generating the acting code for the following query: "
                + tmp_example[-2]["content"]
            )
            function_exmaples += tmp_example
    return function_exmaples


def load_single_character_info(character_pool_path, target_character):
    if target_character == "cognitiveKernel":
        file_name = f"{os.environ.get('MODEL_NAME', '')}_info.json"
        if not os.path.exists(file_name):
            file_name = "info.json"    
    else:
        file_name = "info.json"
    with open(
        os.path.join(character_pool_path, target_character, file_name), "r"
    ) as f:
        basic_info = json.load(f)
        current_character_name = basic_info["name"]
        tmp_character_info = dict()
        tmp_character_info["id"] = basic_info["id"]
        tmp_character_info["shown_title"] = basic_info["shown_title"]
        tmp_character_info["description"] = basic_info["description"]
        tmp_character_info["visible_users"] = basic_info["visible_users"]
        tmp_character_info["head_system_prompt"] = basic_info["head_system_prompt"]
        tmp_character_info["tail_system_prompt"] = basic_info["tail_system_prompt"]
        tmp_character_info["system_prompt_sequence"] = basic_info[
            "system_prompt_sequence"
        ]
        function2implementation = dict()
        function2prompts = dict()
        function2examples = dict()
        available_function_path = os.path.join(
            character_pool_path, target_character, "functions"
        )
        function_dirs = os.listdir(available_function_path)
        if target_character == "cognitiveKernel":
            function_dirs = ["CallWeb", "CallMemoryKernel", "AskLLM"]
        for tmp_function in function_dirs:
            if tmp_function[0] != ".":
                function2implementation[tmp_function] = load_function_implementation(
                    path=available_function_path, function_name=tmp_function
                )
                function2prompts[tmp_function] = load_function_prompt(
                    path=available_function_path, function_name=tmp_function
                )
                function2examples[tmp_function] = load_function_example(
                    path=available_function_path, function_name=tmp_function
                )
        tmp_character_info["function2implementation"] = function2implementation
        tmp_character_info["function2prompts"] = function2prompts
        tmp_character_info["function2examples"] = function2examples
        tmp_character_info["avatar_path"] = os.path.join(
            character_pool_path, target_character, "avatar.png"
        )
        avatar_destination_path = f"/app/static/avatar/{target_character}.png"
        shutil.copy(
            tmp_character_info["avatar_path"],
            avatar_destination_path,
        )
        return tmp_character_info, basic_info["global_db_info"]


def load_character_info(character_pool_path, character_type="global"):
    """This function will load the implementation of the activated functions from the config file.
    Returns:
        dict: a dictionary that maps the function name to the implementation of the function.
    """
    available_characters = os.listdir(character_pool_path)
    character_to_info = dict()
    global_db_info = dict()
    for tmp_character in available_characters:
        try:
            if tmp_character[0] == ".":  # skip the hidden files
                continue
            tmp_character_info, tmp_global_db_info = load_single_character_info(
                character_pool_path=character_pool_path, target_character=tmp_character
            )
            tmp_character_info["character_type"] = character_type
            character_to_info[tmp_character] = tmp_character_info
            global_db_info[tmp_character] = tmp_global_db_info
        except Exception as e:
            print(f"Error in loading {tmp_character}, error: {e}")
            pass

    return character_to_info, global_db_info


class CognitiveKernel(object):
    def __init__(
        self,
        args,
        memory_inference_urls=None,
        model_name="ck",
        service_ip="30.207.99.138:8000",
        gpt_model="gpt-3.5-turbo",
        critic="gpt",
        world_model_ip="30.207.99.138:8000"
    ) -> None:
        if model_name == "ck":
            self.LLM_connection = BaseModelConnection(ip=service_ip)
        else:
            self.LLM_connection = ChatGPTConnection(model_name=model_name)
        self.gpt_connection = ChatGPTConnection(model_name=gpt_model)
        if world_model_ip == "gpt":
            self.world_model_connection = self.gpt_connection
        else:
            if not "|" in world_model_ip:
                self.world_model_connection = [BaseModelConnection(ip=world_model_ip)]
            else:
                self.world_model_connection = [BaseModelConnection(ip=wm_ip) for wm_ip in world_model_ip.split("|")]
            

        if critic == "gpt":
            self.critic_connection = self.gpt_connection
        elif critic == "ck":
            self.critic_connection = BaseModelConnection(ip=service_ip)

        self.args = args
        self.model_name = model_name
        # Start to load the activated functions
        self.memory_inference_urls = memory_inference_urls
        self.character_to_info, self.global_db_info = load_character_info(
            character_pool_path=CHARACTER_POOL_PATH, character_type="global"
        )
        self.customized_character_to_info, self.customized_global_db_info = (
            load_character_info(
                character_pool_path=CUSTOMIZED_CHARACTER_POOL_PATH,
                character_type="customized",
            )
        )

        # merge the customized character info to the global character info
        for tmp_character in self.customized_character_to_info:
            self.character_to_info[tmp_character] = self.customized_character_to_info[
                tmp_character
            ]
            self.global_db_info[tmp_character] = self.customized_global_db_info[
                tmp_character
            ]
        self.username2character = dict()
        for tmp_character in self.character_to_info:
            for tmp_user in self.character_to_info[tmp_character]["visible_users"]:
                if tmp_user not in self.username2character:
                    self.username2character[tmp_user] = list()
                self.username2character[tmp_user].append(tmp_character)
        self.knowledge_engine = KnowledgeEngine(
            args=args, inference_urls=self.memory_inference_urls, mode="inference"
        )
        self.executor_manager = ExecutorManager()
        self.setup_global_db()

    def update_character(self, character_name, character_type):
        if character_type == "customized":
            tmp_character_info, tmp_global_db_info = load_single_character_info(
                character_pool_path=CUSTOMIZED_CHARACTER_POOL_PATH,
                target_character=character_name,
            )
            tmp_character_info["character_type"] = character_type
            self.character_to_info[character_name] = tmp_character_info
            self.global_db_info[character_name] = tmp_global_db_info
            for tmp_user in tmp_character_info["visible_users"]:
                if tmp_user not in self.username2character:
                    self.username2character[tmp_user] = list()
                self.username2character[tmp_user].append(character_name)
        else:
            raise NotImplementedError

    def delete_character(self, character_name):
        character_info = self.character_to_info[character_name]
        for tmp_user in character_info["visible_users"]:
            self.username2character[tmp_user].remove(character_name)
            if len(self.username2character[tmp_user]) == 0:
                del self.username2character[tmp_user]
        self.character_to_info.pop(character_name)
        self.global_db_info.pop(character_name)

    def get_all_characters(self, username):
        visible_characters = list()
        if username in self.username2character:
            visible_characters.extend(self.username2character[username])
        visible_characters.extend(self.username2character["all"])
        all_characters = list()
        for tmp_character_name in visible_characters:
            all_characters.append(
                {
                    "name": tmp_character_name,
                    "key": self.character_to_info[tmp_character_name]["id"],
                    "shownTitle": self.character_to_info[tmp_character_name][
                        "shown_title"
                    ],
                    "title": self.character_to_info[tmp_character_name]["description"],
                    "characterType": self.character_to_info[tmp_character_name][
                        "character_type"
                    ],
                }
            )
        return all_characters

    def setup_global_db(self):
        for tmp_character in self.global_db_info:
            for tmp_db in self.global_db_info[tmp_character]:
                tmp_config = KnowledgeEngineConfig(
                    {
                        "db_name": tmp_db,
                        "db_type": "Sqlite",
                        "db_path": f"{GLOBAL_DB_LOCATIONS}/{self.global_db_info[tmp_character][tmp_db]}.db",
                        "chunk_size": 1024,
                        "neighbor_size": 10,
                    }
                )
                self.knowledge_engine.setup_global_knowledge_engine_module(tmp_config)

    def _get_history_db_connection(self, CKStatus):
        current_model_name = CKStatus["current_model"]
        history_id = CKStatus["history_id"]
        print("current_model_name:", current_model_name)
        print("history_id:", history_id)
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": f"history_{current_model_name}_{history_id}",
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/history_{current_model_name}_{history_id}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config,
                module_name=f"history_{current_model_name}_{history_id}",
            )
        )
        return target_ke_module

    def _get_online_feedback_db_connection(self, current_model_name, current_user):
        assert current_user != ""
        print("current_model_name:", current_model_name)
        print("current user:", current_user)
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": f"online_feedback_{current_model_name}_{current_user}",
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/online_feedback_{current_model_name}_{current_user}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config,
                module_name=f"online_feedback_{current_model_name}_{current_user}",
            )
        )
        return target_ke_module

    def update_online_feedback_db(self, annotation_input_info):
        if ACTIVATE_SHORT_FEEDBACK:
            user_name = annotation_input_info["username"]
            print("user_name:", user_name)
            character_name = annotation_input_info["character_name"]
            print("character_name:", character_name)
            annotation = annotation_input_info["messages_in_train_format"]
            messages = json.loads(annotation)
            messages_reverse = messages[::-1]
            user_query = ""
            for tmp_round in messages_reverse:
                if tmp_round["role"] == "user":
                    user_query = tmp_round["content"]
                    break
            print("messages:", messages)
            print("user_query:", user_query)
            current_db_connection = self._get_online_feedback_db_connection(
                character_name, user_name
            )
            current_db_connection.update_from_documents(
                [user_query], [annotation], visualize=False
            )
        else:
            return

    async def _update_history_db(self, messages, target_ke_module):
        start_time = time.time()
        raw_messages = []
        current_message = {"user_query": "", "system_response": ""}
        for tmp_round in messages:
            if tmp_round["role"] == "user":
                if current_message["user_query"] != "":
                    raw_messages.append(copy.deepcopy(current_message))
                current_message["user_query"] = tmp_round["content"]
            elif tmp_round["role"] == "assistant":
                current_message["system_response"] = tmp_round["content"]
        raw_messages.append(copy.deepcopy(current_message))

        current_timestamp = str(datetime.now())
        tmp_round = raw_messages[-1]["user_query"]
        tmp_meta_data = json.dumps(
            {"timestamp": current_timestamp, "content_type": "user_query"}
        )
        target_ke_module.update_from_documents(
            [tmp_round], [tmp_meta_data], visualize=False
        )

        """
        if len(raw_messages) % HISTORY_SUMMARY_LENGTH == 0:
            last_few_dialog = []
            for tmp_round in raw_messages[-HISTORY_SUMMARY_LENGTH:]:
                last_few_dialog.append({"role": "user", "content": tmp_round["user_query"]})
                last_few_dialog.append({"role": "assistant", "content": tmp_round["system_response"]})
            last_few_dialog = json.dumps(last_few_dialog, ensure_ascii=False)
            tmp_summary = target_ke_module.dialog_summarization_batch(input_queries=[last_few_dialog])[0]
            tmp_summary_meta_data = json.dumps({"timestamp": current_timestamp, "content_type": "summary"})
            target_ke_module.update_from_documents([tmp_summary], [tmp_summary_meta_data], visualize=False)
        """

        print("updating history time:", time.time() - start_time)

    def update_knowledge_engine(self, db_name, db_path, sentences, metadata=None):
        debug_config = KnowledgeEngineConfig(
            {
                "db_name": db_name,
                "db_type": "Sqlite",
                "db_path": db_path,
                "chunk_size": 1024,
                "neighbor_size": 10,
                "retrieval_type": "doc",
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=debug_config
            )
        )

        target_ke_module.update_from_documents(
            sentences, metadata=metadata, visualize=True
        )
        target_ke_module.setup_retriever()

    def retrieve_feedback(self, query, CKStatus):
        """This function will find the top 1 relevant annotation based on the query.

        Args:
            query (_type_): _description_
            CKStatus (_type_): _description_
        """
        current_model_name = CKStatus["current_model"]
        username = CKStatus["username"]
        db_name = f"online_feedback_{current_model_name}_{username}"
        print("trying to retrieve feedback from db:", db_name)

        current_db_connection = self._get_online_feedback_db_connection(
            current_model_name, username
        )

        relevant_history, relevant_meta_data, _ = (
            current_db_connection.find_relevant_knowledge_single_document(
                query=query,
                retrieval_mode="doc",
                sim_threshold=0.5,
                soft_match_top_k=10,
            )
        )
        print(relevant_meta_data)
        return relevant_meta_data

    def retrieve_history(self, query, CKStatus):
        """This function will find the relevant history based on the query.
        Args:
            messages (_type_): _description_
        """
        current_model_name = CKStatus["current_model"]
        history_id = CKStatus["history_id"]
        db_name = f"history_{current_model_name}_{history_id}"
        if db_name not in self.knowledge_engine.local_id2module:
            print("We do not have the history db yet.")
            return [], []

        relevant_history, relevant_meta_data, _ = (
            self.knowledge_engine.find_relevant_info_single_db(
                db_name=f"history_{current_model_name}_{history_id}",
                query=query,
                retrieval_mode="doc",
                soft_match_top_k=10,
                sim_threshold=-10000,
            )
        )
        relevant_meta_data = [json.loads(md) for md in relevant_meta_data]
        time2messages = dict()
        time2summaries = dict()
        for i, tmp_history in enumerate(relevant_history):
            if relevant_meta_data[i]["content_type"] == "summary":
                time2summaries[relevant_meta_data[i]["timestamp"]] = tmp_history
            elif relevant_meta_data[i]["content_type"] == "user_query":
                time2messages[relevant_meta_data[i]["timestamp"]] = tmp_history
            else:
                raise NotImplementedError

        sorted_times = sorted(time2messages.keys())
        sorted_messages = []
        for t in sorted_times:
            sorted_messages.append(time2messages[t])

        sorted_times = sorted(time2summaries.keys())
        sorted_summaries = []
        for t in sorted_times:
            sorted_summaries.append(time2summaries[t])

        print("retrieved user queries:", sorted_messages)
        print("retrieved summaries:", sorted_summaries)
        return sorted_messages, sorted_summaries

    def _get_system_message(self, query, CKStatus):
        """This function will generate the

        Args:
            messages (_type_): _description_
            with_examples (bool, optional): _description_. Defaults to True.
        """
        session_id = CKStatus["session_id"]
        current_character = CKStatus["current_model"]
        system_message = (
            self.character_to_info[current_character]["head_system_prompt"] + "\n"
        )

        for tmp_system_prompt_step in self.character_to_info[current_character][
            "system_prompt_sequence"
        ]:
            if tmp_system_prompt_step["pre_defined"] == True:
                if tmp_system_prompt_step["name"] == "available_functions":
                    system_message += "We have the following available functions:\n"
                    for tmp_function in self.character_to_info[current_character][
                        "function2prompts"
                    ]:
                        system_message += (
                            self.character_to_info[current_character][
                                "function2prompts"
                            ][tmp_function]
                            + "\n"
                        )
                elif tmp_system_prompt_step["name"] == "uploaded_files":
                    if ACTIVATE_KE:
                        file_names = []
                        file_db2descriptions = []
                        for tmp_file_name in CKStatus["uploaded_files"]:
                            file_db2descriptions.append(f"{tmp_file_name}_{session_id}")
                            file_names.append(f"{FILE_LOCATIONS}/{tmp_file_name}")
                        if len(file_db2descriptions) > 0:
                            system_message += (
                                "Avaible DB names and their descriptions:\n"
                            )
                            system_message += "\n".join(file_db2descriptions)
                        system_message += "\n"
                        if len(file_names) > 0:
                            system_message += "Available file paths:\n"
                            system_message += "\n".join(file_names)
                        system_message += "\n"
                        if file_db2descriptions:
                            system_message += "Attention: User just uploaded a file. Please pay attention to that.\n"
                    else:
                        file_names = []
                        for tmp_file_name in CKStatus["uploaded_files"]:
                            file_names.append(f"{FILE_LOCATIONS}/{tmp_file_name}")
                        system_message += "\n"
                        if len(file_names) > 0:
                            system_message += "Available file paths:\n"
                            system_message += "\n".join(file_names)
                        system_message += "\n"
                else:
                    raise NotImplementedError(
                        f"pre_defined step {tmp_system_prompt_step['name']} is not implemented."
                    )
            else:
                if tmp_system_prompt_step["step_type"] == "static":
                    system_message += tmp_system_prompt_step["head"] + "\n"
                    system_message += (
                        "\n".join(tmp_system_prompt_step["content"]) + "\n"
                    )
                elif tmp_system_prompt_step["step_type"] == "dynamic":
                    system_message += tmp_system_prompt_step["head"] + "\n"
                    system_message += (
                        "\n".join(CKStatus[tmp_system_prompt_step["CK_status_key"]])
                        + "\n"
                    )
                else:
                    raise NotImplementedError(
                        f"step type {tmp_system_prompt_step['step_type']} is not implemented."
                    )
        if ACTIVATE_HISTORY:
            start_time = time.time()
            relevent_messages, relevant_summaries = self.retrieve_history(
                query=query, CKStatus=CKStatus
            )
            print("retrieving history time:", time.time() - start_time)
            system_message += "The following sentences are retrieved from the user's previous dialogs. If any of them are relevant to the user's current query then use them when responding to the user: "
            system_message += str(relevent_messages)
            system_message += "\n\n"
        if ACTIVATE_SHORT_FEEDBACK:
            start_time = time.time()
            relevant_feedback = self.retrieve_feedback(query=query, CKStatus=CKStatus)
            if len(relevant_feedback) > 0:
                relevant_feedback = relevant_feedback[:1]
            print("retrieving history time:", time.time() - start_time)
            system_message += "We have the following feedback:\n"
            for tmp_feedback in relevant_feedback:
                system_message += json.dumps(json.loads(tmp_feedback)[1:]) + "\n"
            system_message += "\n"
        system_message += (
            self.character_to_info[current_character]["tail_system_prompt"] + "\n"
        )
        return system_message

    def AskLLM(self, query):
        """This function will ask the base language model with a single query
        Args:
            query (str): input query
        Returns:
            str: returned query
        """
        tmp_messages = [
            {
                "role": "system",
                "name": "head",
                "content": "You are Kirkland, a helpful AI assistant, with no access to external functions.",
            },
            {"role": "user", "content": query},
        ]
        tmp_result = self.LLM_connection.get_response(tmp_messages)
        return tmp_result

    async def planning_execution(self, planning_code, username, session_id, message_id, search_id=0, search=False, sample=False, search_from_step=1, feedback=None, world_model_search=None, search_depth=1 ):
        current_id = f"{username}_{session_id}"
        newly_created_executor, current_executor = (
            self.executor_manager.get_or_create_executor(executor_id=current_id)
        )
        if search or sample:
            search_prompt = f"search_id={search_id},search_from_step={search_from_step},"
        else:
            search_prompt = ""

        if feedback is not None:
            assert sample, "feedback is currently only supported for pure sampling"
            feedback_prompt=f"feedback=\"{feedback}\","
        else:
            feedback_prompt=""

        if newly_created_executor:
            code = f"""
def CallWeb(query, target_url):
    for tmp_response in my_call_web(
        query=query,
        target_url=target_url,
        session_id='{session_id}',
        message_id='{message_id}',
        username='{username}',
        world_model_search='{world_model_search}',
        search_depth=int('{search_depth}'),
        {search_prompt}
        {feedback_prompt}
    ):
        print(tmp_response)
    """
            code += planning_code
        else:
            code = planning_code
        global_vars = globals().copy()
        global_vars["CallMemoryKernel"] = self.CallMemoryKernel
        global_vars["AskLLM"] = self.AskLLM
        if search:
            global_vars["my_call_web"] = self.my_call_web_search
        elif sample:
            global_vars["my_call_web"] = self.my_call_web_sample
        else:
            global_vars["my_call_web"] = self.my_call_web
        current_executor.submit_code(
            code=code,
            global_variable=global_vars,
        )
        async for tmp_output in current_executor.async_output():
            yield tmp_output
    
    async def planning_execution_wm(self, planning_code, username, session_id, message_id, world_model_search=None, search_depth=1, search_id=0, search=False, sample=False, search_from_step=1, feedback=None):
        current_id = f"{username}_{session_id}"
        newly_created_executor, current_executor = (
            self.executor_manager.get_or_create_executor(executor_id=current_id)
        )
        if search or sample:
            search_prompt = f"search_id={search_id},search_from_step={search_from_step},"
        else:
            search_prompt = ""

        if feedback is not None:
            assert sample, "feedback is currently only supported for pure sampling"
            feedback_prompt=f"feedback=\"{feedback}\","
        else:
            feedback_prompt=""

        if newly_created_executor:
            code = f"""
def CallWeb(query, target_url):
    for tmp_response in my_call_web(
        query=query,
        target_url=target_url,
        session_id='{session_id}',
        message_id='{message_id}',
        username='{username}',
        world_model_search='{world_model_search}',
        search_depth=int('{search_depth}'),
        {search_prompt}
        {feedback_prompt}
    ):
        print(tmp_response)
    """
            code += planning_code
        else:
            code = planning_code
        global_vars = globals().copy()
        global_vars["CallMemoryKernel"] = self.CallMemoryKernel
        global_vars["AskLLM"] = self.AskLLM
        if search:
            global_vars["my_call_web"] = self.my_call_web_search
        elif sample:
            global_vars["my_call_web"] = self.my_call_web_sample
        else:
            global_vars["my_call_web"] = self.my_call_web_wm
        current_executor.submit_code(
            code=code,
            global_variable=global_vars,
        )
        async for tmp_output in current_executor.async_output():
            yield tmp_output

    def my_call_web_sample(self, query, target_url, session_id, message_id, username, 
                           search_id=0, search_from_step=1, feedback=None):
        for tmp_response in call_web_sample(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            search_id=search_id,
            search_from_step=search_from_step,
            feedback=feedback,
            critic_connection=self.critic_connection,
        ):
            yield tmp_response

    def my_call_web_search(self, query, target_url, session_id, message_id, username, search_id=0):
        for tmp_response in call_web_search_test(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            search_id=search_id
        ):
            yield tmp_response

    def my_call_web(self, query, target_url, session_id, message_id, username, world_model_search=None, search_depth=1):
        for tmp_response in call_web(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            world_model_search=world_model_search,
            search_depth=search_depth,
            world_model_connection=self.world_model_connection,
            critic_model_connection=self.gpt_connection,
        ):
            yield tmp_response
    
    def my_call_web_wm(self, query, target_url, session_id, message_id, username, world_model_search=None, search_depth=1):
        for tmp_response in call_web(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            world_model_search=world_model_search,
            search_depth=search_depth,
            world_model_connection=self.world_model_connection,
            critic_model_connection=self.gpt_connection,
        ):
            yield tmp_response

    async def planning_execution_webcanvas(
        self, planning_code, username, session_id, message_id, max_steps, storage_state
    ):
        current_id = f"{username}_{session_id}"
        newly_created_executor, current_executor = (
            self.executor_manager.get_or_create_executor(executor_id=current_id)
        )
        if newly_created_executor:
            code = f"""
def CallWeb(query, target_url):
    for tmp_response in my_call_web(
        query=query,
        target_url=target_url,
        session_id='{session_id}',
        message_id='{message_id}',
        username='{username}',
        max_steps={max_steps},
        storage_state={storage_state},
    ):
        print(tmp_response)
    """
            code += planning_code
        else:
            code = planning_code
        global_vars = globals().copy()
        global_vars["CallMemoryKernel"] = self.CallMemoryKernel
        global_vars["AskLLM"] = self.AskLLM
        global_vars["my_call_web"] = self.my_call_webcanvas
        current_executor.submit_code(
            code=code,
            global_variable=global_vars,
        )
        async for tmp_output in current_executor.async_output():
            yield tmp_output

    def my_call_webcanvas(
        self,
        query,
        target_url,
        session_id,
        message_id,
        username,
        max_steps,
        storage_state,
    ):
        for tmp_response in call_webcanvas(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            max_steps=max_steps,
            storage_state=storage_state,
        ):
            yield tmp_response

    def CallMemoryKernel(self, query, db_name):
        """This function will call the local memory kernel to get the corresponding information.
        Args:
            query (_type_): _description_
            db_name (_type_): _description_
        """

        retrieved_info, relevant_meta_data, _ = (
            self.knowledge_engine.find_relevant_info_single_db(
                db_name=db_name,
                query=query,
                retrieval_mode="doc",
            )
        )

        response = [
            (meta, text) for meta, text in zip(relevant_meta_data, retrieved_info)
        ]

        return response
    
    async def execute_and_return_results(self, 
                                    planning_code, 
                                    username, 
                                    CKStatus, 
                                    message_id,
                                    updated_messages,
                                    current_status,
                                    current_pos,
                                    task_id="",
                                    search_id=0,
                                    search=False,
                                    sample=False,
                                    search_from_step=1,
                                    feedback=None):
        results = []
        if search or sample:
            session_id = CKStatus["session_id"]+"@"+str(task_id)
        else:
            session_id = CKStatus["session_id"]
        async for tmp_execution_result in self.planning_execution(
            planning_code=planning_code,
            username=username,
            session_id=session_id,
            message_id=message_id,
            search_id=search_id,
            search=search,
            sample=sample,
            search_from_step=search_from_step,
            feedback=feedback
        ):
            print("in execute_and_return_results, tmp_execution_result:", tmp_execution_result)
            if "[WEB]" in tmp_execution_result:
                if current_status == "empty":
                    execution_result = tmp_execution_result
                    current_pos += 1
                    updated_messages.append(
                        {
                            "group": f"assistant_web_result",
                            "pos": current_pos,
                            "content": execution_result,
                        }
                    )
                    current_status = "web"
                elif current_status == "web":
                    execution_result += tmp_execution_result
                    updated_messages[-1]["content"] = execution_result
                else:
                    current_pos += 1
                    execution_result = tmp_execution_result
                    updated_messages.append(
                        {
                            "group": f"assistant_web_result",
                            "pos": current_pos,
                            "content": execution_result,
                        }
                    )
                    current_status = "web"
            elif "[/WEB]" in tmp_execution_result:
                assert current_status == "web"
                web_content = tmp_execution_result.split("[/WEB]")[0]
                normal_content = tmp_execution_result.split("[/WEB]")[1]
                execution_result += web_content
                updated_messages[-1]["content"] = execution_result
                current_pos += 1
                execution_result = normal_content
                updated_messages.append(
                    {
                        "group": f"assistant_execution_result",
                        "pos": current_pos,
                        "content": execution_result,
                    }
                )
                current_status = "text"
            else:
                if current_status == "empty":
                    current_pos += 1
                    updated_messages.append(
                        {
                            "group": f"assistant_execution_result",
                            "pos": current_pos,
                            "content": tmp_execution_result,
                        }
                    )
                    current_status = "text"
                    execution_result = tmp_execution_result
                else:
                    print("I should not go there")
                    execution_result += tmp_execution_result
                    updated_messages[-1]["content"] = execution_result
            results.append( [json.dumps(updated_messages), execution_result, current_pos])

        return results
    
    def evaluate_first_action(self, feedback_type, user_query, action_output):
        if feedback_type == "gpt":
            prompt = f"""TASK: {user_query}

The latest action : {action_output}
"""
            print("evaluator prompt", prompt)
            eval_output = self.critic_connection.get_response([
                {"role": "system", 
                "content": SYSTEM_PROMPT_START_TEXT},
                {"role": "user", 
                "content": prompt}
            ])
            eval_result = parse_eval_output(eval_output)
            # {
            #     'THOUGHT': thought,
            #     'EVALUATION': evaluation,
            #     'FEEDBACK': feedback
            # }
            print("gpt-4o-eval", eval_result)

            return eval_result
        return  {
                'THOUGHT': "",
                'EVALUATION': "",
                'FEEDBACK': ""
            }
    
    async def concurrent_callweb_v2(self,
        messages,
        CKStatus,
        username="test",
        message_id="",
        stream=True,
        feedback=None):
        """
            feedback: 
                currently only support gpt feedback.
        """
        # set as false for now.
        stream=False

        user_query = messages[-1]["content"]
        local_messages = copy.deepcopy(messages)
        system_message = self._get_system_message(query=user_query, CKStatus=CKStatus)

        local_messages = [
            {"role": "system", "name": "head", "content": system_message}
        ] + local_messages
        local_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_data_logging = copy.deepcopy(local_messages)
        current_pos = 0
        updated_messages = [
            {
                "group": f"assistant_slow_thinking",
                "pos": current_pos,
                "content": "",
            }
        ]
        continue_generation = True
        previous_raw_planning_code = ""

        results = []

        if previous_raw_planning_code == "":
            raw_data_logging.append({"role": "assistant", "content": ""})
            tmp_output = self.LLM_connection.get_response(local_messages)

            # print("critic_connection")
            # print("ftqftq", self.critic_connection)
            # gpt_critic = self.critic_connection.get_response(local_messages)



                ### feedback:
            if feedback == "gpt":
                prompt = f"""TASK: {user_query}

The latest action : {tmp_output}
"""
                print("evaluator prompt", prompt)
                eval_output = self.critic_connection.get_response([
                    {"role": "system", 
                    "content": SYSTEM_PROMPT_START_TEXT},
                    {"role": "user", 
                    "content": prompt}
                ])
                eval_result = parse_eval_output(eval_output)
                print("gpt-4o-eval", eval_result)

                if eval_result['EVALUATION'] == 'POOR':
                    # regenerate the action
                    msgs = copy.deepcopy(local_messages)
                    msgs[-1]['content'] += eval_result['FEEDBACK']
                    tmp_output = self.LLM_connection.get_response_search_test(msgs)
                    print("new action after feedback", tmp_output)



            # print("ftqftq critic", gpt_critic)

            updated_messages[-1]["content"] += tmp_output
            raw_data_logging[-1]["content"] += tmp_output
            yield json.dumps(updated_messages)

            raw_planning_code = updated_messages[-1]["content"]
        else:
            raw_planning_code = previous_raw_planning_code
        # '''

        local_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        
        parts = raw_planning_code.split("```")
        planning_code = parts[1] if len(parts) > 1 else ""
        planning_code = planning_code.replace("python", "")
        execution_result = ""
        current_status = "empty"
        
        print("ftqftq planning_code", planning_code, "end of planning_code")

        ### Core function:

        start = time.time()

        tasks = [
            self.execute_and_return_results(planning_code, 
                                username, 
                                CKStatus, 
                                message_id,
                                copy.deepcopy(updated_messages),
                                current_status,
                                current_pos,
                                _,
                                search_id=_, 
                                sample=True,
                                feedback=feedback
                                )
            for _ in range(4)
        ]
        all_results = await asyncio.gather(*tasks)

        execution_results = []
        for i, results in enumerate(all_results):
            for counter, (output, execution_result, current_pos) in enumerate(results):
                # print("ftqftq output", json.loads(output))
                yield output
            execution_results.append(execution_result)
            print(f"job {i} execution result: {execution_result} ")
        
        updated_messages = json.loads(output)
        # print("updated message", updated_messages)
        from collections import Counter
        def majority_vote(text_list):
            return sorted(Counter(text_list).items(), key = lambda x:x[1], reverse=True)[0][0]

        print("majority vote", majority_vote(execution_results))
        updated_messages[-1]['content'] = majority_vote(execution_results)

        end = time.time()
        print("4 times run time", end - start)
        #### 

        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            # pattern to remove the broswer ID and page ID
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Directly answer the user query.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Asking for more information from the user.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Asking for more information from the user.",
                }
            )
        current_pos += 1
        updated_messages.append(
            {
                "group": f"assistant_final_output",
                "pos": current_pos,
                "content": "",
            }
        )
        raw_data_logging.append({"role": "assistant", "content": ""})
        final_output = ""

        ### FTQ: uncomment this later

        final_output = self.LLM_connection.get_response(local_messages)
        raw_data_logging[-1]["content"] = final_output
        if final_output not in ["<|im_continue|>"]:
            updated_messages[-1]["content"] += final_output
            yield json.dumps(updated_messages)
        
        if "<|im_continue|>" in final_output:
            continue_generation = True
            previous_raw_planning_code = final_output
            updated_messages[-1]["content"] = final_output.replace(
                "<|im_continue|>", ""
            )
            updated_messages[-1]["group"] = "assistant_slow_thinking"
            yield json.dumps(updated_messages)
        else:
            continue_generation = False
        update_or_create_rawdata(
            session_id=CKStatus["session_id"],
            message_id=message_id,
            username=username,
            messages_in_train_format=raw_data_logging,
            updated_time=datetime.now().isoformat(),
        )

        # saving the local history to a history db.
        if ACTIVATE_HISTORY:
            target_history_ke = self._get_history_db_connection(CKStatus=CKStatus)
            task = asyncio.create_task(
                self._update_history_db(raw_data_logging, target_history_ke)
            )
    
    async def concurrent_callweb_v3(self,
        messages,
        CKStatus,
        username="test",
        message_id="",
        stream=True,
        feedback=None,
        K_plan=2):
        """
            feedback: 
                currently only support gpt feedback.
        """
        # set as false for now.
        stream=False

        user_query = messages[-1]["content"]
        local_messages = copy.deepcopy(messages)
        system_message = self._get_system_message(query=user_query, CKStatus=CKStatus)

        local_messages = [
            {"role": "system", "name": "head", "content": system_message}
        ] + local_messages
        local_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_data_logging = copy.deepcopy(local_messages)
        current_pos = 0
        updated_messages = [
            {
                "group": f"assistant_slow_thinking",
                "pos": current_pos,
                "content": "",
            }
        ]
        continue_generation = True
        previous_raw_planning_code = ""


        ### 1. Generate K_plan callweb queries.

        start = time.time()

        print("type", type(self.LLM_connection.get_response_search_test))

        planning_tasks = [
            asyncio.to_thread(self.LLM_connection.get_response_search_test,
                               copy.deepcopy(local_messages),
                              temperature=0.7, 
                              seed=_)
            for _ in range(10)
        ]

        planning_results = await asyncio.gather(*planning_tasks)
        # await asyncio.gather(*planning_tasks)

        end = time.time()
        print("10 time get_response time: ", end - start)
        # 10 times 1.8s
        # 1 times 0.74s.

        for res in planning_results:
            updated_messages[-1]["content"] += res
            yield json.dumps(updated_messages)



        results = []

        ### 2. start K_plan browsers.

        browsers = []

        

        # browser_id = get_browser(storage_state, geo_location)

        '''
        if previous_raw_planning_code == "":
            raw_data_logging.append({"role": "assistant", "content": ""})
            tmp_output = self.LLM_connection.get_response(local_messages)


            # print("ftqftq critic", gpt_critic)

            updated_messages[-1]["content"] += tmp_output
            raw_data_logging[-1]["content"] += tmp_output
            yield json.dumps(updated_messages)

            raw_planning_code = updated_messages[-1]["content"]
        else:
            raw_planning_code = previous_raw_planning_code

        local_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        
        parts = raw_planning_code.split("```")
        planning_code = parts[1] if len(parts) > 1 else ""
        planning_code = planning_code.replace("python", "")
        execution_result = ""
        current_status = "empty"
        
        print("ftqftq planning_code", planning_code, "end of planning_code")

        ### Core function:

        start = time.time()

        tasks = [
            self.execute_and_return_results(planning_code, 
                                username, 
                                CKStatus, 
                                message_id,
                                copy.deepcopy(updated_messages),
                                current_status,
                                current_pos,_,
                                search_id=_, 
                                sample=True,
                                feedback=feedback
                                )
            for _ in range(4)
        ]
        all_results = await asyncio.gather(*tasks)

        execution_results = []
        for i, results in enumerate(all_results):
            for counter, (output, execution_result, current_pos) in enumerate(results):
                # print("ftqftq output", json.loads(output))
                yield output
            execution_results.append(execution_result)
            print(f"job {i} execution result: {execution_result} ")
        
        updated_messages = json.loads(output)
        # print("updated message", updated_messages)
        from collections import Counter
        def majority_vote(text_list):
            return sorted(Counter(text_list).items(), key = lambda x:x[1], reverse=True)[0][0]

        print("majority vote", majority_vote(execution_results))
        updated_messages[-1]['content'] = majority_vote(execution_results)

        end = time.time()
        print("4 times run time", end - start)
        #### 

        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            # pattern to remove the broswer ID and page ID
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Directly answer the user query.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            local_messages.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Asking for more information from the user.",
                }
            )
            raw_data_logging.append(
                {
                    "role": "system",
                    "name": "actor",
                    "content": f"Asking for more information from the user.",
                }
            )
        current_pos += 1
        updated_messages.append(
            {
                "group": f"assistant_final_output",
                "pos": current_pos,
                "content": "",
            }
        )
        raw_data_logging.append({"role": "assistant", "content": ""})
        final_output = ""

        ### FTQ: uncomment this later

        final_output = self.LLM_connection.get_response(local_messages)
        raw_data_logging[-1]["content"] = final_output
        if final_output not in ["<|im_continue|>"]:
            updated_messages[-1]["content"] += final_output
            yield json.dumps(updated_messages)
        
        if "<|im_continue|>" in final_output:
            continue_generation = True
            previous_raw_planning_code = final_output
            updated_messages[-1]["content"] = final_output.replace(
                "<|im_continue|>", ""
            )
            updated_messages[-1]["group"] = "assistant_slow_thinking"
            yield json.dumps(updated_messages)
        else:
            continue_generation = False
        update_or_create_rawdata(
            session_id=CKStatus["session_id"],
            message_id=message_id,
            username=username,
            messages_in_train_format=raw_data_logging,
            updated_time=datetime.now().isoformat(),
        )

        # saving the local history to a history db.
        if ACTIVATE_HISTORY:
            target_history_ke = self._get_history_db_connection(CKStatus=CKStatus)
            task = asyncio.create_task(
                self._update_history_db(raw_data_logging, target_history_ke)
            )
        '''
    
    # generate_concurrent_test
    async def concurrent_callweb(self,
        messages,
        CKStatus,
        username="test",
        message_id="",
        stream=True):
        """This function is a test for performing async callweb and return the results

        Args:
            messages (_type_): the target message
            role (str, optional): which role we should as the model to perform. select from ['actor', 'critic', 'improver']. Defaults to 'actor'.
        """

        print("hello, in ftq search test")
        operations = {
            "web":"",
        }

        # set as false for now.
        stream=False

        user_query = messages[-1]["content"]
        local_messages = copy.deepcopy(messages)
        system_message = self._get_system_message(query=user_query, CKStatus=CKStatus)

        local_messages = [
            {"role": "system", "name": "head", "content": system_message}
        ] + local_messages
        local_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_data_logging = copy.deepcopy(local_messages)
        current_pos = 0
        updated_messages = [
            {
                "group": f"assistant_slow_thinking",
                "pos": current_pos,
                "content": "",
            }
        ]
        continue_generation = True
        previous_raw_planning_code = ""

        results = []

        # only two options for the first step.
        for key, val in operations.items():

            # raw_planning_code = "```python\nCallWeb(query=\"Search for Yangqiu Song on Google and find out who he is.\", target_url=\"https://www.google.com/\")\n```"
            # updated_messages[-1]["content"] += raw_planning_code
            # raw_data_logging[-1]["content"] += raw_planning_code
            
            # '''
            # 暂时comment掉
            if previous_raw_planning_code == "":
                action_header = val
                updated_messages[-1]["content"] += action_header
                raw_data_logging.append({"role": "assistant", "content": ""})
                # async for tmp_output in generate_stream(
                #     current_connection=self.LLM_connection, messages=local_messages, stream=stream, additional_prompt=action_header
                # ):
                tmp_output = self.LLM_connection.get_response(local_messages)
                updated_messages[-1]["content"] += tmp_output
                raw_data_logging[-1]["content"] += tmp_output
                yield json.dumps(updated_messages)

                raw_planning_code = updated_messages[-1]["content"]
            else:
                raw_planning_code = previous_raw_planning_code
            # '''

            local_messages.append({"role": "assistant", "content": raw_planning_code})
            planning_decision = "code"
            
            parts = raw_planning_code.split("```")
            planning_code = parts[1] if len(parts) > 1 else ""
            planning_code = planning_code.replace("python", "")
            execution_result = ""
            current_status = "empty"
            print("ftqftq planning_code", planning_code, "end of planning_code")


            ### Core function:

            start = time.time()

            tasks = [
                self.execute_and_return_results(planning_code, 
                                    username, 
                                    CKStatus, 
                                    message_id,
                                    copy.deepcopy(updated_messages),
                                    current_status,
                                    current_pos,_,search_id=_, search=True)
                for _ in range(1)
            ]
            all_results = await asyncio.gather(*tasks)


            for i, results in enumerate(all_results):
                for counter, (output, execution_result, current_pos) in enumerate(results):
                    print("ftqftq output", json.loads(output))
                    # update_or_create_rawdata(
                    #     session_id=CKStatus["session_id"]+"_"+str(i),
                    #     message_id=f"{message_id}@@web@@{counter+1}",
                    #     username=username,
                    #     messages_in_train_format=json.loads(output),
                    #     updated_time=datetime.now().isoformat(),
                    # )
                    yield output
            
            updated_messages = json.loads(output)
            
            
            end = time.time()
            print("4 times run time", end - start)
            #### 

            if planning_decision == "code":
                cleaned_execution_result = execution_result.replace("stop", "")
                # pattern to remove the broswer ID and page ID
                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                new_cleaned = []
                for step in cleaned_execution_result.split("\n"):
                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                cleaned_execution_result = "\n".join(new_cleaned)
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
            elif planning_decision == "direct":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
            elif planning_decision == "need_info":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
            current_pos += 1
            updated_messages.append(
                {
                    "group": f"assistant_final_output",
                    "pos": current_pos,
                    "content": "",
                }
            )
            raw_data_logging.append({"role": "assistant", "content": ""})
            final_output = ""

            ### FTQ: uncomment this later

            final_output = self.LLM_connection.get_response(local_messages)
            raw_data_logging[-1]["content"] = final_output
            if final_output not in ["<|im_continue|>"]:
                updated_messages[-1]["content"] += final_output
                yield json.dumps(updated_messages)
            
            if "<|im_continue|>" in final_output:
                continue_generation = True
                previous_raw_planning_code = final_output
                updated_messages[-1]["content"] = final_output.replace(
                    "<|im_continue|>", ""
                )
                updated_messages[-1]["group"] = "assistant_slow_thinking"
                yield json.dumps(updated_messages)
            else:
                continue_generation = False
            update_or_create_rawdata(
                session_id=CKStatus["session_id"],
                message_id=message_id,
                username=username,
                messages_in_train_format=raw_data_logging,
                updated_time=datetime.now().isoformat(),
            )

        # saving the local history to a history db.
        if ACTIVATE_HISTORY:
            target_history_ke = self._get_history_db_connection(CKStatus=CKStatus)
            task = asyncio.create_task(
                self._update_history_db(raw_data_logging, target_history_ke)
            )


    async def generate(
        self,
        messages,
        CKStatus,
        username="test",
        message_id="",
        max_trials=3,
    ):
        """This function is the main generation function of cognitive kernel. It will respond based on the current messages and role.

        Args:
            messages (_type_): the target message
            message_log (_type_): the target message log
            role (str, optional): which role we should as the model to perform. select from ['actor', 'critic', 'improver']. Defaults to 'actor'.
        """
        user_query = messages[-1]["content"]
        local_messages = copy.deepcopy(messages)
        print(1)
        system_message = self._get_system_message(query=user_query, CKStatus=CKStatus)
        print(2)
        local_messages = [
            {"role": "system", "name": "head", "content": system_message}
        ] + local_messages
        local_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_data_logging = copy.deepcopy(local_messages)
        current_pos = 0
        updated_messages = [
            {
                "group": f"assistant_slow_thinking",
                "pos": current_pos,
                "content": "",
            }
        ]
        current_trial = 0
        continue_generation = True
        previous_raw_planning_code = ""
        while current_trial < max_trials and continue_generation:
            current_trial += 1
            if previous_raw_planning_code == "":
                raw_data_logging.append({"role": "assistant", "content": ""})
                async for tmp_output in generate_stream(
                    current_connection=self.LLM_connection, messages=local_messages
                ):
                    updated_messages[-1]["content"] += tmp_output
                    raw_data_logging[-1]["content"] += tmp_output
                    yield json.dumps(updated_messages)

                raw_planning_code = updated_messages[-1]["content"]
            else:
                raw_planning_code = previous_raw_planning_code

            # zero-shot Llama3.3 special token
            if "<|python_tag|>" in raw_planning_code:
                raw_planning_code = raw_planning_code.replace("<|python_tag|>", "```python\n")

            local_messages.append({"role": "assistant", "content": raw_planning_code})
            planning_decision = "code"
            if "Direct answering" in raw_planning_code:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
            elif "Additional Info" in raw_planning_code:
                planning_decision = "need_info"
                planning_code = raw_planning_code
                execution_result = ""
            else:
                if "```" in raw_planning_code:
                    planning_decision = "code"
                    parts = raw_planning_code.split("```")
                    planning_code = parts[1] if len(parts) > 1 else ""
                    planning_code = planning_code.replace("python", "")
                    execution_result = ""
                    current_status = "empty"
                    async for tmp_execution_result in self.planning_execution(
                        planning_code=planning_code,
                        username=username,
                        session_id=CKStatus["session_id"],
                        message_id=message_id,
                    ):
                        if "[WEB]" in tmp_execution_result:
                            if current_status == "empty":
                                execution_result = tmp_execution_result
                                current_pos += 1
                                updated_messages.append(
                                    {
                                        "group": f"assistant_web_result",
                                        "pos": current_pos,
                                        "content": execution_result,
                                    }
                                )
                                current_status = "web"
                            elif current_status == "web":
                                execution_result += tmp_execution_result
                                updated_messages[-1]["content"] = execution_result
                            else:
                                current_pos += 1
                                execution_result = tmp_execution_result
                                updated_messages.append(
                                    {
                                        "group": f"assistant_web_result",
                                        "pos": current_pos,
                                        "content": execution_result,
                                    }
                                )
                                current_status = "web"
                        elif "[/WEB]" in tmp_execution_result:
                            assert current_status == "web"
                            web_content = tmp_execution_result.split("[/WEB]")[0]
                            normal_content = tmp_execution_result.split("[/WEB]")[1]
                            execution_result += web_content
                            updated_messages[-1]["content"] = execution_result
                            current_pos += 1
                            execution_result = normal_content
                            updated_messages.append(
                                {
                                    "group": f"assistant_execution_result",
                                    "pos": current_pos,
                                    "content": execution_result,
                                }
                            )
                            current_status = "text"
                        else:
                            if current_status == "empty":
                                current_pos += 1
                                updated_messages.append(
                                    {
                                        "group": f"assistant_execution_result",
                                        "pos": current_pos,
                                        "content": tmp_execution_result,
                                    }
                                )
                                current_status = "text"
                                execution_result = tmp_execution_result
                            else:
                                print("I should not go there")
                                execution_result += tmp_execution_result
                                updated_messages[-1]["content"] = execution_result
                        yield json.dumps(updated_messages)

                else:
                    planning_decision = "direct"
                    planning_code = raw_planning_code
                    execution_result = ""
            if planning_decision == "code":
                cleaned_execution_result = execution_result.replace("stop", "")
                # pattern to remove the broswer ID and page ID
                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                new_cleaned = []
                for step in cleaned_execution_result.split("\n"):
                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                cleaned_execution_result = "\n".join(new_cleaned)
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
            elif planning_decision == "direct":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Directly answer the user query.",
                    }
                )
            elif planning_decision == "need_info":
                cleaned_execution_result = ""
                local_messages.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
                raw_data_logging.append(
                    {
                        "role": "system",
                        "name": "actor",
                        "content": f"Asking for more information from the user.",
                    }
                )
            current_pos += 1
            updated_messages.append(
                {
                    "group": f"assistant_final_output",
                    "pos": current_pos,
                    "content": "",
                }
            )
            raw_data_logging.append({"role": "assistant", "content": ""})
            final_output = ""
            # if os.environ['MODEL_NAME'].startswith("gpt") and planning_decision == "code":
            # openai gpt may still focus on the initial system message and 
            # generate something like callweb or direct answering
            local_messages = [local_messages[-1]]
            async for tmp_output in generate_stream(
                current_connection=self.LLM_connection, messages=local_messages
            ):
                final_output += tmp_output
                raw_data_logging[-1]["content"] += tmp_output
                if tmp_output not in ["<|im_continue|>"]:
                    updated_messages[-1]["content"] += tmp_output
                    yield json.dumps(updated_messages)
            if "<|im_continue|>" in final_output:
                continue_generation = True
                previous_raw_planning_code = final_output
                updated_messages[-1]["content"] = final_output.replace(
                    "<|im_continue|>", ""
                )
                updated_messages[-1]["group"] = "assistant_slow_thinking"
                yield json.dumps(updated_messages)
            else:
                continue_generation = False
            update_or_create_rawdata(
                session_id=CKStatus["session_id"],
                message_id=message_id,
                username=username,
                messages_in_train_format=raw_data_logging,
                updated_time=datetime.now().isoformat(),
            )

        # saving the local history to a history db.
        if ACTIVATE_HISTORY:
            target_history_ke = self._get_history_db_connection(CKStatus=CKStatus)
            task = asyncio.create_task(
                self._update_history_db(raw_data_logging, target_history_ke)
            )

    async def clean_up(self, CKStatus, username):
        current_id = f"{username}_{CKStatus['session_id']}"
        self.executor_manager.cleanup_executor(task_id=current_id)

    async def generate_for_demo(
        self, messages, CKStatus, username, message_id, mode="normal"
    ):
        """This is the main generation function for the demo

        Args:
            messages (_type_): user messages in the chatgpt format
            CKStatus (_type_): all meta information about the current CK conversation
        Returns:
            _type_: the final output.
        """

        print("messages:", messages)
        print("mode:", mode)
        if mode.startswith("concurrent"):
            feedback = None
            if mode == "concurrent_feedback":
                feedback = "gpt"
            async for updated_message in self.concurrent_callweb_v3(
                messages=messages,
                CKStatus=CKStatus,
                username=username,
                message_id=message_id,
                feedback=feedback
            ):
                yield updated_message
        else:
            async for updated_message in self.generate(
                messages=messages,
                CKStatus=CKStatus,
                username=username,
                message_id=message_id,
            ):
                yield updated_message

    def get_llm_evaluation_result(self, gold_output, generated_output):
        new_prompt = f"Please evaluate the following output:\n\nGold Output:\n{gold_output}\n\nGenerated Output:\n{generated_output}\n\n Only output 0 or 1: 0 means the generated output is not correct, 1 means the generated output is correct."
        result = self.gpt_connection.get_response(
            [{"role": "user", "content": new_prompt}]
        )
        print("gold_output:", gold_output)
        print("generated_output:", generated_output)
        print("automatic evaluation result:", result)
        return result

    async def evaluation_lite_by_batch(self, test_data):
        planning_success_count = 0
        execution_success_count = 0

        def handle_instance(tmp_instance):
            nonlocal planning_success_count, execution_success_count
            tmp_instance = json.loads(tmp_instance)
            test_messages = tmp_instance[:-3]
            gold_planning = tmp_instance[-3]["content"]
            messages_before_final_output = tmp_instance[:-1]
            gold_answer = tmp_instance[-1]["content"]

            raw_planning_code = self.LLM_connection.get_response(test_messages)
            planning_success_prediction = self.get_llm_evaluation_result(
                gold_output=gold_planning, generated_output=raw_planning_code
            )

            if "1" in planning_success_prediction:
                planning_success_count += 1
            final_output = self.LLM_connection.get_response(
                messages_before_final_output
            )
            execution_success_prediction = self.get_llm_evaluation_result(
                gold_output=gold_answer, generated_output=final_output
            )
            if "1" in execution_success_prediction:
                execution_success_count += 1

        with ThreadPoolExecutor(max_workers=len(test_data)) as executor:
            executor.map(handle_instance, test_data)

        return planning_success_count / len(test_data), execution_success_count / len(
            test_data
        )

    async def evaluation_by_batch(self, test_data):
        planning_success_count = 0
        execution_success_count = 0

        for tmp_instance in test_data:
            tmp_instance = json.loads(tmp_instance)
            test_messages = tmp_instance[:-3]
            role = tmp_instance[-2]["role"]
            user_query = tmp_instance[-3]["content"]
            gold_planning = tmp_instance[-3]
            gold_answer = tmp_instance[-2]

            raw_planning_code = self.LLM_connection.get_response(test_messages)
            test_messages.append({"role": "assistant", "content": raw_planning_code})
            planning_decision = "code"
            if "Direct answering" in raw_planning_code:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
            elif "Additional Info" in raw_planning_code:
                planning_decision = "need_info"
                planning_code = raw_planning_code
                execution_result = ""
            else:
                if "```" in raw_planning_code:
                    planning_decision = "code"
                    parts = raw_planning_code.split("```")
                    planning_code = parts[1] if len(parts) > 1 else ""
                    planning_code = planning_code.replace("python", "")
                    execution_result = ""
                    current_pos = 1
                    current_status = "empty"
                    async for tmp_execution_result in self.planning_execution(
                        planning_code=planning_code,
                        username="evaluation",
                        session_id="evaluation_" + uuid.uuid4().hex,
                        message_id="evaluation_" + uuid.uuid4().hex,
                    ):
                        if "[WEB]" in tmp_execution_result:
                            if current_status == "empty":
                                execution_result = tmp_execution_result
                                current_status = "web"
                            elif current_status == "web":
                                execution_result += tmp_execution_result
                            else:
                                current_pos += 1
                                execution_result = tmp_execution_result
                                current_status = "web"
                        elif "[/WEB]" in tmp_execution_result:
                            assert current_status == "web"
                            web_content = tmp_execution_result.split("[/WEB]")[0]
                            normal_content = tmp_execution_result.split("[/WEB]")[1]
                            execution_result += web_content
                            current_pos += 1
                            execution_result = normal_content
                            current_status = "text"
                        else:
                            if current_status == "empty":
                                execution_result = tmp_execution_result
                                current_status = "text"
                            else:
                                execution_result += tmp_execution_result
                else:
                    planning_decision = "direct"
                    planning_code = raw_planning_code
                    execution_result = ""
            if planning_decision == "code":
                cleaned_execution_result = execution_result.replace("stop", "")
                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                new_cleaned = []
                for step in cleaned_execution_result.split("\n"):
                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                cleaned_execution_result = "\n".join(new_cleaned)
                test_messages.append(
                    {
                        "role": "system",
                        "name": role,
                        "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                    }
                )
            elif planning_decision == "direct":
                cleaned_execution_result = ""
                test_messages.append(
                    {
                        "role": "system",
                        "name": role,
                        "content": f"Directly answer the user query.",
                    }
                )
            elif planning_decision == "need_info":
                cleaned_execution_result = ""
                test_messages.append(
                    {
                        "role": "system",
                        "name": role,
                        "content": f"Asking for more information from the user.",
                    }
                )
            final_output = self.LLM_connection.get_response(test_messages)
            planning_success_prediction = self.get_llm_evaluation_result(
                gold_output=gold_planning, generated_output=planning_code
            )
            execution_success_prediction = self.get_llm_evaluation_result(
                gold_output=gold_answer, generated_output=final_output
            )

            if "1" in planning_success_prediction:
                planning_success_count += 1
            if "1" in execution_success_prediction:
                execution_success_count += 1

        if len(test_data) == 0:
            return 0, 0
        else:
            return planning_success_count / len(
                test_data
            ), execution_success_count / len(test_data)

    async def evaluation(self, batch_size=4):

        test_data = get_annotations_for_evaluation()
        print("total test data:", len(test_data))
        test_data_by_tag = dict()
        for tmp_data in test_data:
            if tmp_data["tag"] not in test_data_by_tag:
                test_data_by_tag[tmp_data["tag"]] = []
            test_data_by_tag[tmp_data["tag"]].append(tmp_data["annotations"])

        # print('test_data_by_tag:', test_data_by_tag)
        tags = ["Normal", "DocAgent", "WebAgent", "PersonalHistory", "Mix"]
        results = []
        all_planning_success_count = 0
        all_execution_success_count = 0
        total_number = 0
        finished_number = 0
        for tmp_tag in tags:
            if tmp_tag in test_data_by_tag:
                total_number += len(test_data_by_tag[tmp_tag])

        for tmp_tag in tags:
            if tmp_tag not in test_data_by_tag:
                continue
            batches = [
                test_data_by_tag[tmp_tag][i : i + batch_size]
                for i in range(0, len(test_data_by_tag[tmp_tag]), batch_size)
            ]
            current_tag_planning_success_count = 0
            current_tag_execution_success_count = 0
            current_total_instance = 0
            for tmp_batch in batches:
                planning_success_rate, execution_success_rate = (
                    await self.evaluation_lite_by_batch(tmp_batch)
                )
                current_tag_execution_success_count += execution_success_rate * len(
                    tmp_batch
                )
                current_tag_planning_success_count += planning_success_rate * len(
                    tmp_batch
                )
                finished_number += len(tmp_batch)
                current_total_instance += len(tmp_batch)
                yield json.dumps(
                    {
                        "progress": int(finished_number * 100 / total_number),
                        "is_completed": False,
                        "result": [],
                    }
                ) + "\n"
            results.append(
                {
                    "name": tmp_tag,
                    "planning": current_tag_planning_success_count,
                    "execution": current_tag_execution_success_count,
                    "total": current_total_instance,
                }
            )
            all_planning_success_count += planning_success_rate * len(
                test_data_by_tag[tmp_tag]
            )
            all_execution_success_count += execution_success_rate * len(
                test_data_by_tag[tmp_tag]
            )
        results.append(
            {
                "name": "Overall",
                "planning": all_planning_success_count,
                "execution": all_execution_success_count,
                "total": total_number,
            }
        )
        yield json.dumps(
            {"progress": 100, "is_completed": True, "result": results}
        ) + "\n"

    async def inference_api(self, input_messages, full_info=False):
        current_messages = copy.deepcopy(input_messages)
        user_query = current_messages[-1]["content"]
        role = "actor"
        current_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_planning_code = self.LLM_connection.get_response(current_messages)
        # zero-shot llama
        if "<|python_tag|>" in raw_planning_code:
            raw_planning_code = raw_planning_code.replace("<|python_tag|>", "```python\n")
        current_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        tmp_message_id = "evaluation_" + uuid.uuid4().hex
        if "Direct answering" in raw_planning_code:
            planning_decision = "direct"
            planning_code = raw_planning_code
            execution_result = ""
        elif "Additional Info" in raw_planning_code:
            planning_decision = "need_info"
            planning_code = raw_planning_code
            execution_result = ""
        else:
            if "```" in raw_planning_code:
                planning_decision = "code"
                parts = raw_planning_code.split("```")
                planning_code = parts[1] if len(parts) > 1 else ""
                planning_code = planning_code.replace("python", "")
                execution_result = ""
                current_pos = 1
                current_status = "empty"
                async for tmp_execution_result in self.planning_execution(
                    planning_code=planning_code,
                    username="evaluation",
                    session_id=tmp_session_id,
                    message_id=tmp_message_id,
                ):
                    if "[WEB]" in tmp_execution_result:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "web"
                        elif current_status == "web":
                            execution_result += tmp_execution_result
                        else:
                            current_pos += 1
                            execution_result = tmp_execution_result
                            current_status = "web"
                    elif "[/WEB]" in tmp_execution_result:
                        assert current_status == "web"
                        web_content = tmp_execution_result.split("[/WEB]")[0]
                        normal_content = tmp_execution_result.split("[/WEB]")[1]
                        execution_result += web_content
                        current_pos += 1
                        execution_result = normal_content
                        current_status = "text"
                    else:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "text"
                        else:
                            execution_result += tmp_execution_result
            else:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Asking for more information from the user.",
                }
            )
        local_messages = copy.deepcopy(current_messages)
        # if os.environ['MODEL_NAME'].startswith("gpt") and planning_decision == "code":
        # openai gpt may still focus on the initial system message and 
        # generate something like callweb or direct answering
        local_messages = [local_messages[-1]]
        final_output = self.LLM_connection.get_response(local_messages)
        current_messages.append({"role": "assistant", "content": final_output})
        if full_info:
            other_logs = get_rawdata_by_session_id(session_id=tmp_session_id)
            # FTQ fix, don't return idx2element
            return {"messages": current_messages, "other_logs": other_logs}
        else:
            return current_messages
        

    async def inference_api_gaia(self, input_messages, full_info=False):
        """
            This function is a light modification to the inference on GAIA dataset.
            The call_web is conducted using a policy model, cognitive_kernel.LLM_connection
            And the answers are generated by cognitive_kernel.gpt_connection

        """
        current_messages = copy.deepcopy(input_messages)
        user_query = current_messages[-1]["content"]
        role = "actor"
        current_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_planning_code = self.LLM_connection.get_response(current_messages)
        # zero-shot llama
        if "<|python_tag|>" in raw_planning_code:
            raw_planning_code = raw_planning_code.replace("<|python_tag|>", "```python\n")
        current_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        tmp_message_id = "evaluation_" + uuid.uuid4().hex
        if "Direct answering" in raw_planning_code:
            planning_decision = "direct"
            planning_code = raw_planning_code
            execution_result = ""
        elif "Additional Info" in raw_planning_code:
            planning_decision = "need_info"
            planning_code = raw_planning_code
            execution_result = ""
        else:
            if "```" in raw_planning_code:
                planning_decision = "code"
                parts = raw_planning_code.split("```")
                planning_code = parts[1] if len(parts) > 1 else ""
                planning_code = planning_code.replace("python", "")
                execution_result = ""
                current_pos = 1
                current_status = "empty"
                async for tmp_execution_result in self.planning_execution(
                    planning_code=planning_code,
                    username="evaluation",
                    session_id=tmp_session_id,
                    message_id=tmp_message_id,
                ):
                    if "[WEB]" in tmp_execution_result:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "web"
                        elif current_status == "web":
                            execution_result += tmp_execution_result
                        else:
                            current_pos += 1
                            execution_result = tmp_execution_result
                            current_status = "web"
                    elif "[/WEB]" in tmp_execution_result:
                        assert current_status == "web"
                        web_content = tmp_execution_result.split("[/WEB]")[0]
                        normal_content = tmp_execution_result.split("[/WEB]")[1]
                        execution_result += web_content
                        current_pos += 1
                        execution_result = normal_content
                        current_status = "text"
                    else:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "text"
                        else:
                            execution_result += tmp_execution_result
            else:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""

        if planning_decision == "code":

            # if the action stopped:

            # if "```stop" in execution_result:

            print("execution_result:", execution_result)

            # if is_stopped(execution_result):
            if execution_result.strip().startswith("stop"):

                cleaned_execution_result = execution_result.replace("stop", "")
                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                new_cleaned = []
                for step in cleaned_execution_result.split("\n"):
                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                cleaned_execution_result = "\n".join(new_cleaned)
                
                original_query = user_query.split("Please interact with")[0].split('Now given a task:')[1].strip()
                search_query = raw_planning_code.split("query=")[1].split(", target_url")[0].strip()

                current_messages.append(
                    {
                        "role": "system",
                        "name": role,
                        "content": f"Answer original question: {original_query}, based on the retrieved information “{cleaned_execution_result}” by searching {search_query}",
                    }
                )
            else:
                original_query = user_query.split("Please interact with")[0].split('Now given a task:')[1].strip()
                current_messages.append(
                    {
                        "role": "system",
                        "name": role,
                        "content": f"Directly answer the user query. {original_query}",
                    }
                )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Directly answer the user query.",
                }
            )

        
        final_answer_messages = [copy.deepcopy(current_messages[-1])]
        print("final_answer_messages:", final_answer_messages)

        final_output = self.gpt_connection.get_response(final_answer_messages)
        print("final_output:", final_output)
        current_messages.append({"role": "assistant", "content": final_output})
        if full_info:
            other_logs = get_rawdata_by_session_id(session_id=tmp_session_id)
            # FTQ fix, don't return idx2element
            return {"messages": current_messages, "other_logs": other_logs}
        else:
            return current_messages
        
    async def inference_api_gaia_new_workflow(self, 
                                         input_messages, 
                                         full_info=False,
                                         world_model_type=None,
                                         world_model_search_depth=1,
                                         num_call_web_retry=1,
                                         max_outer_depth=7):
        """
            This function is a light modification to the inference on GAIA dataset.

            New workflow is applied.

            New parameters:
                num_call_web_retry (int): number of retrying for calling web.
                max_outer_depth (int): maximumx number of actions.
        """
        if not world_model_type in ["worldmodel", "webdreamer"]:
            world_model_type = None

        initial_messages = copy.deepcopy(input_messages)
        user_query = initial_messages[-1]["content"]
        role = "actor"
        raw_planning_code = self.gpt_connection.get_response(initial_messages)

        print("GAIA-debug::", raw_planning_code)

        # zero-shot llama-3.x
        if "<|python_tag|>" in raw_planning_code:
            raw_planning_code = raw_planning_code.replace("<|python_tag|>", "```python\n")

        # if searching in google, then change it to bing.com
        if '"www.google.com/"' in raw_planning_code or '"www.google.com"' in raw_planning_code:
            raw_planning_code = raw_planning_code.replace('www.google.com', 'www.bing.com')

        initial_messages.append({"role": "assistant", "content": raw_planning_code})
        current_messages = copy.deepcopy(initial_messages)
        current_messages[0]['content'] = GAIA_follow_up_prompt

        cnter = 0

        final_action = None

        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        # tmp_message_id = "evaluation_" + uuid.uuid4().hex

        execution_result_list = []

        while cnter < max_outer_depth:

            ## get what's between ``` ```

            def process_raw_planning_code(code):
                assert "```python" in code
                code = code.split("```python")[1].strip()
                assert "```" in code
                code = "```".join(code.split("```")[:-1]).strip()

                actions = [a.strip().strip(',') for a in code.split('\n')]

                return actions
            
            if not "```python" in raw_planning_code:
                break

            if 'www.google.com/"' in raw_planning_code or 'www.google.com"' in raw_planning_code:
                raw_planning_code = raw_planning_code.replace('www.google.com', 'www.bing.com')

            
            actions = process_raw_planning_code(raw_planning_code)

            return_answer = False

            tmp_execution_result_list = []


            for a_id, action in enumerate(actions):
                if 'Final answer' in action:
                    ## return the final answer
                    return_answer = True
                    final_action = action
                    break
                else:
                    query_pattern = r'query=["\']([^"\']+)["\']'
                    query = re.search(query_pattern, action).group(1)

                    if action.startswith('AskLLM'):
                        tmp_message_list_ask_llm = [{
                            "role": "system",
                            "content": "You are a helpful assistant. Please answer the query concisely"
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                        ]
                        execution_result = self.gpt_connection.get_response(tmp_message_list_ask_llm)
                        print('GAIA-debug::AskLLM', execution_result)
                    elif action.startswith('CallWeb'):


                        for retry_num in range(num_call_web_retry):

                            execution_result = ""
                            current_status = "empty"

                            
                            tmp_msg_id = f"depth_{cnter}_action_{a_id}_attempt_{retry_num}"

                            async for tmp_execution_result in self.planning_execution(
                                planning_code="\n"+action,
                                username="evaluation",
                                session_id=tmp_session_id,
                                message_id=tmp_msg_id,
                                world_model_search=world_model_type,
                                search_depth=world_model_search_depth,
                            ):
                                if "[WEB]" in tmp_execution_result:
                                    if current_status == "empty":
                                        execution_result = tmp_execution_result
                                        current_status = "web"
                                    elif current_status == "web":
                                        execution_result += tmp_execution_result
                                    else:
                                        execution_result = tmp_execution_result
                                        current_status = "web"
                                elif "[/WEB]" in tmp_execution_result:
                                    assert current_status == "web"
                                    web_content = tmp_execution_result.split("[/WEB]")[0]
                                    normal_content = tmp_execution_result.split("[/WEB]")[1]
                                    execution_result += web_content
                                    execution_result = normal_content
                                    current_status = "text"
                                else:
                                    if current_status == "empty":
                                        execution_result = tmp_execution_result
                                        current_status = "text"
                                    else:
                                        execution_result += tmp_execution_result
                                
                            if execution_result.strip().startswith("stop"):
                                # stopped
                                cleaned_execution_result = execution_result.replace("stop", "")
                                pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
                                new_cleaned = []
                                for step in cleaned_execution_result.split("\n"):
                                    new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
                                cleaned_execution_result = "\n".join(new_cleaned)
                                execution_result = cleaned_execution_result
                                break
                            else:
                                execution_result = "The web agent failed to retrieve useful information."
                            print('GAIA-debug::CallWeb', execution_result)
                    tmp_execution_result_list.append((query, execution_result))
            execution_result_list.append(tmp_execution_result_list)

            if return_answer:
                break

            # generate next planning code

            execusion_result_msg = "\n\n".join([f"The returned result for query `{query}` is {execution_result}." for query, execution_result in tmp_execution_result_list])
            
            current_messages.append({"role": "user", 
                                     "content": execusion_result_msg + "\nPlease generate the next action."})
            raw_planning_code = self.gpt_connection.get_response(current_messages)
            current_messages.append({"role": "assistant", 
                                       "content": raw_planning_code})
            print('GAIA-debug::next raw_planning_code', raw_planning_code)
            
            cnter += 1

        return current_messages, get_rawdata_by_session_id(session_id=tmp_session_id)
    
    
    async def inference_api_wm(self, input_messages, world_model_search=None, search_depth=1, full_info=False):
        print("in inference_api_wm, inference type",world_model_search )
        current_messages = copy.deepcopy(input_messages)
        user_query = current_messages[-1]["content"]
        role = "actor"
        current_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_planning_code = self.LLM_connection.get_response(current_messages)
        # zero-shot llama
        if "<|python_tag|>" in raw_planning_code:
            raw_planning_code = raw_planning_code.replace("<|python_tag|>", "```python\n")
        current_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        tmp_message_id = "evaluation_" + uuid.uuid4().hex
        if "Direct answering" in raw_planning_code:
            planning_decision = "direct"
            planning_code = raw_planning_code
            execution_result = ""
        elif "Additional Info" in raw_planning_code:
            planning_decision = "need_info"
            planning_code = raw_planning_code
            execution_result = ""
        else:
            if "```" in raw_planning_code:
                planning_decision = "code"
                parts = raw_planning_code.split("```")
                planning_code = parts[1] if len(parts) > 1 else ""
                planning_code = planning_code.replace("python", "")
                execution_result = ""
                current_pos = 1
                current_status = "empty"
                async for tmp_execution_result in self.planning_execution_wm(
                    planning_code=planning_code,
                    username="evaluation",
                    session_id=tmp_session_id,
                    message_id=tmp_message_id,
                    world_model_search=world_model_search,
                    search_depth=search_depth,
                ):
                    if "[WEB]" in tmp_execution_result:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "web"
                        elif current_status == "web":
                            execution_result += tmp_execution_result
                        else:
                            current_pos += 1
                            execution_result = tmp_execution_result
                            current_status = "web"
                    elif "[/WEB]" in tmp_execution_result:
                        assert current_status == "web"
                        web_content = tmp_execution_result.split("[/WEB]")[0]
                        normal_content = tmp_execution_result.split("[/WEB]")[1]
                        execution_result += web_content
                        current_pos += 1
                        execution_result = normal_content
                        current_status = "text"
                    else:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "text"
                        else:
                            execution_result += tmp_execution_result
            else:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Asking for more information from the user.",
                }
            )
        local_messages = copy.deepcopy(current_messages)
        # if os.environ['MODEL_NAME'].startswith("gpt") and planning_decision == "code":
        # openai gpt may still focus on the initial system message and 
        # generate something like callweb or direct answering
        local_messages = [local_messages[-1]]
        final_output = self.LLM_connection.get_response(local_messages)
        current_messages.append({"role": "assistant", "content": final_output})
        if full_info:
            other_logs = get_rawdata_by_session_id(session_id=tmp_session_id)
            # FTQ fix, don't return idx2element
            return {"messages": current_messages, "other_logs": other_logs}
        else:
            return current_messages

    async def inference_api_webcanvas(
        self, input_messages, max_steps, storage_state, full_info=False
    ):
        current_messages = copy.deepcopy(input_messages)
        user_query = current_messages[-1]["content"]
        role = "actor"
        current_messages.append(
            {
                "role": "system",
                "name": "actor",
                "content": "Please generate the slow thinking code",
            }
        )
        raw_planning_code = self.LLM_connection.get_response(current_messages)
        current_messages.append({"role": "assistant", "content": raw_planning_code})
        planning_decision = "code"
        tmp_session_id = "evaluation_" + uuid.uuid4().hex
        tmp_message_id = "evaluation_" + uuid.uuid4().hex
        if "Direct answering" in raw_planning_code:
            planning_decision = "direct"
            planning_code = raw_planning_code
            execution_result = ""
        elif "Additional Info" in raw_planning_code:
            planning_decision = "need_info"
            planning_code = raw_planning_code
            execution_result = ""
        else:
            if "```" in raw_planning_code:
                planning_decision = "code"
                parts = raw_planning_code.split("```")
                planning_code = parts[1] if len(parts) > 1 else ""
                planning_code = planning_code.replace("python", "")
                execution_result = ""
                current_pos = 1
                current_status = "empty"
                async for tmp_execution_result in self.planning_execution_webcanvas(
                    planning_code=planning_code,
                    username="evaluation",
                    session_id=tmp_session_id,
                    message_id=tmp_message_id,
                    max_steps=max_steps,
                    storage_state=storage_state,
                ):
                    if "[WEB]" in tmp_execution_result:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "web"
                        elif current_status == "web":
                            execution_result += tmp_execution_result
                        else:
                            current_pos += 1
                            execution_result = tmp_execution_result
                            current_status = "web"
                    elif "[/WEB]" in tmp_execution_result:
                        assert current_status == "web"
                        web_content = tmp_execution_result.split("[/WEB]")[0]
                        normal_content = tmp_execution_result.split("[/WEB]")[1]
                        execution_result += web_content
                        current_pos += 1
                        execution_result = normal_content
                        current_status = "text"
                    else:
                        if current_status == "empty":
                            execution_result = tmp_execution_result
                            current_status = "text"
                        else:
                            execution_result += tmp_execution_result
            else:
                planning_decision = "direct"
                planning_code = raw_planning_code
                execution_result = ""
        if planning_decision == "code":
            cleaned_execution_result = execution_result.replace("stop", "")
            pattern = r"^(?:\[[^\]]+\])\s*(\[[^\]]+\])\s*(\[[^\]]+\])"
            new_cleaned = []
            for step in cleaned_execution_result.split("\n"):
                new_cleaned.append(re.sub(pattern, r"[WEB]", step, count=1))
            cleaned_execution_result = "\n".join(new_cleaned)
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"based on the feedback “{cleaned_execution_result}” to answer query: {user_query}.",
                }
            )
        elif planning_decision == "direct":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Directly answer the user query.",
                }
            )
        elif planning_decision == "need_info":
            cleaned_execution_result = ""
            current_messages.append(
                {
                    "role": "system",
                    "name": role,
                    "content": f"Asking for more information from the user.",
                }
            )
        final_output = self.LLM_connection.get_response(current_messages)
        current_messages.append({"role": "assistant", "content": final_output})
        if full_info:
            other_logs = get_rawdata_by_session_id(session_id=tmp_session_id)
            return {"messages": current_messages, "other_logs": other_logs}
        else:
            return current_messages

    async def inference_api_history_retrieval(self, candidate_messages, target_query):
        tmp_history_db_name = f"history_tmp_{uuid.uuid4().hex}"
        tmp_config = KnowledgeEngineConfig(
            {
                "db_name": tmp_history_db_name,
                "db_type": "Sqlite",
                "db_path": f"/app/Database_local/{tmp_history_db_name}.db",
                "chunk_size": 1024,
                "neighbor_size": 10,
            }
        )
        target_ke_module = (
            self.knowledge_engine.get_local_knowledge_engine_module_fresh(
                config_file=tmp_config, module_name=tmp_history_db_name
            )
        )
        for tmp_round in candidate_messages:
            await self._update_history_db(tmp_round, target_ke_module)
        doc, metadata, _ = self.knowledge_engine.find_relevant_info_single_db(
            db_name=tmp_history_db_name,
            query=target_query,
            retrieval_mode="doc",
            soft_match_top_k=10,
            sim_threshold=-10000,
        )

        sorted_res = sorted(
            zip(doc, metadata), key=lambda x: json.loads(x[1])["timestamp"]
        )
        return [i[0] for i in sorted_res]

    async def inference_api_call_web(
        self,
        query,
        target_url,
        session_id,
        message_id,
        username,
        max_steps,
        storage_state,
        geo_location,
    ):
        results = []
        for tmp_response in call_web(
            llm_connection=self.LLM_connection,
            query=query,
            target_url=target_url,
            session_id=session_id,
            message_id=message_id,
            username=username,
            max_steps=max_steps,
            storage_state=storage_state,
            geo_location=geo_location,
            yield_full_message=True,
        ):
            results.append(tmp_response)
        return results
