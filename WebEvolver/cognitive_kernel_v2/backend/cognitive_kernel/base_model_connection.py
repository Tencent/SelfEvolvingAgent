import hmac
import os
import time
import uuid
import requests
import openai
import json
import asyncio
from typing import Optional

INFERENCE_SERVER_ENGINE = os.environ.get("INFERENCE_SERVER_ENGINE", "vLLM")


class BaseModelConnection:
    def __init__(self, ip="30.207.99.138:8000"):
        "Initialize the base model connection"
        self.ip = ip

    def format_message(message: str, role: str, name: str = None) -> str:
        if role == "system" and name:
            return f"<|im_start|>{role} name={name}\n{message}<|im_end|>"
        else:
            return f"<|im_start|>{role}\n{message}<|im_end|>"

    def _generate_query(self, messages: list()):
        """This function will generate the input messages to be the query we send to the base model.

        Args:
            messages (list): input messages
        """

        message_with_history = ""

        for tmp_round in messages:
            message_with_history += f"<|start_header_id|>{tmp_round['role']}<|end_header_id|>\n\n{tmp_round['content']}<|eot_id|>"
               
        message_with_history += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return message_with_history

    def get_reward_score(self, messages: list()):
        query = self._generate_query(messages[:-1])
        response = self._generate_query([messages[-1]])
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": query,
            "response": response,
            "temperature": 0,
            "max_tokens": 1,
        }
        url = "http://" + self.ip + "/ck_check_reward_score"
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json()["positive_logits"]
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            return None
        else:
            print("Failed:", response.status_code)
            return None
        
    def get_response_search_test(self, 
                                 messages: list(), 
                                 additional_prompt: str = '',
                                 logprobs: int = -1,
                                 additional_stop_token: str = '',
                                 temperature: float = 0.0, 
                                 num_return_seq: int = 1,
                                 top_p: float = 1.0,
                                 top_k: int = -1,
                                 seed: Optional[int] = None):
        query = self._generate_query(messages) + additional_prompt
        print("BaseModelConnection::get_respons_search_test", query)
        headers = {"Content-Type": "application/json"}
        stop_tokens = ["<|im_end|>"]
        if additional_stop_token != '':
            stop_tokens += [additional_stop_token]
        if INFERENCE_SERVER_ENGINE == "vLLM":
            data = {
                    "prompt": query,
                    "model": "ck",
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": 1024 * 4,
                    "stop": stop_tokens,
                    "n": num_return_seq,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": seed,
                }
            if logprobs > 1:
                data["logprobs"] = logprobs
            
            url = "http://" + self.ip + "/v1/completions"
            # url = "http://" + self.ip + "/generate"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=False
            )
            if response.status_code == 200:
                lines = response.text.split('\n')

                # Filter out non-JSON lines and parse each JSON object
                json_objects = []
                for line in lines:
                    if line.strip() and line.startswith('data: '):
                        try:
                            json_obj = json.loads(line[6:])  # Remove 'data: ' prefix
                            json_objects.append(json_obj)
                        except json.JSONDecodeError:
                            continue
                # Print the parsed JSON objects
                # for obj in json_objects:
                #     print(json.dumps(obj, indent=2))
                # print(json_objects[0])
                # text = "".join([json_objects[i]['choices'][0]['text'] for i in range(len(json_objects))])
                if logprobs > 1 or num_return_seq > 1:
                    return lines
                else:
                    return response.json()["choices"][0]["text"].replace('<|reserved_special_token_1|>', '<|im_continue|>')
        else:
            raise NotImplementedError

    def get_response(self, messages: list(), temperature: float = 0.0, do_print: bool = True):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        if do_print:
            print('get_response query:', query)
        if INFERENCE_SERVER_ENGINE == "tgi":
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": temperature,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                },
            }
            url = "http://" + self.ip + "/generate"
            response = requests.post(url, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                return (
                    response.json()["generated_text"]
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "")
                )
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                return "Failed:" + str(response.status_code)
        # elif INFERENCE_SERVER_ENGINE == "vLLM":
        #     data = {
        #         "prompt": query,
        #         "stream": False,
        #         "temperature": 0,
        #         "max_tokens": 1024,
        #         "stop_token_ids": [128258],
        #     }
        #     url = "http://" + self.ip + "/generate"
        #     response = requests.post(
        #         url, headers=headers, data=json.dumps(data), stream=True
        #     )
        #     if response.status_code == 200:
        #         return json.loads(response.text)["text"][0].replace(query, "")
        #     elif response.status_code == 422:
        #         print("Failed:", response.status_code)
        #         return "The input is too long. Please clean your history or try a shorter input."
        #     else:
        #         print("Failed:", response.status_code)
        #         return "Failed:" + str(response.status_code)
        elif INFERENCE_SERVER_ENGINE == "vLLM":
            data = {
                "prompt": query,
                "model": "ck",
                "temperature": temperature,
                "max_tokens": 2048,
                "stop": ["<|im_end|>"],
            }
            url = "http://" + self.ip + "/v1/completions"
            # url = "http://" + self.ip + "/generate"
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["text"].replace('<|reserved_special_token_1|>', '<|im_continue|>')
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed happened here:", response.status_code)
                return "Failed:" + str(response.status_code)
        else:
            raise NotImplementedError(
                "Inference server engine {} is not implemented".format(
                    INFERENCE_SERVER_ENGINE
                )
            )

    async def connection_testing(self, messages: list()):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}

        data = {
            "prompt": query,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1024,
            "stop_token_ids": [128258],
        }
        url = "http://" + self.ip + "/generate"
        response = requests.post(
            url, headers=headers, data=json.dumps(data), stream=True
        )
        if response.status_code == 200:
            yield json.loads(response.text)["text"][0].replace(query, "")
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            yield "The input is too long. Please clean your history or try a shorter input."
        else:
            print("Failed:", response.status_code)
            yield "Failed:" + str(response.status_code)

    async def streaming_connection_testing(self, messages: list()):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}

        data = {
            "prompt": query,
            "stream": True,
            "temperature": 0,
            "max_tokens": 1024,
            "stop_token_ids": [128258],
        }
        url = "http://" + self.ip + "/generate"
        start_time = time.time()
        response = requests.post(
            url, headers=headers, data=json.dumps(data), stream=True
        )
        print("time for the first token:", time.time() - start_time)
        if response.status_code == 200:
            previouse_text = query
            print("time for the first token:", time.time() - start_time)
            try:
                buffer = ""

                for chunk in response.iter_content(chunk_size=1):

                    if chunk.endswith(b"\0"):
                        buffer += chunk.decode("utf-8")[:-1]
                        try:
                            json_data = json.loads(buffer)
                            new_text = json_data["text"][0]
                            start_time = time.time()
                            yield new_text.replace(previouse_text, "")
                            previouse_text = new_text
                        except json.JSONDecodeError as e:
                            print(f"Parsing Error: {e}")
                        buffer = ""  # 重置缓冲区
                    else:
                        buffer += chunk.decode("utf-8")
            except json.JSONDecodeError as e:
                print(f"Parsing Error: {e}")
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            yield "The input is too long. Please clean your history or try a shorter input."
        else:
            print("Failed:", response.status_code)
            yield "Failed:" + str(response.status_code)

    async def ck_generate_stream(self, messages: list(), ck_mode=None, ck_k=8, ck_n=1):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        if ck_mode is not None:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": 0,
                "max_tokens": 1024,
                "ck_k": ck_k,
                "ck_n": ck_n,
                "length_penalty": 0,
                "ck_mode": ck_mode,
            }
        else:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": 0,
                "max_tokens": 256,
            }
        url = "http://" + self.ip + "/ck_generate"
        print(data)
        response = requests.post(
            url, headers=headers, data=json.dumps(data), stream=True
        )

        if response.status_code == 200:

            try:
                buffer = ""

                for chunk in response.iter_content(chunk_size=1):

                    if chunk.endswith(b"\0"):
                        buffer += chunk.decode("utf-8")[:-1]
                        try:
                            json_data = json.loads(buffer)
                            yield json_data["new_token"]
                        except json.JSONDecodeError as e:
                            print(f"parsing error: {e}")
                        buffer = ""
                    else:
                        buffer += chunk.decode("utf-8")
            except json.JSONDecodeError as e:
                print(f"parsing error: {e}")
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            yield "The input is too long. Please clean your history or try a shorter input."
        else:
            print("Failed:", response.status_code)
            yield "Failed:" + str(response.status_code)

    async def ck_generate_stream_eval(
        self, messages: list(), ck_mode=None, ck_k=8, ck_n=1, ck_d=0
    ):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        if ck_mode is not None:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": 0,
                "max_tokens": 256,
                "ck_k": ck_k,
                "ck_n": ck_n,
                "ck_d": ck_d,
                "ck_mode": ck_mode,
                "ck_length_penalty": 1,
            }
        else:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": 0,
                "max_tokens": 256,
            }
        print(data)
        url = "http://" + self.ip + "/ck_generate"
        response = requests.post(
            url, headers=headers, data=json.dumps(data), stream=True
        )
        if response.status_code == 200:
            try:
                buffer = ""

                for chunk in response.iter_content(chunk_size=1):

                    if chunk.endswith(b"\0"):
                        buffer += chunk.decode("utf-8")[:-1]
                        try:
                            json_data = json.loads(buffer)
                            yield json_data
                        except json.JSONDecodeError as e:
                            print(f"parsing error: {e}")
                        buffer = ""
                    else:
                        buffer += chunk.decode("utf-8")

            except json.JSONDecodeError as e:
                print(f"parsing error: {e}")
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            yield "The input is too long. Please clean your history or try a shorter input."
        else:
            print("Failed:", response.status_code)
            yield "Failed:" + str(response.status_code)

    async def get_response_stream(self, messages: list()):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}

        if INFERENCE_SERVER_ENGINE == "tgi":
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": 1024,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                },
            }
            url = "http://" + self.ip + "/generate_stream"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=True
            )

            counter = 0
            if response.status_code == 200:
                for line in response.iter_lines():
                    counter += 1
                    if line:
                        decoded_line = line.decode("utf-8")
                        decoded_line = json.loads(decoded_line[5:])
                        if not decoded_line["generated_text"]:
                            current_token = decoded_line["token"]["text"]
                            yield current_token

            elif response.status_code == 422:
                print("Failed:", response.status_code)
                yield "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                yield "Failed:" + str(response.status_code)
        elif INFERENCE_SERVER_ENGINE == "vLLM":
            print("lalala")
            data = {
                "prompt": query,
                "model": "ck",
                "stream": True,
                "temperature": 0,
                "max_tokens": 1024,
                "stop": ["<|im_end|>"],
            }
            url = "http://" + self.ip + "/v1/completions"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=True
            )
            if response.status_code == 200:
                start_time = time.time()
                buffer = b""
                all_chunks = list()
                for chunk in response.iter_content(chunk_size=1):
                    buffer += chunk
                    if buffer.endswith(b"\n\n"):
                        buffer = buffer.decode("utf-8")
                        if "[DONE]" not in buffer:
                            try:
                                json_data = json.loads(buffer[6:])
                                new_text = json_data["choices"][0]["text"].replace('<|reserved_special_token_1|>', '<|im_continue|>')
                                print(
                                    f"Received token: {new_text}, time: {time.time() - start_time}"
                                )
                                yield new_text
                                start_time = time.time()
                            except json.JSONDecodeError as e:
                                print(f"parsing error: {e}")
                        buffer = b""
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                yield "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                yield "Failed:" + str(response.status_code)
        else:
            raise NotImplementedError(
                "Inference server engine {} is not implemented".format(
                    INFERENCE_SERVER_ENGINE
                )
            )

    async def get_response_stream_check(self, messages: list()):
        positive_token_id = 32007
        negative_token_id = 32008
        top_tokens = 5
        max_trial = 2
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        current_trial = 0
        while current_trial < max_trial:
            current_trial += 1
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": 1024,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                    "top_n_tokens": top_tokens,
                },
            }
            url = "http://" + self.ip + "/generate_stream"
            response = requests.post(
                url, headers=headers, data=json.dumps(data), stream=True
            )
            counter = 0
            full_sentence = ""
            finished_generation = False
            if response.status_code == 200:
                for line in response.iter_lines():
                    counter += 1
                    if counter <= 3:
                        continue
                    if line:
                        decoded_line = line.decode("utf-8")
                        decoded_line = json.loads(decoded_line[5:])
                        # print(json.loads(decoded_line[5:]))
                        if not decoded_line["generated_text"]:
                            id2prob = dict()
                            for tmp_token in decoded_line["top_tokens"]:
                                id2prob[tmp_token["id"]] = tmp_token["logprob"]
                            if (
                                negative_token_id in id2prob
                                and id2prob[negative_token_id] > -0.1
                            ):
                                full_sentence = ""
                                yield full_sentence
                                break
                            else:
                                full_sentence += decoded_line["token"]["text"]
                                yield full_sentence
                        else:
                            finished_generation = True
                        # print(f"Received token: {current_token}")
                if finished_generation:
                    break
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                yield "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code, response.text)
                yield "Failed:" + str(response.status_code)


import os
from openai import OpenAI, AzureOpenAI


class ChatGPTConnection:
    def __init__(self, 
                 model_name="gpt-3.5-turbo",
                 ):
        self._model_name = model_name

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        print("base url", base_url, "api key", api_key)

        if os.environ.get("USE_AZURE", 'False') == "True":
            self.client = AzureOpenAI(
                azure_endpoint = base_url, 
                api_key=api_key,  
                api_version="2024-02-01"
            )
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )

    def get_response(self, messages: list()):
        try:
            completion = self.client.chat.completions.create(
                model=self._model_name, messages=messages
            )
            return completion.choices[0].message.content

        except:
            time.sleep(5)
            completion = self.client.chat.completions.create(
                model=self._model_name, messages=messages
            )
            return completion.choices[0].message.content

    def get_response_search_test(self, 
                                 messages: list(), 
                                 additional_prompt: str = '',
                                 logprobs: int = -1,
                                 additional_stop_token: str = '',
                                 temperature: float = 0.0, 
                                 num_return_seq: int = 1,
                                 top_p: float = 1.0,
                                 seed: Optional[int] = None,
                                 max_try_count=1):
        # query = self._generate_query(messages) + additional_prompt
        # print("BaseModelConnection::get_respons_search_test", query)
        # headers = {"Content-Type": "application/json"}
        # stop_tokens = ["<|im_end|>"]
        # if additional_stop_token != '':
            # stop_tokens += [additional_stop_token]

        
        additional_message = []
        if len(additional_prompt) > 0:
            additional_message.append({"role": "user", "content": additional_prompt})

        counter = 1
        while True:

            if logprobs > 0:
                logprobs = True
            else:
                logprobs = False
            
            try:
                # print("hahahha", messages + additional_message)
                completion = self.client.chat.completions.create(
                    model=self._model_name, 
                    messages=messages + additional_message,
                    temperature=temperature,
                    n=num_return_seq,
                    top_p=top_p,
                    seed=seed,
                    logprobs=logprobs
                )
                # print("hahahha", completion.choices[0].message.content)
                if logprobs:
                    return None
                    # not implemented yet
                else:
                    return completion.choices[0].message.content

            except:
                counter += 1
                if counter > max_try_count:
                    return None
                time.sleep(1)
                
default_LLM_connection = BaseModelConnection()
