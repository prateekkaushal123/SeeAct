# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time

import backoff
import openai
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    InvalidRequestError
)

from line_profiler import LineProfiler
import base64
import io
from PIL import Image
import json

'''
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
'''
def encode_image(image_path, quality=80, resize_to=(480,344)):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(resize_to)
    byte_stream = io.BytesIO()
    image.save(byte_stream, format="JPEG", quality=quality)
    encode_image = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
    return encode_image



class Engine:
    def __init__(self) -> None:
        pass

    def tokenize(self, input):
        return self.tokenizer(input)


class OpenaiEngine(Engine):
    def __init__(
            self,
            api_key=None,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        Engine.__init__(self, **kwargs)

    def encode_image(self, image_path):
        with open(self, image_path, "rb") as image_file:
            #return base64.b64encode(image_file.read()).decode('utf-8')
            image = Image.open(image_file)
            byte_stream = io.BytesIO()
            image.save(byte_stream, format="JPEG", quality=80)
            return base64.b64decode(byte_stream.getvalue()).decode('utf-8')

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError, ServiceUnavailableError, InvalidRequestError),
    )
    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput__0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        openai.api_key = self.api_keys[self.current_key_idx]
        prompt0 = prompt[0]
        prompt1 = prompt[1]
        prompt2 = prompt[2]

        if turn_number == 0:
            base64_image = encode_image(image_path)
            # Assume one turn dialogue
            prompt1_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "low"},
                                                                 }]},
            ]
            response1 = openai.ChatCompletion.create(
                model=model if model else self.model,
                messages=prompt1_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            answer1 = [choice["message"]["content"] for choice in response1["choices"]][0]

            return answer1
        elif turn_number == 1:
            base64_image = encode_image(image_path)
            prompt2_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "low"}, }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput__0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, ]
            response2 = openai.ChatCompletion.create(
                model=model if model else self.model,
                messages=prompt2_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            return [choice["message"]["content"] for choice in response2["choices"]][0]
        
    def convert_completed_user_task_to_capability(self, prompt_text):
        openai.api_key = self.api_keys[self.current_key_idx]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",
                                                messages=[{"role": "user", "content": prompt_text}],max_tokens=1024)
        print("\nCapability string:\n")
        print(response)
        capability_string = response.choices[0].message.content;
        print(capability_string)
        return json.loads(capability_string)
    
    def get_playwright_action_from_history(self, prompt_text):
        openai.api_key = self.api_keys[self.current_key_idx]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",messages=[{"role": "user", "content": prompt_text}],max_tokens=1024)
        return response.choices[0].message.content
    
    def generate_css_selector(self, example):
        openai.api_key = self.api_keys[self.current_key_idx]
        prompt = f"Given the following example: '{example}', generate a valid CSS selector for the target element. send only the css selector."
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",
                                                messages=[{"role": "user", "content": prompt}],max_tokens=1024)
        css_selector = response.choices[0].text.strip()
        return css_selector
    
    @backoff.on_exception(
        backoff.expo, (APIError, RateLimitError, APIConnectionError, ServiceUnavailableError, InvalidRequestError),
        )
    
    def find_relevant_capability(self, user_input):
        openai.api_key = self.api_keys[self.current_key_idx]
        try:
            with open("task_history.json", "r") as f:
                task_history = json.load(f)
        except FileNotFoundError:
            print("task_history.json file not found.")
            return None
        
        prompt = f"User Input: {user_input}\n\n"
        prompt += "Based on the user input, find the most relevant capability from the following capabilities and their descriptions:\n\n"
        for capability_key, capability_data in task_history["capability"].items():
            prompt += f"Capability Key: {capability_key}\n"
            prompt += f"Description: {capability_data['description']}\n\n"
            
        prompt += "Response: Based on the user input, send the most relevant capability. If no capability found send None. Send only the key and no other text."
        print(f"\nCapability Prompt:\n{prompt}\n")
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",messages=[{"role": "user", "content": prompt}],max_tokens=1024)
        relevant_capability_key = response.choices[0].message.content
        print(f"\nRelavant capability key: {relevant_capability_key}\n")
        if relevant_capability_key in task_history["capability"]:
            return task_history["capability"][relevant_capability_key]
        else:
            print(f"No relevant capability found for the input: {user_input}")
            return None
    
    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError, ServiceUnavailableError, InvalidRequestError),) 
    
    def get_browser_operations(self, capability, user_input):
        openai.api_key = self.api_keys[self.current_key_idx]
        browser_operation_history = capability["browser_operation_history"]
        prompt = f"\n{browser_operation_history}\n\nreplace the above data to solve for customer input: {user_input}\n\ndont change the order.\ndont remove elements which were not changed.\n\n"
        print(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",
                                                messages=[{"role": "user", "content": prompt}],max_tokens=1024)
        list_data = response.choices[0].message.content.split("', '")
        list_data = [item.replace("'", "") for item in list_data]
        return list_data


class OpenaiEngine_MindAct(Engine):
    def __init__(
            self,
            api_key=None,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        Engine.__init__(self, **kwargs)

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError, ServiceUnavailableError),
    )
    def generate(self, prompt, max_new_tokens=50, temperature=0, model=None, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        openai.api_key = self.api_keys[self.current_key_idx]
        if isinstance(prompt, str):
            # Assume one turn dialogue
            prompt = [
                {"role": "user", "content": prompt},
            ]
        response = openai.ChatCompletion.create(
            model=model if model else self.model,
            messages=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [choice["message"]["content"] for choice in response["choices"]]
