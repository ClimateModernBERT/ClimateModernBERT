import openai
import os
import sys
import time
import logging
import diskcache as dc
import threading
import random
from openai import OpenAI

log = logging.getLogger()


# Reuse the same generation configuration style as in step_fact_check_thinking
GENERATION_CONFIGS = {
    'openai/gpt-oss-120b': {
        'extra_body': {"reasoning_effort": 'medium'},
    },
    "Qwen/Qwen3-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 20},
    },
    "microsoft/Phi-4-reasoning-plus": {
        "temperature": 0.8,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 50},
    },
    "gpt-5.1-2025-11-13": {"reasoning_effort": 'medium'},
    "gpt-5-mini-2025-08-07": {"reasoning_effort": 'medium'},
}


class LocalChat:
    """
    Local OpenAI-compatible chat client with persistent disk cache.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        base_url: str = "http://localhost:8000/v1",
        cache_path: str = os.path.expanduser("~") + "/.cache",
        system_prompt: str = None,
        generation_config: dict = None,
        cache_is_file: bool = False,
        temp_perturbation: float = 0.0,
    ):
        if cache_is_file:
            self.cache_path = cache_path
            cache_dir = os.path.dirname(cache_path) or "."
        else:
            cache_dir = cache_path
            self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")

        os.makedirs(cache_dir, exist_ok=True)

        self.base_url = base_url
        self.model = model
        self.generation_config = GENERATION_CONFIGS[model] if generation_config is None else generation_config
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=self.base_url, api_key='EMPTY')
        self.temp_perturbation = temp_perturbation

        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()

    def ask(self, message: str, sample_idx: int, **kwargs) -> str:
        reply, reasoning_content = self.cache.get((self.model, message, sample_idx), ('', ''))
        if reply == '':
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ]
            else:
                messages = [
                    {"role": "user", "content": message},
                ]
            chat = self._send_request(messages, sample_idx, self.generation_config)
            if chat is None:
                reply, reasoning_content = '', ''
            else:
                reasoning_content = getattr(chat.choices[0].message, "reasoning_content", "")
                reply = chat.choices[0].message.content

            with self._lock:
                self.cache[(self.model, message, sample_idx)] = (reply, reasoning_content)
        else:
            # pass
            print("Loaded from cache")

        if not reply:
            print("Reply is empty")
            print(f"Chat object: {chat}")

        low = reply.lower()
        if "please provide" in low or "to assist you" in low or "as an ai language model" in low:
            return "", ""

        return reply, reasoning_content

    def _send_request(self, messages, sample_idx, generation_config):
        # Create a copy to avoid mutating the original config
        config = generation_config.copy()
        config['seed'] = config['seed'] + sample_idx
        if 'temperature' in config and self.temp_perturbation > 0:
            config['temperature'] = config['temperature'] + random.uniform(-self.temp_perturbation, self.temp_perturbation)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **config,
            )
        except Exception as e:
            log.info(f"Request to OpenAI failed with exception: {e}.")
            return None

        return response


class DeepSeekChat:
    def __init__(
            self,
            cache_path: str,
            api_base: str | None = "https://api.deepseek.com/v1",
            model: str = "deepseek-reasoner",
            api_key: str | None = None,
            wait_times: tuple = (5, 10),
    ):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        if cache_path is None:
            cache_path = '~/.cache'
        self.cache_path = os.path.join(cache_path, "deepseek_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.wait_times = wait_times
        
        # Initialize cache with proper settings
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()
        print(f"Using {self.api_base}")
        self.client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)

    def ask(self, message: str, json_output=False) -> str:
        # First try to get from cache without lock
        reply = self.cache.get((self.model, message), '')
        
        if reply == '':
            if self.api_key is None:
                raise Exception("Cant ask DeepSeek without token.")
            messages = [
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages, json_output)
            if chat is None:
                reply = ""
            else:
                reply = chat.choices[0].message.content
            # Only lock when writing to cache
            with self._lock:
                self.cache[(self.model, message)] = reply

        if any(x in reply.lower() for x in ["please provide", "to assist you", "as an ai language model"]):
            return ""

        return reply

    def _send_request(self, messages, json_output=False):
        chat_args = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.6,
        }
        if json_output:
            chat_args['response_format'] = {'type': 'json_object'}
        for i in range(len(self.wait_times)):
            try:
                return self.client.chat.completions.create(**chat_args)
            except Exception as e:
                sleep_time = self.wait_times[i]
                log.info(
                    f"Request failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)
        try:
            return self.client.chat.completions.create(**chat_args)
        except Exception as e:
            sys.stderr.write(f'Error: {e}')
            return None

    def __del__(self):
        """Cleanup method to properly close the cache when the instance is destroyed."""
        if hasattr(self, 'cache'):
            self.cache.close()


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        base_url: str = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        generation_config: dict = {},
        system_prompt: str = "You are an intelligent assistant.",
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.base_url = base_url
        self.generation_config = GENERATION_CONFIGS[openai_model] if generation_config is None else generation_config
        self.system_prompt = system_prompt

    def ask(self, message: str, json_output=False) -> str:
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        openai_responses = dc.Cache(self.cache_path, **cache_settings)

        if (self.openai_model, message) in openai_responses:
            reply = openai_responses[(self.openai_model, message)]

        else:
            # Ask openai
            if openai.api_key is None:
                raise Exception(
                    "Cant ask openAI without token. "
                    "Please specify OPENAI_API_KEY in environment parameters."
                )
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ]
            else:
                messages = [
                    {"role": "user", "content": message},
                ]
            chat = self._send_request(messages)
            reply = chat.choices[0].message.content

            openai_responses[(self.openai_model, message)] = reply
            openai_responses.close()

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        for i in range(len(sleep_time_values)):
            try:
                return openai.OpenAI(base_url=self.base_url).chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    **self.generation_config,
                )
            except Exception as e:
                sleep_time = sleep_time_values[i]
                log.info(
                    f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return openai.OpenAI(base_url=self.base_url).chat.completions.create(
            model=self.openai_model,
            messages=messages,
            **self.generation_config,
        )