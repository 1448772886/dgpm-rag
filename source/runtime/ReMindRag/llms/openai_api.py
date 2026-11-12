from .base import AgentBase
from openai import OpenAI
import time
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from openai import APIConnectionError, APIError, RateLimitError
import requests.exceptions
import urllib3.exceptions 
import socket

_usage_lock = threading.Lock()


def _token_usage_path():
    configured = os.environ.get("REMIND_TOKEN_USAGE_LOG")
    if configured:
        return Path(configured)
    try:
        args = sys.argv
        test_name = args[args.index("--test_name") + 1]
        title_index = args[args.index("--title_index") + 1]
    except (ValueError, IndexError):
        return None
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "eval" / "database" / test_name / title_index / "token_usage.jsonl"


def _record_token_usage(model, usage):
    if usage is None:
        return
    path = _token_usage_path()
    if path is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class OpenaiAgent(AgentBase):
    def __init__(self, base_url: str, api_key: str, llm_model_name: str, time_out: int = 120, max_retries: int = 3, retry_delay: float = 1.0):
        self.base_url = base_url
        self.llm_model_name = llm_model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=time_out
        )
        
    def generate_response(self, system_prompt: Optional[str], chat_history: List[Dict[str, Any]]) -> str:
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + chat_history
        else:
            messages = chat_history

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                request_args = dict(
                    model=self.llm_model_name,
                    messages=messages,
                    seed=123,
                    temperature=0
                )
                if self.llm_model_name.startswith('Qwen/'):
                    request_args['extra_body'] = {'enable_thinking': False}
                elif self.llm_model_name.startswith('deepseek-ai/'):
                    request_args['extra_body'] = {'thinking': {'type': 'disabled'}}
                response = self.client.chat.completions.create(**request_args)
                _record_token_usage(self.llm_model_name, getattr(response, "usage", None))
                return response.choices[0].message.content
                
            except (APIConnectionError, APIError, RateLimitError,
                   requests.exceptions.ConnectionError,
                   requests.exceptions.RequestException,
                   urllib3.exceptions.ProtocolError,
                   urllib3.exceptions.HTTPError,
                   ConnectionResetError,
                   socket.timeout,
                   TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    # time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise Exception(f"Failed after {self.max_retries} retries. Last error: {str(last_error)}") from last_error
