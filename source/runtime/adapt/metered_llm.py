"""Metered OpenAI-compatible LLM agent for the ReMindRAG reproduction.

Drop-in replacement for ReMindRag.llms.OpenaiAgent with three additions needed
to reproduce the paper's cost-efficiency metric (avg. LLM tokens per query):

1. Token metering: accumulates provider-reported prompt/completion tokens in a
   class-level counter shared by all instances (chunk/kg/generate/judge agents).
   Query-scope accounting: call MeteredOpenaiAgent.reset_usage() before a query
   and read get_usage() after.
2. GLM reasoning fallback: bigmodel GLM-5.3-Flash can return empty `content`
   (reasoning tokens consume the budget); fall back to `reasoning_content`.
3. Retry with backoff instead of the official silent busy-retry loop.

This file only ADDS behavior; the official library is untouched.
"""

import os
import inspect
import json
import threading
import time
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIError,
    OpenAI,
    RateLimitError,
)
import requests.exceptions
import socket
import urllib3.exceptions

from ReMindRag.llms.base import AgentBase


class MeteredOpenaiAgent(AgentBase):
    # Usage is shared by all agents in one worker thread, but isolated between
    # concurrent evaluation threads.  The runner uses 20 threads; a process-
    # global ledger makes reset_usage()/get_usage() race and assigns tokens to
    # the wrong question.
    _usage_lock = threading.Lock()
    _usage_template: Dict[str, float] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "failed_calls": 0,
    }
    _usage_local = threading.local()

    @classmethod
    def _ledger(cls) -> Dict[str, float]:
        ledger = getattr(cls._usage_local, "ledger", None)
        if ledger is None:
            ledger = dict(cls._usage_template)
            cls._usage_local.ledger = ledger
        return ledger

    def __init__(
        self,
        base_url: str,
        api_key: str,
        llm_model_name: str,
        time_out: int = None,
        max_retries: int = None,
        retry_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        disable_thinking: Optional[bool] = None,
        thinking_effort: Optional[str] = None,
    ):
        self.base_url = base_url
        self.llm_model_name = llm_model_name
        self.api_key = api_key
        # V15 sets both values centrally; direct callers keep safe legacy
        # defaults only when they do not run under the V15 launcher.
        time_out = int(time_out if time_out is not None else os.environ.get("REMIND_API_TIMEOUT", "180"))
        max_retries = int(max_retries if max_retries is not None else os.environ.get("REMIND_API_RETRIES", "8"))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        # instance-level thinking override: True/False forces this agent's calls
        # to send thinking.type:disabled regardless of the global env; None =
        # follow the global env (REMIND_DISABLE_THINKING). Enables mixed-agent
        # processes (e.g. thinking-ON builder + thinking-OFF traverser).
        self._disable_thinking = disable_thinking
        # instance-level effort override (bigmodel GLM): None = follow the
        # global env (REMIND_THINKING_EFFORT). Scoped so a GLM judge can run
        # effort=low in a process whose backbone agents send no thinking keys.
        self._thinking_effort = thinking_effort

        # The explicit retry loop below owns retries. Disable the SDK's hidden
        # retries so the configured count is the true maximum.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=time_out,
                             max_retries=0)

    # ---- usage ledger -------------------------------------------------
    @classmethod
    def _record(cls, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int):
        ledger = cls._ledger()
        ledger["calls"] += 1
        ledger["prompt_tokens"] += prompt_tokens
        ledger["completion_tokens"] += completion_tokens
        ledger["reasoning_tokens"] += reasoning_tokens

    @classmethod
    def _record_failure(cls):
        cls._ledger()["failed_calls"] += 1

    @classmethod
    def get_usage(cls) -> Dict[str, float]:
        return dict(cls._ledger())

    @classmethod
    def reset_usage(cls):
        cls._usage_local.ledger = {k: 0 for k in cls._usage_template}

    def _trace_call(self, status, **usage):
        path = os.environ.get('REMIND_CALL_TRACE')
        if not path:
            return
        stages = {'judge', '_v6_judge', 'judge_sufficient_information',
                  'find_next_node', 'get_chunk_summary', 'get_update_relation_and_chunk',
                  'run_single', 'generate_rag_ans'}
        stage = next((frame.function for frame in inspect.stack() if frame.function in stages),
                     'other_or_mc')
        row = dict(time=time.time(), status=status, stage=stage,
                   model=self.llm_model_name, **usage)
        with self._usage_lock:
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    # ---- AgentBase interface -------------------------------------------
    def generate_response(
        self,
        system_prompt: Optional[str],
        chat_history: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + chat_history
        else:
            messages = chat_history

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.llm_model_name,
                    messages=messages,
                    seed=123,
                    temperature=0,
                )
                # Platform-specific thinking knobs:
                # - DeepSeek official / bigmodel GLM: REMIND_DISABLE_THINKING=1
                #   sends {"thinking":{"type":"disabled"}} (OpenAI format, per
                #   DeepSeek V4 docs; effort default is high)
                # - bigmodel GLM-5.3 effort: REMIND_THINKING_EFFORT=low|high|max
                #   ({"thinking":{"type":"enabled","effort":...}})
                effort = (self._thinking_effort if self._thinking_effort is not None
                          else os.environ.get("REMIND_THINKING_EFFORT", "")).strip()
                if effort:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled", "effort": effort}}
                elif (self._disable_thinking if self._disable_thinking is not None else os.environ.get('REMIND_DISABLE_THINKING', '0') == '1'):
                    # DashScope's Qwen OpenAI-compatible endpoint uses
                    # enable_thinking=false; other supported providers use
                    # the OpenAI-style thinking.type knob.
                    if 'dashscope.aliyuncs.com' in self.base_url.lower():
                        kwargs["extra_body"] = {"enable_thinking": False}
                    else:
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = getattr(message, "content", None)
                if not content or not str(content).strip():
                    # GLM may spend the whole budget on reasoning; fall back.
                    content = getattr(message, "reasoning_content", None) or ""

                usage = getattr(response, "usage", None)
                prompt_tokens = completion_tokens = reasoning_tokens = 0
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    details = getattr(usage, "completion_tokens_details", None)
                    reasoning_tokens = (
                        getattr(details, "reasoning_tokens", 0) or 0
                        if details is not None
                        else 0
                    )
                self._record(prompt_tokens, completion_tokens, reasoning_tokens)
                self._trace_call('success', prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens,
                                 reasoning_tokens=reasoning_tokens)
                return content

            except (
                APIConnectionError,
                APIError,
                RateLimitError,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
                urllib3.exceptions.ProtocolError,
                urllib3.exceptions.HTTPError,
                ConnectionResetError,
                socket.timeout,
                TimeoutError,
            ) as e:
                last_error = e
                self._record_failure()
                self._trace_call('failure', error_type=type(e).__name__)
                if attempt < self.max_retries:
                    delay = min(self.retry_delay * (2**attempt), self.retry_max_delay)
                    time.sleep(delay)
                    continue
                raise Exception(
                    f"Failed after {self.max_retries} retries. Last error: {str(last_error)}"
                ) from last_error

