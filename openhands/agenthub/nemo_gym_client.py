"""Shared NeMo Gym client for making model calls via the NeMo Gym server.

This module provides a reusable client that any agent can use to route
LLM completions through the NeMo Gym infrastructure instead of calling
litellm directly.
"""

import json
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any

from openhands.core.logger import openhands_logger as logger
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient
from nemo_gym.server_utils import get_response_json, raise_for_status

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.core.message import Message
    from openhands.llm.llm import LLM, ModelResponse


class NemoGymClient:
    """Client that proxies LLM completions through the NeMo Gym server.

    Usage::

        # In agent __init__:
        self.nemo_gym_client = NemoGymClient(self.llm)

        # In agent step (async):
        response = await self.nemo_gym_client.model_call(messages, tools)
    """

    _CORE_TOKEN_FIELD_KEYS = (
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
    )

    _MOE_FIELD_KEYS = (
        "prompt_moe_topk_indices",
        "generation_moe_topk_indices",
        "moe_metadata",
    )

    _PROVIDER_SPECIFIC_FIELD_KEYS = _CORE_TOKEN_FIELD_KEYS + _MOE_FIELD_KEYS

    def __init__(self, llm: "LLM") -> None:
        self.ng_server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=get_global_config_dict(),
        )
        self.model_server_cookies = None
        self.llm = llm

    async def model_call(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> "ModelResponse":
        """Make a model call via the NeMo Gym server, with automatic metrics tracking.

        Args:
            messages: Conversation messages (OpenHands Message objects).
            tools: Optional list of tool definitions for function calling.
            request_kwargs: Optional extra chat completion fields to forward.

        Returns:
            A validated ModelResponse from the server.
        """
        start_time = time.time()
        response = await self._post_completion(messages, tools, request_kwargs=request_kwargs)
        self._update_model_call_time(start_time)
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_moe_history_elems(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _normalize_request_messages(
        cls, message_dicts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        anchor_idx = -1
        for idx in range(len(message_dicts) - 1, -1, -1):
            if all(field in message_dicts[idx] for field in cls._CORE_TOKEN_FIELD_KEYS):
                anchor_idx = idx
                break

        if anchor_idx < 0:
            return message_dicts

        moe_history: dict[str, list[Any]] = {field: [] for field in cls._MOE_FIELD_KEYS}
        moe_field_seen = {field: False for field in cls._MOE_FIELD_KEYS}
        for idx in range(anchor_idx + 1):
            message = message_dicts[idx]
            is_anchor = idx == anchor_idx
            if not is_anchor:
                for field in cls._CORE_TOKEN_FIELD_KEYS:
                    message.pop(field, None)
            for field in cls._MOE_FIELD_KEYS:
                if field not in message:
                    continue
                moe_field_seen[field] = True
                moe_history[field].extend(cls._as_moe_history_elems(message[field]))
                if not is_anchor:
                    del message[field]

        anchor_message = message_dicts[anchor_idx]
        for field in cls._MOE_FIELD_KEYS:
            if moe_field_seen[field]:
                anchor_message[field] = moe_history[field]

        return message_dicts

    async def _post_completion(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> "ModelResponse":
        from openhands.llm.llm import ModelResponse

        message_dicts = self._normalize_request_messages(
            [m.model_dump() for m in messages]
        )

        params: dict = {
            "messages": message_dicts,
            **self.llm._nemo_gym_llm_kwargs,
        }
        if tools:
            params["tools"] = tools
        if request_kwargs:
            params.update({k: v for k, v in request_kwargs.items() if k not in ("messages", "tools")})

        # Measure per-call round-trip latency so it's surfaced in
        # `Metrics.response_latencies` (and therefore in the eval output.jsonl
        # via `get_metrics(state)`), mirroring the litellm path in
        # `openhands/llm/llm.py::_completion`.
        latency_start = time.perf_counter()
        model_response = await self.ng_server_client.post(
            server_name=os.getenv("NEMO_GYM_MODEL_SERVER_NAME"),
            url_path="/v1/chat/completions",
            json=params,
            cookies=self.model_server_cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)
        latency = time.perf_counter() - latency_start
        self.llm.metrics.add_response_latency(
            latency, model_response_json.get("id", "unknown")
        )
        self.model_server_cookies = model_response.cookies

        response: ModelResponse = ModelResponse.model_validate(model_response_json)

        response_message_dict = model_response_json["choices"][0]["message"]
        provider_specific_fields = {
            key: response_message_dict[key] for key in self._PROVIDER_SPECIFIC_FIELD_KEYS if key in response_message_dict
        }
        if provider_specific_fields:
            response._provider_specific_fields = provider_specific_fields

        self._log_completion(
            messages, model_response_json, provider_specific_fields, params
        )

        return response

    def _log_completion(
        self,
        messages: list["Message"],
        model_response_json: dict,
        provider_specific_fields: dict,
        params: dict,
    ) -> None:
        log_file = os.path.join(
            self.llm.config.log_completions_folder,
            f"{self.llm.config.model.replace('/', '__')}-{time.time()}.json",
        )
        _d = {
            "messages": [m.model_dump() for m in messages],
            "request_messages": params.get("messages"),
            "response": model_response_json,
            "provider_specific_fields": provider_specific_fields,
            "kwargs": {
                k: v for k, v in params.items() if k not in ("messages", "client")
            },
            "timestamp": time.time(),
        }

        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(log_file))
        with os.fdopen(temp_fd, "w") as f:
            f.write(json.dumps(_d))
        os.replace(temp_path, log_file)

    @staticmethod
    def _update_model_call_time(start_time: float) -> None:
        metrics_fpath = os.environ["NEMO_GYM_METRICS_FPATH"]
        with open(metrics_fpath) as f:
            existing_dict = json.loads(f.read())

        model_call_time_taken = existing_dict.get("total_model_call_time", 0.0)
        existing_dict["total_model_call_time"] = (
            model_call_time_taken + time.time() - start_time
        )

        with open(metrics_fpath, "w") as f:
            json.dump(existing_dict, f)
