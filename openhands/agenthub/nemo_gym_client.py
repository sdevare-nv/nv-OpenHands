"""Shared NeMo Gym client for making model calls via the NeMo Gym server.

This module provides a reusable client that any agent can use to route
LLM completions through the NeMo Gym infrastructure instead of calling
litellm directly.
"""

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient
from nemo_gym.server_utils import get_response_json, raise_for_status

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.core.message import Message
    from openhands.llm.llm import LLM, ModelResponse


NEMO_GYM_SAMPLING_SEED_PROTOCOL_VERSION = 1


class SamplingSession:
    """Opt-in episode sampling; only whole-episode retries are supported."""

    def __init__(self, episode_seed: int) -> None:
        self.episode_seed = episode_seed
        self.next_call = 0
        self.busy = False

    @classmethod
    def from_environment(cls) -> "SamplingSession | None":
        seed = os.environ.get("NEMO_GYM_SAMPLING_SEED")
        if seed is None:
            return None
        if not seed.isascii() or not seed.isdecimal() or int(seed) >= 2**63:
            raise ValueError(
                "NEMO_GYM_SAMPLING_SEED must be a nonnegative signed 64-bit integer"
            )
        if os.environ.get("REPLAY_MESSAGES_PATH"):
            raise ValueError(
                "Seeded NeMo Gym sampling supports whole-episode replay only"
            )
        return cls(int(seed))

    @contextmanager
    def call(self, messages: list["Message"]) -> Iterator[int]:
        if self.busy:
            raise RuntimeError("Concurrent seeded NeMo Gym model calls are unsupported")
        if self.next_call == 0:
            roles = [message.model_dump().get("role") for message in messages]
            if any(role in ("assistant", "tool", "function") for role in roles):
                raise ValueError(
                    "Seeded NeMo Gym sampling requires a fresh episode without assistant/tool history"
                )
        self.busy = True
        try:
            # v1 is fully specified by these ASCII bytes, SHA256, big endian and mask.
            payload = (
                f"nemo-gym/model-call/v1:{self.episode_seed}:{self.next_call}".encode(
                    "ascii"
                )
            )
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
                (1 << 63) - 1
            )
            yield seed
        except BaseException:
            # Retrying an uncommitted logical call must reuse its seed.
            raise
        else:
            self.next_call += 1
        finally:
            self.busy = False


class NemoGymClient:
    """Client that proxies LLM completions through the NeMo Gym server.

    Set ``NEMO_GYM_SAMPLING_SEED`` to a nonnegative signed 64-bit episode seed
    to send a distinct reproducible seed with each sequential model call. Failed
    calls retain their seed for retry. Start from a fresh episode; the call ordinal
    is not restored from recorded messages. The model server must support seeds.

    Usage::

        # In agent __init__:
        self.nemo_gym_client = NemoGymClient(self.llm)

        # In agent step (async):
        response = await self.nemo_gym_client.model_call(messages, tools)
    """

    def __init__(self, llm: "LLM") -> None:
        self.ng_server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=get_global_config_dict(),
        )
        self.model_server_cookies = None
        self.llm = llm
        self._sampling = SamplingSession.from_environment()

    async def model_call(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
    ) -> "ModelResponse":
        """Make a model call via the NeMo Gym server, with automatic metrics tracking.

        Args:
            messages: Conversation messages (OpenHands Message objects).
            tools: Optional list of tool definitions for function calling.

        Returns:
            A validated ModelResponse from the server.
        """
        start_time = time.time()
        if self._sampling is None:
            response = await self._post_completion(messages, tools)
            self._update_model_call_time(start_time)
            return response
        with self._sampling.call(messages) as seed:
            response = await self._post_completion(messages, tools, sampling_seed=seed)
            self._update_model_call_time(start_time)
            return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_completion(
        self,
        messages: list["Message"],
        tools: "list[ChatCompletionToolParam] | None" = None,
        *,
        sampling_seed: int | None = None,
    ) -> "ModelResponse":
        from openhands.llm.llm import ModelResponse

        message_dicts = [m.model_dump() for m in messages]

        params: dict = {
            "messages": message_dicts,
            **self.llm._nemo_gym_llm_kwargs,
        }
        if sampling_seed is not None:
            params["seed"] = sampling_seed
        if tools:
            params["tools"] = tools

        fields_to_remove = [
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        ]
        last_occurrence_idx_seen = False
        for message in reversed(message_dicts):
            if last_occurrence_idx_seen:
                for field in fields_to_remove:
                    if field in message:
                        del message[field]
            elif all(field in message for field in fields_to_remove):
                last_occurrence_idx_seen = True

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
        completion_timestamp = datetime.now(timezone.utc).isoformat()
        response_id = model_response_json.get("id", "unknown")
        self.llm.metrics.add_response_latency(
            latency, response_id, timestamp=completion_timestamp
        )

        response_message_dict = model_response_json["choices"][0]["message"]
        usage = model_response_json.get("usage") or {}
        prompt_token_details = usage.get("prompt_tokens_details") or {}
        prompt_token_ids = response_message_dict.get("prompt_token_ids") or []
        generation_token_ids = response_message_dict.get("generation_token_ids") or []
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        self.llm.metrics.add_token_usage(
            prompt_tokens=(
                prompt_tokens if prompt_tokens is not None else len(prompt_token_ids)
            ),
            completion_tokens=(
                completion_tokens
                if completion_tokens is not None
                else len(generation_token_ids)
            ),
            cache_read_tokens=(
                prompt_token_details.get("cached_tokens", 0)
                if isinstance(prompt_token_details, dict)
                else 0
            ),
            cache_write_tokens=0,
            context_window=0,
            response_id=response_id,
        )
        self.model_server_cookies = model_response.cookies

        response: ModelResponse = ModelResponse.model_validate(model_response_json)

        provider_specific_fields: dict = {}
        if response_message_dict.get("prompt_token_ids"):
            provider_specific_fields = {
                "prompt_token_ids": response_message_dict["prompt_token_ids"],
                "generation_token_ids": response_message_dict["generation_token_ids"],
                "generation_log_probs": response_message_dict["generation_log_probs"],
            }
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
