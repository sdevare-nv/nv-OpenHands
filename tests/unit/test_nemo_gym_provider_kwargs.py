"""NeMo Gym requests must not contain LiteLLM AWS transport options."""

import asyncio
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming

_spec = importlib.util.spec_from_file_location(
    "nemo_gym_provider_kwargs_client",
    Path(__file__).parents[2] / "openhands/agenthub/nemo_gym_client.py",
)
client_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client_module)


@pytest.mark.parametrize(
    "aws_options",
    [
        {},
        {"aws_region_name": None},
        {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
        },
    ],
)
def test_aws_options_are_not_sent_to_gym(aws_options):
    kwargs = {
        "model": "test-model",
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_completion_tokens": 16,
        **aws_options,
    }
    original = copy.deepcopy(kwargs)
    client = client_module.NemoGymClient.__new__(client_module.NemoGymClient)
    client.llm = SimpleNamespace(_nemo_gym_llm_kwargs=kwargs)
    client.model_server_cookies = None
    # Stop at the HTTP boundary; this test exercises real payload construction.
    client.ng_server_client = SimpleNamespace(
        post=AsyncMock(side_effect=RuntimeError("captured"))
    )
    messages = [
        SimpleNamespace(model_dump=lambda: {"role": "user", "content": "hello"})
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(RuntimeError, match="captured"):
        asyncio.run(client._post_completion(messages, tools))
    payload = client.ng_server_client.post.call_args.kwargs["json"]
    assert payload == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "seed": 42,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_completion_tokens": 16,
        "tools": tools,
    }
    NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
    assert kwargs == original
