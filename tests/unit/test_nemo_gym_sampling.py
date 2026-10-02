"""Request-level sampling tests without loading unrelated browser agents."""

import asyncio
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

_spec = importlib.util.spec_from_file_location(
    "nemo_gym_sampling_client",
    Path(__file__).parents[2] / "openhands/agenthub/nemo_gym_client.py",
)
client_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client_module)


@pytest.fixture
def setup_client(monkeypatch):
    monkeypatch.delenv("REPLAY_MESSAGES_PATH", raising=False)
    monkeypatch.delenv("NEMO_GYM_SAMPLING_SEED", raising=False)
    monkeypatch.setattr(client_module, "ServerClient", Mock())
    monkeypatch.setattr(client_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(client_module, "raise_for_status", AsyncMock())
    payload = {
        "id": "test-response",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    monkeypatch.setattr(
        client_module, "get_response_json", AsyncMock(return_value=payload)
    )

    def create(seed=None):
        if seed is not None:
            monkeypatch.setenv("NEMO_GYM_SAMPLING_SEED", str(seed))
        llm = Mock(_nemo_gym_llm_kwargs={"model": "test"})
        client = client_module.NemoGymClient(llm)
        client.ng_server_client = SimpleNamespace(
            post=AsyncMock(return_value=SimpleNamespace(cookies={}))
        )
        client._log_completion = Mock()
        client._update_model_call_time = Mock()
        return client

    return create


def messages(role="user"):
    return [SimpleNamespace(model_dump=lambda: {"role": role, "content": "hello"})]


def expected(seed, ordinal):
    digest = hashlib.sha256(
        f"nemo-gym/model-call/v1:{seed}:{ordinal}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@pytest.mark.asyncio
async def test_payload_and_independent_episode_order(setup_client):
    first, second = setup_client(42), setup_client(42)
    await first.model_call(messages())
    await second.model_call(messages())
    await second.model_call(messages("assistant"))
    await first.model_call(messages("assistant"))
    for client in (first, second):
        assert [
            call.kwargs["json"]["seed"]
            for call in client.ng_server_client.post.call_args_list
        ] == [expected(42, 0), expected(42, 1)]


@pytest.mark.asyncio
async def test_unseeded_payload_unchanged(setup_client):
    client = setup_client()
    await client.model_call(messages("assistant"))
    assert "seed" not in client.ng_server_client.post.call_args.kwargs["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("failed request"), asyncio.CancelledError()]
)
async def test_failed_request_reuses_seed(setup_client, error):
    client = setup_client(7)
    client.ng_server_client.post.side_effect = [error, SimpleNamespace(cookies={})]
    with pytest.raises(type(error)):
        await client.model_call(messages())
    await client.model_call(messages())
    assert [
        call.kwargs["json"]["seed"]
        for call in client.ng_server_client.post.call_args_list
    ] == [expected(7, 0)] * 2


@pytest.mark.asyncio
async def test_invalid_response_reuses_seed(setup_client, monkeypatch):
    client = setup_client(7)
    valid = client_module.get_response_json.return_value
    monkeypatch.setattr(
        client_module,
        "get_response_json",
        AsyncMock(side_effect=[{"choices": []}, valid]),
    )
    with pytest.raises(IndexError):
        await client.model_call(messages())
    await client.model_call(messages())
    assert [
        call.kwargs["json"]["seed"]
        for call in client.ng_server_client.post.call_args_list
    ] == [expected(7, 0)] * 2


@pytest.mark.asyncio
async def test_concurrent_calls_rejected(setup_client):
    client = setup_client(7)
    entered, release = asyncio.Event(), asyncio.Event()

    async def post(**kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(cookies={})

    client.ng_server_client.post.side_effect = post
    first = asyncio.create_task(client.model_call(messages()))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="Concurrent"):
            await client.model_call(messages())
    finally:
        release.set()
        await first
    assert client.ng_server_client.post.call_count == 1


@pytest.mark.parametrize("seed", ["", "-1", "1.5", "true", str(2**63), "١"])
def test_invalid_environment_seed(setup_client, seed):
    with pytest.raises(ValueError, match="NEMO_GYM_SAMPLING_SEED"):
        setup_client(seed)


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_seed_bounds(setup_client, seed):
    assert setup_client(seed)._sampling.episode_seed == seed


def test_partial_replay_rejected(setup_client, monkeypatch):
    monkeypatch.setenv("REPLAY_MESSAGES_PATH", "recorded.json")
    with pytest.raises(ValueError, match="whole-episode"):
        setup_client(7)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "tool", "function"])
async def test_partial_history_rejected(setup_client, role):
    client = setup_client(7)
    with pytest.raises(ValueError, match="fresh episode"):
        await client.model_call(messages(role))
    client.ng_server_client.post.assert_not_called()
