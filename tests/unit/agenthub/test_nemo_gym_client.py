import copy
import json
from types import SimpleNamespace

from openhands.agenthub.nemo_gym_client import NemoGymClient
from openhands.core.message import Message, TextContent


def test_normalize_request_messages_accumulates_moe_history_on_anchor():
    messages = [
        {
            "role": "assistant",
            "content": "older",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
            "prompt_moe_topk_indices": {"source": "older"},
            "generation_moe_topk_indices": [{"source": "older-gen"}],
            "moe_metadata": {"source": "older-meta"},
        },
        {
            "role": "user",
            "content": "follow-up",
            "prompt_moe_topk_indices": {"source": "user-older"},
        },
        {
            "role": "assistant",
            "content": "anchor",
            "prompt_token_ids": [3],
            "generation_token_ids": [4],
            "generation_log_probs": [-0.2],
            "prompt_moe_topk_indices": [{"source": "anchor"}],
            "generation_moe_topk_indices": {"source": "anchor-gen"},
        },
        {
            "role": "assistant",
            "content": "newer-partial",
            "prompt_moe_topk_indices": {"source": "newer"},
        },
    ]

    actual = NemoGymClient._normalize_request_messages(copy.deepcopy(messages))

    assert "prompt_token_ids" not in actual[0]
    assert "generation_token_ids" not in actual[0]
    assert "generation_log_probs" not in actual[0]
    assert "prompt_moe_topk_indices" not in actual[0]
    assert "generation_moe_topk_indices" not in actual[0]
    assert "moe_metadata" not in actual[0]

    assert "prompt_moe_topk_indices" not in actual[1]

    assert actual[2]["prompt_token_ids"] == [3]
    assert actual[2]["generation_token_ids"] == [4]
    assert actual[2]["generation_log_probs"] == [-0.2]
    assert actual[2]["prompt_moe_topk_indices"] == [
        {"source": "older"},
        {"source": "user-older"},
        {"source": "anchor"},
    ]
    assert actual[2]["generation_moe_topk_indices"] == [
        {"source": "older-gen"},
        {"source": "anchor-gen"},
    ]
    assert actual[2]["moe_metadata"] == [{"source": "older-meta"}]

    assert actual[3]["prompt_moe_topk_indices"] == {"source": "newer"}


def test_normalize_request_messages_without_anchor_leaves_messages_unchanged():
    messages = [
        {
            "role": "assistant",
            "content": "older",
            "prompt_moe_topk_indices": {"source": "older"},
        },
        {
            "role": "assistant",
            "content": "newer",
            "generation_moe_topk_indices": [{"source": "newer"}],
        },
    ]

    expected = copy.deepcopy(messages)
    actual = NemoGymClient._normalize_request_messages(messages)

    assert actual == expected


def test_log_completion_writes_request_messages(tmp_path):
    client = NemoGymClient.__new__(NemoGymClient)
    client.llm = SimpleNamespace(
        config=SimpleNamespace(
            log_completions_folder=str(tmp_path),
            model="test-model",
        )
    )

    request_messages = [
        {
            "role": "assistant",
            "content": "anchor",
            "prompt_moe_topk_indices": [{"source": "older"}, {"source": "anchor"}],
        }
    ]

    client._log_completion(
        messages=[Message(role="user", content=[TextContent(text="hello")])],
        model_response_json={
            "choices": [{"message": {"role": "assistant", "content": "done"}}]
        },
        provider_specific_fields={},
        params={"messages": request_messages},
    )

    [log_file] = list(tmp_path.iterdir())
    logged = json.loads(log_file.read_text())
    assert logged["request_messages"] == request_messages

