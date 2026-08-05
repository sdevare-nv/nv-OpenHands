"""Losslessness tests for Codex-native result transport.

Codex intentionally applies its own output budgets before constructing an
observation.  These tests protect that already-produced body from the generic
30k ``CmdOutputObservation`` limit while it crosses the event and runtime
transport layers.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.agenthub.codex_agent.tool_output import (
    format_shell_output,
    truncate_function_output,
)
from openhands.core.schema import ObservationType
from openhands.events import EventSource
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.events.observation.commands import (
    MAX_CMD_OUTPUT_SIZE,
    CmdOutputMetadata,
)
from openhands.events.serialization import (
    event_from_dict,
    event_to_dict,
    observation_from_dict,
)
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)


_GENERIC_TRUNCATION_NOTICE = "\n[... Observation truncated due to length ...]\n"


def _codex_metadata(*, model: str = "gpt-5.6-sol") -> ToolCallMetadata:
    """Build realistic serializable metadata without making an LLM call."""
    model_response = {
        "id": "response-codex-transport",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-codex-transport",
                            "type": "function",
                            "function": {
                                "name": CODEX_SHELL_COMMAND_TOOL_NAME,
                                "arguments": '{"command":"produce-output"}',
                            },
                        }
                    ],
                }
            }
        ],
        "created": 0,
        "model": model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "total_tokens": 2,
        },
    }
    return ToolCallMetadata(
        function_name=CODEX_SHELL_COMMAND_TOOL_NAME,
        tool_call_id="call-codex-transport",
        model_response=model_response,
        total_calls_in_response=1,
        tool_result_format="codex",
    )


def _wire_round_trip(observation: CmdOutputObservation) -> CmdOutputObservation:
    """Round-trip through the same JSON-compatible event shape used on HTTP."""
    encoded = json.dumps(event_to_dict(observation))
    restored = event_from_dict(json.loads(encoded))
    assert isinstance(restored, CmdOutputObservation)
    return restored


def _expected_generic_content(content: str) -> str:
    if len(content) <= MAX_CMD_OUTPUT_SIZE:
        return content
    half = MAX_CMD_OUTPUT_SIZE // 2
    return content[:half] + _GENERIC_TRUNCATION_NOTICE + content[-half:]


def _cmd_observation(
    content: str,
    *,
    max_content_size: int | None = None,
) -> CmdOutputObservation:
    return CmdOutputObservation(
        content=content,
        command="produce-output",
        metadata=CmdOutputMetadata(
            exit_code=7,
            pid=314,
            working_dir="/workspace",
        ),
        max_content_size=max_content_size,
    )


def test_token_policy_shell_body_above_30k_survives_marked_wire_round_trip() -> None:
    """The 10k-token shell budget is about 40k bytes, above the legacy cap."""
    content = format_shell_output(
        "HEAD\n" + "x" * 79_990 + "\nTAIL",
        exit_code=7,
        duration_seconds=1.25,
        model_name="gpt-5.6-sol",
    )
    assert len(content) > MAX_CMD_OUTPUT_SIZE
    assert "tokens truncated" in content

    original = _cmd_observation(content)
    original.tool_call_metadata = _codex_metadata(model="gpt-5.6-sol")
    serialized = event_to_dict(original)
    restored = _wire_round_trip(original)

    assert serialized["content"] == content
    assert serialized["tool_call_metadata"]["tool_result_format"] == "codex"
    assert restored.content == content
    assert restored.command == original.command
    assert restored.metadata == original.metadata
    assert restored.tool_call_metadata is not None
    assert restored.tool_call_metadata.tool_result_format == "codex"
    assert restored.tool_call_metadata.function_name == CODEX_SHELL_COMMAND_TOOL_NAME


def test_byte_policy_unicode_shell_body_survives_wire_round_trip_exactly() -> None:
    """Byte-limited models can split beside multibyte Unicode boundaries."""
    content = format_shell_output(
        "HEAD\n" + "界" * 8_000 + "\nTAIL",
        exit_code=0,
        duration_seconds=0.0,
        model_name="gpt-5.2",
    )
    assert "chars truncated" in content
    assert content.encode("utf-8").startswith(b"Exit code: 0\n")

    original = _cmd_observation(content)
    original.tool_call_metadata = _codex_metadata(model="gpt-5.2")
    restored = _wire_round_trip(original)

    assert restored.content == content
    assert restored.content.encode("utf-8") == content.encode("utf-8")
    assert restored.tool_call_metadata is not None
    assert restored.tool_call_metadata.model_response.model == "gpt-5.2"


def test_token_policy_function_body_above_30k_survives_wire_round_trip() -> None:
    """Function history uses a 1.2x token budget (roughly 48k bytes)."""
    content = truncate_function_output(
        "BEGIN\n" + "z" * 95_000 + "\nEND",
        model_name="gpt-5.6-sol",
    )
    assert len(content) > MAX_CMD_OUTPUT_SIZE
    assert "tokens truncated" in content

    restored = _wire_round_trip(_cmd_observation(content))

    assert restored.content == content
    assert restored.content.count("tokens truncated") == 1


def test_observation_from_dict_treats_producer_content_as_authoritative() -> None:
    """Deserialization must neither truncate nor mutate a producer payload."""
    content = "BEGIN|" + "0123456789" * 4_000 + "|END"
    payload = {
        "observation": ObservationType.RUN,
        "content": content,
        "extras": {
            "command": "produce-output",
            "hidden": False,
            "metadata": {
                "exit_code": 9,
                "pid": 2718,
                "working_dir": "/workspace",
            },
            # Constructor-only transport input from an older/custom producer
            # must not be allowed to re-truncate authoritative wire content.
            "max_content_size": 8,
        },
    }
    original_payload = copy.deepcopy(payload)

    restored = observation_from_dict(payload)

    assert isinstance(restored, CmdOutputObservation)
    assert restored.content == content
    assert restored.exit_code == 9
    assert restored.command_id == 2718
    assert payload == original_payload


def test_deprecated_cmd_metadata_payload_is_lossless_and_not_mutated() -> None:
    """Legacy exit_code/command_id extras still migrate without body loss."""
    content = "legacy:" + "l" * (MAX_CMD_OUTPUT_SIZE + 101)
    payload = {
        "observation": ObservationType.RUN,
        "content": content,
        "extras": {
            "command": "legacy-command",
            "exit_code": 23,
            "command_id": 42,
        },
    }
    original_payload = copy.deepcopy(payload)

    restored = observation_from_dict(payload)

    assert isinstance(restored, CmdOutputObservation)
    assert restored.content == content
    assert restored.exit_code == 23
    assert restored.command_id == 42
    assert payload == original_payload


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("a" * (MAX_CMD_OUTPUT_SIZE - 1), id="one-under-limit"),
        pytest.param("b" * MAX_CMD_OUTPUT_SIZE, id="exact-limit"),
        pytest.param("c" * (MAX_CMD_OUTPUT_SIZE + 1), id="one-over-limit"),
        pytest.param("界" * (MAX_CMD_OUTPUT_SIZE + 1), id="multibyte-over-limit"),
    ],
)
def test_unmarked_source_constructor_applies_generic_limit_once_and_stays_stable(
    content: str,
) -> None:
    """Generic producers still bound output; transport does not truncate twice."""
    original = CmdOutputObservation(
        content=content,
        command="legacy-command",
        metadata=CmdOutputMetadata(exit_code=0),
    )
    expected = _expected_generic_content(content)

    first_round_trip = _wire_round_trip(original)
    second_round_trip = _wire_round_trip(first_round_trip)

    assert original.content == expected
    assert first_round_trip.content == expected
    assert second_round_trip.content == expected
    assert second_round_trip.content.count(_GENERIC_TRUNCATION_NOTICE) == (
        1 if len(content) > MAX_CMD_OUTPUT_SIZE else 0
    )
    assert first_round_trip.tool_call_metadata is None


class _TransportClient(ActionExecutionClient):
    async def connect(self) -> None:
        raise AssertionError("transport tests must not connect to a real runtime")

    @property
    def action_execution_server_url(self) -> str:
        return "http://action-server.invalid"


def _make_transport_client(payload: dict):
    client = object.__new__(_TransportClient)
    client.action_semaphore = threading.Semaphore(1)
    client.event_stream = MagicMock(name="event_stream")
    client._export_latest_git_provider_tokens = AsyncMock(
        name="export_latest_git_provider_tokens"
    )
    response = SimpleNamespace(is_closed=True, json=lambda: copy.deepcopy(payload))
    client._send_action_server_request = MagicMock(
        name="send_action_server_request",
        return_value=response,
    )
    return client


def test_action_client_and_runtime_base_preserve_body_then_attach_codex_marker() -> (
    None
):
    """Exercise HTTP decode plus Runtime._handle_action's metadata handoff."""
    content = format_shell_output(
        "FIRST\n" + "q" * 80_000 + "\nLAST",
        exit_code=0,
        duration_seconds=2.5,
        model_name="gpt-5.6-sol",
    )
    assert len(content) > MAX_CMD_OUTPUT_SIZE

    server_observation = _cmd_observation(content)
    client = _make_transport_client(event_to_dict(server_observation))
    action = CmdRunAction(command="produce-output")
    action.set_hard_timeout(17, blocking=False)
    action.tool_call_metadata = _codex_metadata(model="gpt-5.6-sol")

    async def call_direct(function, *args, **kwargs):
        return function(*args, **kwargs)

    # Runtime normally dispatches the synchronous client method in a worker.
    # Running it inline keeps this unit test deterministic while retaining the
    # complete decode-and-metadata-attachment path under test.
    with patch("openhands.runtime.base.call_sync_from_async", side_effect=call_direct):
        asyncio.run(client._handle_action(action))

    client._export_latest_git_provider_tokens.assert_awaited_once_with(action)
    request = client._send_action_server_request.call_args
    assert request.args == (
        "POST",
        "http://action-server.invalid/execute_action",
    )
    assert request.kwargs["timeout"] == 22
    serialized_action = request.kwargs["json"]["action"]
    assert serialized_action["tool_call_metadata"]["tool_result_format"] == "codex"
    assert serialized_action["tool_call_metadata"]["tool_call_id"] == (
        "call-codex-transport"
    )

    client.event_stream.add_event.assert_called_once()
    emitted, source = client.event_stream.add_event.call_args.args
    assert source is EventSource.AGENT
    assert isinstance(emitted, CmdOutputObservation)
    assert emitted.content == content
    assert emitted.tool_call_metadata is action.tool_call_metadata
    assert emitted.tool_call_metadata.tool_result_format == "codex"
    assert emitted.cause == action.id


def test_unmarked_action_client_path_keeps_source_applied_generic_truncation() -> None:
    """The transport fix must not enlarge output already bounded by its source."""
    raw_content = "legacy-start|" + "v" * 40_000 + "|legacy-end"
    server_observation = CmdOutputObservation(
        content=raw_content,
        command="legacy-command",
        metadata=CmdOutputMetadata(exit_code=3),
    )
    expected = _expected_generic_content(raw_content)
    client = _make_transport_client(event_to_dict(server_observation))
    action = CmdRunAction(command="legacy-command")
    action.set_hard_timeout(4, blocking=False)

    restored = client.send_action_for_execution(action)

    assert isinstance(restored, CmdOutputObservation)
    assert restored.content == expected
    assert restored.content.count(_GENERIC_TRUNCATION_NOTICE) == 1
    assert restored.tool_call_metadata is None
