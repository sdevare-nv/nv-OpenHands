"""Tests for the NeMo Gym chat request body built by the Gym client."""

from openhands.agenthub.nemo_gym_client import gym_chat_params


def _messages() -> list[dict]:
    return [
        {'role': 'user', 'content': 'task'},
        {
            'role': 'assistant',
            'content': 'look',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'execute_bash', 'arguments': '{}'},
                }
            ],
            'prompt_token_ids': [1, 2],
            'generation_token_ids': [3],
            'generation_log_probs': [-0.1],
        },
        {
            'role': 'tool',
            'content': 'out',
            'tool_call_id': 'call_1',
            'name': 'execute_bash',
        },
    ]


def test_tool_message_name_is_dropped():
    params = gym_chat_params(_messages(), {})
    tool_message = params['messages'][2]
    assert tool_message == {'role': 'tool', 'content': 'out', 'tool_call_id': 'call_1'}


def test_other_message_fields_are_kept():
    params = gym_chat_params(_messages(), {})
    assistant = params['messages'][1]
    assert assistant['tool_calls'][0]['function']['name'] == 'execute_bash'
    assert assistant['prompt_token_ids'] == [1, 2]
    assert assistant['generation_token_ids'] == [3]
    assert assistant['generation_log_probs'] == [-0.1]


def test_unset_and_litellm_only_kwargs_are_not_sent():
    llm_kwargs = {
        'model': 'policy',
        'temperature': 1.0,
        'top_p': 1.0,
        'seed': None,
        'max_completion_tokens': None,
        'aws_region_name': None,
        'aws_access_key_id': 'set-but-litellm-only',
        'aws_secret_access_key': 'set-but-litellm-only',
    }
    params = gym_chat_params(_messages(), llm_kwargs)
    assert set(params) == {'messages', 'model', 'temperature', 'top_p'}


def test_tools_only_when_given():
    tools = [{'type': 'function', 'function': {'name': 'execute_bash'}}]
    assert 'tools' not in gym_chat_params(_messages(), {})
    assert gym_chat_params(_messages(), {}, tools)['tools'] == tools
