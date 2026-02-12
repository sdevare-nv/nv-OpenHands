#!/usr/bin/env python3
"""Test script to verify provider-specific fields (prompt_token_ids, generation_token_ids, generation_log_probs)
are properly captured in MessageActions and saved to trajectory files.

This script tests the fix for ensuring that the last message (MessageAction without tool calls)
contains these provider-specific fields when saved.
"""

import json
import tempfile
from pathlib import Path

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import response_to_actions
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict, event_from_dict
from openhands.memory.conversation_memory import ConversationMemory
from openhands.core.message import Message


def test_message_action_provider_fields():
    """Test that MessageAction captures provider-specific fields from LLM response."""
    print("\n" + "="*80)
    print("TEST 1: MessageAction gets provider-specific fields")
    print("="*80)
    
    # Create a mock response that simulates an LLM response without tool calls
    response = ModelResponse(
        id='test-response-id-123',
        choices=[
            {
                'message': {
                    'content': 'This is the final response message',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    
    # Simulate provider-specific fields being set by the LLM class
    # This mimics what happens in openhands/llm/llm.py
    response._provider_specific_fields = {
        'prompt_token_ids': [1, 2, 3, 4, 5],
        'generation_token_ids': [10, 20, 30, 40],
        'generation_log_probs': [-0.1, -0.2, -0.15, -0.25],
    }
    
    # Convert response to actions
    actions = response_to_actions(response)
    
    # Verify we get a MessageAction
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    action = actions[0]
    assert isinstance(action, MessageAction), f"Expected MessageAction, got {type(action)}"
    
    # Verify the provider-specific fields are set on the action
    print(f"✓ Action type: {type(action).__name__}")
    print(f"✓ Action content: {action.content}")
    print(f"✓ Response ID: {action.response_id}")
    
    assert action.prompt_token_ids == [1, 2, 3, 4, 5], \
        f"Expected prompt_token_ids [1, 2, 3, 4, 5], got {action.prompt_token_ids}"
    print(f"✓ prompt_token_ids: {action.prompt_token_ids}")
    
    assert action.generation_token_ids == [10, 20, 30, 40], \
        f"Expected generation_token_ids [10, 20, 30, 40], got {action.generation_token_ids}"
    print(f"✓ generation_token_ids: {action.generation_token_ids}")
    
    assert action.generation_log_probs == [-0.1, -0.2, -0.15, -0.25], \
        f"Expected generation_log_probs [-0.1, -0.2, -0.15, -0.25], got {action.generation_log_probs}"
    print(f"✓ generation_log_probs: {action.generation_log_probs}")
    
    print("\n✅ TEST 1 PASSED: MessageAction correctly captures provider-specific fields\n")
    return action


def test_serialization(action):
    """Test that provider-specific fields are preserved during serialization."""
    print("="*80)
    print("TEST 2: Serialization and deserialization preserves fields")
    print("="*80)
    
    # Serialize the action to dict
    action_dict = event_to_dict(action)
    
    print(f"✓ Serialized action to dict")
    print(f"  Keys in dict: {list(action_dict.keys())}")
    
    # Verify the fields are in the serialized dict
    assert 'prompt_token_ids' in action_dict, "prompt_token_ids not in serialized dict"
    assert 'generation_token_ids' in action_dict, "generation_token_ids not in serialized dict"
    assert 'generation_log_probs' in action_dict, "generation_log_probs not in serialized dict"
    
    print(f"✓ prompt_token_ids in dict: {action_dict['prompt_token_ids']}")
    print(f"✓ generation_token_ids in dict: {action_dict['generation_token_ids']}")
    print(f"✓ generation_log_probs in dict: {action_dict['generation_log_probs']}")
    
    # Deserialize back to action
    restored_action = event_from_dict(action_dict)
    
    # Verify the fields are restored
    assert restored_action.prompt_token_ids == action.prompt_token_ids, \
        "prompt_token_ids not preserved after deserialization"
    assert restored_action.generation_token_ids == action.generation_token_ids, \
        "generation_token_ids not preserved after deserialization"
    assert restored_action.generation_log_probs == action.generation_log_probs, \
        "generation_log_probs not preserved after deserialization"
    
    print(f"✓ Deserialized action preserves all fields")
    
    print("\n✅ TEST 2 PASSED: Serialization preserves provider-specific fields\n")
    return action_dict


def test_jsonl_file_format(action_dict):
    """Test that fields are saved correctly in JSONL format (like in trajectories)."""
    print("="*80)
    print("TEST 3: JSONL file format (trajectory format)")
    print("="*80)
    
    # Write to a temporary JSONL file (simulating trajectory saving)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
        json.dump(action_dict, f)
        f.write('\n')
    
    print(f"✓ Written to temporary file: {temp_path}")
    
    # Read back from the file
    with open(temp_path, 'r') as f:
        loaded_dict = json.loads(f.readline())
    
    # Verify all fields are present
    assert 'prompt_token_ids' in loaded_dict, "prompt_token_ids not in JSONL"
    assert 'generation_token_ids' in loaded_dict, "generation_token_ids not in JSONL"
    assert 'generation_log_probs' in loaded_dict, "generation_log_probs not in JSONL"
    
    print(f"✓ Loaded from JSONL file")
    print(f"  prompt_token_ids: {loaded_dict['prompt_token_ids']}")
    print(f"  generation_token_ids: {loaded_dict['generation_token_ids']}")
    print(f"  generation_log_probs: {loaded_dict['generation_log_probs']}")
    
    # Clean up
    Path(temp_path).unlink()
    print(f"✓ Cleaned up temporary file")
    
    print("\n✅ TEST 3 PASSED: Fields correctly saved in JSONL format\n")


def test_conversation_memory_integration(action):
    """Test that ConversationMemory properly extracts fields from MessageAction."""
    print("="*80)
    print("TEST 4: ConversationMemory integration")
    print("="*80)
    
    # Create a ConversationMemory instance
    memory = ConversationMemory()
    
    # Process the action
    messages = memory._process_action(action, {}, vision_is_active=False)
    
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    message = messages[0]
    
    assert isinstance(message, Message), f"Expected Message, got {type(message)}"
    
    print(f"✓ ConversationMemory processed action into Message")
    print(f"  Message role: {message.role}")
    
    # Verify the provider-specific fields are in the Message
    assert message.prompt_token_ids == action.prompt_token_ids, \
        "prompt_token_ids not transferred to Message"
    print(f"✓ Message.prompt_token_ids: {message.prompt_token_ids}")
    
    assert message.generation_token_ids == action.generation_token_ids, \
        "generation_token_ids not transferred to Message"
    print(f"✓ Message.generation_token_ids: {message.generation_token_ids}")
    
    assert message.generation_log_probs == action.generation_log_probs, \
        "generation_log_probs not transferred to Message"
    print(f"✓ Message.generation_log_probs: {message.generation_log_probs}")
    
    # Test serialization of Message
    message_dict = message.serialize_model()
    print(f"\n✓ Message serialized to dict")
    
    # Check if the fields are in the serialized message
    assert 'prompt_token_ids' in message_dict, "prompt_token_ids not in serialized Message"
    assert 'generation_token_ids' in message_dict, "generation_token_ids not in serialized Message"
    assert 'generation_log_probs' in message_dict, "generation_log_probs not in serialized Message"
    
    print(f"✓ All fields present in serialized Message:")
    print(f"  prompt_token_ids: {message_dict['prompt_token_ids']}")
    print(f"  generation_token_ids: {message_dict['generation_token_ids']}")
    print(f"  generation_log_probs: {message_dict['generation_log_probs']}")
    
    print("\n✅ TEST 4 PASSED: ConversationMemory correctly transfers fields to Message\n")


def test_tool_call_actions():
    """Test that actions with tool calls also get provider-specific fields."""
    print("="*80)
    print("TEST 5: Actions with tool calls also get provider-specific fields")
    print("="*80)
    
    # Create a response with a tool call
    response = ModelResponse(
        id='test-tool-call-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'execute_bash',
                                'arguments': json.dumps({'command': 'ls -la', 'security_risk': 'LOW'}),
                            },
                            'id': 'tool-call-123',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )
    
    # Add provider-specific fields
    response._provider_specific_fields = {
        'prompt_token_ids': [100, 200, 300],
        'generation_token_ids': [400, 500],
        'generation_log_probs': [-0.5, -0.6],
    }
    
    actions = response_to_actions(response)
    
    assert len(actions) > 0, "Expected at least 1 action"
    action = actions[0]
    
    print(f"✓ Action type: {type(action).__name__}")
    
    # Verify provider-specific fields are set even on tool call actions
    assert action.prompt_token_ids == [100, 200, 300], \
        f"Expected prompt_token_ids on tool call action, got {action.prompt_token_ids}"
    print(f"✓ Tool call action has prompt_token_ids: {action.prompt_token_ids}")
    
    assert action.generation_token_ids == [400, 500], \
        f"Expected generation_token_ids on tool call action"
    print(f"✓ Tool call action has generation_token_ids: {action.generation_token_ids}")
    
    assert action.generation_log_probs == [-0.5, -0.6], \
        f"Expected generation_log_probs on tool call action"
    print(f"✓ Tool call action has generation_log_probs: {action.generation_log_probs}")
    
    print("\n✅ TEST 5 PASSED: Tool call actions also get provider-specific fields\n")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# TESTING PROVIDER-SPECIFIC FIELDS IN ACTIONS")
    print("#"*80)
    print("\nThis test verifies that prompt_token_ids, generation_token_ids, and")
    print("generation_log_probs are properly captured from LLM responses and saved")
    print("in trajectory files, including for MessageAction (final responses).\n")
    
    try:
        # Run tests
        action = test_message_action_provider_fields()
        action_dict = test_serialization(action)
        test_jsonl_file_format(action_dict)
        test_conversation_memory_integration(action)
        test_tool_call_actions()
        
        # Final summary
        print("#"*80)
        print("# ALL TESTS PASSED! ✅")
        print("#"*80)
        print("\nSummary:")
        print("  ✓ MessageActions capture provider-specific fields from LLM responses")
        print("  ✓ Fields are preserved during serialization/deserialization")
        print("  ✓ Fields are correctly saved in JSONL trajectory files")
        print("  ✓ ConversationMemory transfers fields to Message objects")
        print("  ✓ Tool call actions also get provider-specific fields")
        print("\nThe fix ensures that the last message in trajectories saved by run_infer.py")
        print("will contain prompt_token_ids, generation_token_ids, and generation_log_probs.")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

