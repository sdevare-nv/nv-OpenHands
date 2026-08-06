import asyncio
import copy
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Literal, Optional
import time
import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

import openhands.agenthub
from evaluation.benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from evaluation.benchmarks.swe_bench.patch_runtime import (
    capture_host_held_baseline,
    dispatch_authenticated_staging,
    dispatch_patch_completion,
    remove_nested_git_dirs,
)
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.benchmarks.swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_INSTALL,
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
    MAP_VERSION_TO_INSTALL,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    check_maximum_retries_exceeded,
    codeact_user_response,
    terminus_2_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import  CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    Observation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue
from evaluation.benchmarks.swe_bench.replay_utils import messages_to_replay_events  # noqa: E402

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_BROWSER_FOR_SWEBENCH = os.environ.get('ENABLE_BROWSER_FOR_SWEBENCH', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'
BenchMode = Literal['swe', 'swt', 'swt-ci']

# Multilingual support: language for SWE-bench_Multilingual docker images, instructions, and gitignore scripts
LANGUAGE = os.environ.get('LANGUAGE', 'python')
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', '')

# Global variable to track dataset type
DATASET_TYPE = 'SWE-bench'

# Maps instance_id -> (replay_events, initial_action) for replay-and-continue mode
REPLAY_DATA: dict[str, tuple[list, MessageAction]] = {}

MAX_RETRIES = 3

########################################
# START Custom profiling code
########################################

from io import StringIO
from pathlib import Path
from typing import Optional

import yappi


class Profiler:
    def __init__(self, name: str, base_profile_dir: Path) -> None:
        self.name = name
        self.base_profile_dir = base_profile_dir

        self.required_str = None

    def start(self) -> None:
        yappi.set_clock_type("CPU")
        yappi.start()
        print(f"🔍 Enabled profiling for {self.name}")

    def stop(self) -> None:
        print(f"🛑 Stopping profiler for {self.name}. Check {self.base_profile_dir} for the metrics!")
        yappi.stop()
        self.dump()

    def dump(self) -> None:
        self.base_profile_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.base_profile_dir / f"{self.name}.log"
        callgrind_path = self.base_profile_dir / f"{self.name}.callgrind"

        yappi.get_func_stats().save(callgrind_path, type="CALLGRIND")

        buffer = StringIO()
        yappi.get_func_stats().print_all(
            out=buffer,
            columns={
                0: ("name", 200),
                1: ("ncall", 10),
                2: ("tsub", 8),
                3: ("ttot", 8),
                4: ("tavg", 8),
            },
        )

        buffer.seek(0)
        res = ""
        past_header = False
        for line in buffer:
            if not past_header or (self.required_str and self.required_str in line):
                res += line

            if line.startswith("name"):
                past_header = True

        with open(log_path, "w") as f:
            f.write(res)


def update_metrics(update_dict: Dict[str, Any]) -> None:
    import os, json

    metrics_fpath = os.environ["NEMO_GYM_METRICS_FPATH"]
    with open(metrics_fpath) as f:
        existing_dict = json.loads(f.read())

    existing_dict = {k: v for k, v in existing_dict.items() if v is not None}
    update_dict = {k: v for k, v in update_dict.items() if v is not None}

    with open(metrics_fpath, "w") as f:
        json.dump(existing_dict | update_dict, f)


########################################
# END Custom profiling code
########################################


def set_dataset_type(dataset_name: str) -> str:
    """Set dataset type based on dataset name."""
    global DATASET_TYPE
    name_lower = dataset_name.lower()

    if 'nv-internal-1' in name_lower:
        DATASET_TYPE = 'nv-internal-1'
    elif 'swe-rebench-v2' in name_lower or 'swe-rebench_v2' in name_lower:
        DATASET_TYPE = 'SWE-rebench-V2'
    elif 'swe-gym' in name_lower:
        DATASET_TYPE = 'SWE-Gym'
    elif 'r2e-gym' in name_lower:
        DATASET_TYPE = 'R2E-Gym'
    elif 'swe-bench-live' in name_lower:
        DATASET_TYPE = 'SWE-bench-Live'
    elif 'swe-rebench' in name_lower:
        DATASET_TYPE = 'SWE-rebench'
    elif 'multimodal' in name_lower:
        DATASET_TYPE = 'Multimodal'
    elif 'multilingual' in name_lower:
        DATASET_TYPE = 'SWE-bench_Multilingual'
    elif 'swe-bench-ext' in name_lower:
        DATASET_TYPE = 'swe-bench-ext'
    elif 'deepswe' in name_lower:
        DATASET_TYPE = 'deepswe'
    else:
        DATASET_TYPE = 'SWE-bench'

    logger.info(f'Dataset type set to: {DATASET_TYPE}')
    os.environ['SWE_BENCH_DATASET_TYPE'] = DATASET_TYPE

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'Terminus2Agent': terminus_2_user_response,
}

AGENT_CLASSES_THAT_STOP_WITHOUT_TOOL_CALL = {'OpenCodeAgent', 'CodexAgent'}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    if DATASET_TYPE in ('SWE-bench-Live', 'deepswe'):
        # deepswe works in-place at /app and its instances carry no `version`
        # field, so key the (unused) workspace name off the instance_id.
        return instance.instance_id
    else:
        return f'{instance.repo}__{instance.version}'.replace('/', '__')


def _get_multilingual_instruction(instance: pd.Series, workspace_dir_name: str) -> str:
    """Return a language-specific instruction for SWE-bench Multilingual.

    Ported from multi_swe_bench/run_infer.py to support multilingual evaluation
    through the main swe_bench pipeline.
    """
    workspace_path = f'/workspace/{workspace_dir_name}'
    base_commit = instance['base_commit']
    problem = instance.problem_statement

    # Per-language instruction templates with language-specific reproduction commands
    instructions = {
        'python': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a python code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'java': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Java code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Java environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a Java class to reproduce the error and execute it by first compiling with `javac <classname>.java` and then running with `java <classname>` using the BashTool, to confirm the error\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script or class and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce class or script, and run them to make sure your fix handles these cases as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions or classes you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'go': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Go code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Go environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script or a function to reproduce the error and execute it with `go run <filename.go>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'c': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a C code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development C environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error by compiling your C code (for example, using `gcc <filename.c> -o <executable>`) and then running the executable using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'cpp': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a C++ code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development C++ environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create or adapt a small executable (e.g., a main file or a test driver) to reproduce the issue. Build and run it (for example, by using `g++ -o reproduce reproduce.cpp && ./reproduce` via the BashTool) to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'javascript': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Javascript code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Javascript environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `node <filename.js>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'typescript': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Typescript code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Typescript environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `ts-node <filename.ts>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'rust': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Rust code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Rust environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a reproduction script (or binary) that triggers the error and execute it with `cargo run --bin <filename>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'ruby': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a Ruby code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development Ruby environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `ruby <filename.rb>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
        'php': (
            f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
            f"I've uploaded a PHP code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
            f'<issue_description>\n{problem}\n</issue_description>\n\n'
            'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
            "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Also the development PHP environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
            'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
            'Follow these steps to resolve the issue:\n'
            '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
            '2. Create a script to reproduce the error and execute it with `php <filename.php>` using the BashTool, to confirm the error.\n'
            '3. Edit the sourcecode of the repo to resolve the issue.\n'
            '4. Rerun your reproduce script and confirm that the error is fixed!\n'
            '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
            f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
            '   - The issue you are fixing\n'
            '   - The files you modified\n'
            '   - The functions you changed\n'
            '   Make sure all these tests pass with your changes.\n'
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        ),
    }

    # Language-agnostic fallback
    default_instruction = (
        f'<uploaded_files>\n{workspace_path}\n</uploaded_files>\n'
        f"I've uploaded a code repository in the directory {workspace_path}. Consider the following issue description:\n\n"
        f'<issue_description>\n{problem}\n</issue_description>\n\n'
        'Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\n'
        "I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        "Also the development environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.\n"
        'Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the <issue_description> is satisfied.\n'
        'Follow these steps to resolve the issue:\n'
        '1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n'
        '2. Create a script to reproduce the error and execute it using the BashTool, to confirm the error.\n'
        '3. Edit the sourcecode of the repo to resolve the issue.\n'
        '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        '5. Think about edgecases, add comprehensive tests for them in your reproduce script, and run them to make sure your fix handles them as well.\n'
        f'6. Once you are done with the initial implementation, please carefully re-read the problem description and check the difference between the current code and the base commit {base_commit}. Do you think that the issue has been completely and comprehensively solved? Write tests to check the correctness of the solution, specifically focusing on tests that may point out any remaining problems that are not yet solved. Run all of the tests in the repo and check if any of them fail, and if they do fix the code. Repeat this process of carefully reading the problem description and current implementation, testing, and fixing any problems until you are confident that the current implementation is correct. Find and run any tests in the repo that are related to:\n'
        '   - The issue you are fixing\n'
        '   - The files you modified\n'
        '   - The functions you changed\n'
        '   Make sure all these tests pass with your changes.\n'
        "Your thinking should be thorough and so it's fine if it's very long.\n"
    )

    return instructions.get(LANGUAGE.lower(), default_instruction)


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    workspace_path = _get_workspace_path(instance, workspace_dir_name)

    custom_instruction_template_path = metadata.details.get('instruction_template_path')
    # For SWE-bench_Multilingual, use per-language instruction prompts
    if DATASET_TYPE == 'SWE-bench_Multilingual' and not (custom_instruction_template_path and os.path.isfile(custom_instruction_template_path)):
        instruction = _get_multilingual_instruction(instance, workspace_dir_name)
        if RUN_WITH_BROWSING:
            instruction += (
                '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
            )
        return MessageAction(content=instruction)

    mode = metadata.details['mode']
    llm_model = metadata.llm_config.model

    # Check for custom instruction template path (absolute path takes precedence)
    custom_instruction_template_path = metadata.details.get('instruction_template_path')

    if custom_instruction_template_path and os.path.isfile(custom_instruction_template_path):
        # Use custom instruction template from provided path
        prompts_dir = os.path.dirname(custom_instruction_template_path)
        template_name = os.path.basename(custom_instruction_template_path)
        logger.info(f'Using custom instruction template: {custom_instruction_template_path}')
    else:
        # Determine the template file based on mode and LLM
        if metadata.instruction_template_name:
            template_name = metadata.instruction_template_name
        elif mode.startswith('swt'):
            template_name = 'swt.j2'
        elif mode == 'swe':
            if 'gpt-4.1' in llm_model:
                template_name = 'swe_gpt4.j2'
            else:
                template_name = (
                    'swe_default.j2'  # Default for 'swe' mode (regular swe-bench)
                )
        else:
            # Fallback or error handling if mode is unexpected
            logger.error(f'Unexpected evaluation mode: {mode}. Falling back to default.')
            template_name = 'swe_default.j2'

        # Default prompts directory
        prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')

    logger.debug(f'Using instruction template file: {template_name}')
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        'instance': instance,
        'workspace_path': workspace_path,
        'metadata': metadata,  # Pass metadata if needed in templates
    }

    # Add specific context for swt-ci mode if needed
    if mode == 'swt-ci':
        context['test_instructions'] = (
            f'The following command can be used to run the tests: `{list(MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE[instance.repo].values())[0]}`. Make sure they fail in the expected way.\n'
        )
    else:
        context['test_instructions'] = ''  # Ensure it's defined for other modes

    # Render the instruction
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    if 'image_assets' in instance:
        assets = json.loads(instance['image_assets'])
        assert 'problem_statement' in assets, (
            'problem_statement is required in image_assets'
        )
        image_urls = assets['problem_statement']
        return MessageAction(content=instruction, image_urls=image_urls)
    return MessageAction(content=instruction)


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DEFAULT_DOCKER_IMAGE_PREFIX = os.environ.get(
    'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/'
)
logger.info(f'Default docker image prefix: {DEFAULT_DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(
    instance_id: str,
    swebench_official_image: bool = False,
) -> str:
    if swebench_official_image:
        # Official SWE-Bench image
        # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
        # SWE-bench-Live uses the same naming convention as SWE-Bench
        if DATASET_TYPE == 'SWE-bench-Live':
            docker_image_prefix = 'docker.io/starryzhang/'
        elif DATASET_TYPE == 'SWE-bench':
            docker_image_prefix = 'docker.io/swebench/'
        elif DATASET_TYPE == 'SWE-rebench':
            docker_image_prefix = 'docker.io/swerebench/'
        elif DATASET_TYPE in ['R2E-Gym', 'nv-internal-1', 'SWE-rebench-V2', 'swe-bench-ext', 'deepswe']:
            docker_image_prefix = 'UNAVAILABLE'
        elif DATASET_TYPE == 'SWE-bench_Multilingual':
            docker_image_prefix = 'docker.io/swebench/'
        else:
            pass
        if DATASET_TYPE in ('swe-bench-ext', 'deepswe'):
            # These ids have no `__` repo/name separator and the image is supplied
            # externally (SIF), so don't attempt to split.
            repo, name = instance_id, instance_id
        else:
            repo, name = instance_id.split('__')
        image_name = f'{docker_image_prefix.rstrip("/")}/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
        logger.debug(f'Using official SWE-Bench image: {image_name}')
        return image_name
    else:
        # OpenHands version of the image
        docker_image_prefix = DEFAULT_DOCKER_IMAGE_PREFIX
        image_name = 'sweb.eval.x86_64.' + instance_id
        image_name = image_name.replace(
            '__', '_s_'
        )  # to comply with docker image naming convention
        return (docker_image_prefix.rstrip('/') + '/' + image_name).lower()





def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    if DATASET_TYPE in ('SWE-rebench', 'SWE-rebench-V2'):
        # SWE-rebench instances have their own image_name field
        base_container_image = instance.get('image_name', '')
    elif DATASET_TYPE == 'SWE-bench_Multilingual' and LANGUAGE.lower() != 'python':
        # Multilingual non-python: use mswebench image naming convention
        container_name = instance.get('repo', '').lower().replace('/', '_m_')
        instance_id = instance.get('instance_id', '')
        tag_suffix = instance_id.split('-')[-1] if instance_id else ''
        base_container_image = f'mswebench/{container_name}:pr-{tag_suffix}'
    else:
        use_swebench_official_image = DATASET_TYPE != 'SWE-Gym'
        base_container_image = get_instance_docker_image(
            instance['instance_id'],
            swebench_official_image=use_swebench_official_image,
        )
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=ENABLE_BROWSER_FOR_SWEBENCH,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )

    # Optional override for the OpenHands file store location (where
    # `sessions/<id>/events/*.json` get written). By default
    # get_openhands_config_for_eval() plants this at `$CWD/.eval_sessions`,
    # which is noisy when running eval from the repo root. Set
    # EVAL_FILE_STORE_PATH=/some/path to redirect it anywhere you like.
    # Supports `~` expansion and relative paths (resolved against CWD).
    custom_file_store_path = os.environ.get('EVAL_FILE_STORE_PATH')
    if custom_file_store_path:
        custom_file_store_path = os.path.abspath(
            os.path.expanduser(custom_file_store_path)
        )
        os.makedirs(custom_file_store_path, exist_ok=True)
        config.file_store_path = custom_file_store_path
        logger.info(
            f'Overriding eval file_store_path via EVAL_FILE_STORE_PATH: '
            f'{config.file_store_path}'
        )

    # Optional toggle to disable the on-disk file store entirely. When set,
    # OpenHands swaps LocalFileStore for InMemoryFileStore, so no session
    # JSONs are written to disk. openhands/core/main.py already guards its
    # end-of-run save_to_session() call with `config.file_store != 'memory'`,
    # so the agent loop itself is unaffected — event subscribers (controller,
    # memory, condenser) keep working because they read/write through the
    # same (now in-memory) FileStore abstraction. Trade-off: you lose on-disk
    # session replay/recovery for the run; infer_logs/ and llm_completions/
    # are unaffected and `history` is still captured in output.jsonl.
    if os.environ.get('EVAL_DISABLE_FILE_STORE', 'true').lower() in (
        '1', 'true', 'yes',
    ):
        config.file_store = 'memory'
        logger.info(
            'EVAL_DISABLE_FILE_STORE set: using InMemoryFileStore '
            '(no session JSONs will be written to disk).'
        )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # get 'draft_editor' config if exists
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = (
        get_llms_for_routing_config()
    )  # Populate with LLMs for routing from config.toml file

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=ENABLE_BROWSER_FOR_SWEBENCH,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        model_routing=model_routing_config,
        system_prompt_filename=metadata.agent_config.system_prompt_filename
        if metadata.agent_config
        else 'system_prompt.j2',
        system_prompt_path=SYSTEM_PROMPT_PATH,
        system_prompt_long_horizon_path=SYSTEM_PROMPT_LONG_HORIZON_PATH,
        include_turns_remaining_reminder=INCLUDE_TURNS_REMAINING_REMINDER,
    )
    config.set_agent_config(agent_config)

    return config


def _run_harness_action(runtime: Runtime, action):
    """Run a harness-owned action, exempt from the anti-cheat blacklist.

    All commands issued directly by the harness (runtime setup, deep-reset,
    binary-patch scaffolding, completion) are trusted and must not be subject
    to the command blacklist that constrains the agent's rollout — otherwise
    harness git plumbing like `git merge-base --is-ancestor` is blocked and
    the runtime returns an ``ErrorObservation`` (no ``exit_code``), crashing
    the instance. Setting ``bypass_blacklist`` is safe because the agent's
    own actions are constructed by the function-call parser and can never set
    this flag. Non-command actions (e.g. FileReadAction) don't hit the
    blacklist, so we only tag CmdRunActions.
    """
    if isinstance(action, CmdRunAction):
        action.bypass_blacklist = True
    return runtime.run_action(action)


def _run_patch_export_command(
    runtime: Runtime,
    workspace_path: str,
    command: str,
    timeout: int,
) -> Observation:
    # Completion runs after the agent has controlled the persistent shell. A
    # fresh static session prevents shell-local aliases, functions, and PATH
    # changes from shadowing the Git/coreutils commands that authenticate and
    # filter the exported patch. An explicit cwd also makes the operation
    # independent of the persistent shell's final directory.
    action = CmdRunAction(
        command=command,
        is_static=True,
        cwd=workspace_path,
    )
    action.set_hard_timeout(timeout)
    logger.info(action, extra={'msg_type': 'ACTION'})
    observation = _run_harness_action(runtime, action)
    logger.info(observation, extra={'msg_type': 'OBSERVATION'})
    return observation


def _ensure_patch_export_success(
    observation: Observation, failure_message: str
) -> None:
    assert_and_raise(
        isinstance(observation, CmdOutputObservation)
        and observation.exit_code == 0,
        f'{failure_message}: {str(observation)}',
    )


def _report_suppressed_patch_cleanup(
    label: str, cleanup_error: BaseException
) -> None:
    logger.exception('%s: %r', label, cleanup_error)


def _interrupt_stuck_command(runtime: Runtime) -> None:
    """Free the tmux session when a previous command timed out but is still running.

    The runtime rejects new commands with "previous command still running"
    until the stuck process exits. We escalate via is_input=True (which
    bypasses that guard and writes keystrokes straight to the tmux pane)
    and probe with `true` between attempts to check shell responsiveness:

      1. C-c                 (SIGINT — kills most well-behaved processes)
      2. C-c x4              (some processes swallow the first SIGINT,
                              e.g. pytest's graceful-then-abort flow)
      3. C-z + `kill -9 %%`  (SIGTSTP suspends the foreground job and
                              returns the shell prompt; SIGKILL on `%%`
                              — bash's "current job", i.e. the one we
                              just suspended — terminates it for good)

    Best-effort; logs a warning if all escalations fail. Never raises.
    """
    def _shell_responsive() -> bool:
        # `true` is a no-op. Goes through the same "previous command still
        # running" guard as any other command, so if it returns 0 the shell
        # is genuinely free to accept new commands.
        probe = CmdRunAction(command='true')
        probe.set_hard_timeout(15)
        obs = _run_harness_action(runtime, probe)
        return isinstance(obs, CmdOutputObservation) and obs.exit_code == 0

    logger.info(
        'Previous command timed out and is still running; '
        'sending C-c (is_input=True) to interrupt...'
    )
    _run_harness_action(runtime, CmdRunAction(command='C-c', is_input=True))
    if _shell_responsive():
        return

    logger.info('First C-c did not free the shell; sending C-c x4...')
    for _ in range(4):
        _run_harness_action(runtime, CmdRunAction(command='C-c', is_input=True))
    if _shell_responsive():
        return

    logger.info(
        'Repeated C-c did not free the shell; sending C-z to suspend and '
        'kill -9 %% to terminate the suspended job...'
    )
    _run_harness_action(runtime, CmdRunAction(command='C-z', is_input=True))
    # After C-z the foreground job is stopped and the shell prompt returns,
    # so a normal (non-is_input) command goes through. `%%` is bash's
    # "current job" — always the most recently suspended/backgrounded one,
    # so it picks our suspended job regardless of any other jobs that might
    # be lingering in the session. Suppress kill's "no such job" error if
    # the suspend was actually a clean exit.
    _run_harness_action(runtime, CmdRunAction(command='kill -9 %% 2>/dev/null; true'))
    if _shell_responsive():
        return

    logger.warning(
        'Shell is still unresponsive after C-c, C-c x4, and C-z + kill -9 %%. '
        'Subsequent commands will likely fail; relying on top-level retry.'
    )



def _deep_reset_to_base_commit(runtime: Runtime, base_commit: str) -> None:
    # 2) Remove all remotes so remote-tracking refs can no longer resolve.
    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if obs.exit_code == -1:
        _interrupt_stuck_command(runtime)
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    # 3) Pin HEAD at base_commit and delete every local ref that could
    # reach commits past base_commit: branches, tags, stash, notes, and
    # any stray remote-tracking refs. Then move the original branch back
    # to base_commit so HEAD is symbolic again.
    deep_reset_cmd = (
        f'BASE=$(git rev-parse --verify {base_commit}^{{commit}}) && '
        'echo "Base commit: $BASE" && '

        # Capture the branch HEAD currently points at (before we detach below).
        # Falls back to `main` if HEAD is already detached on entry.
        'ORIG_BRANCH=$(git symbolic-ref --short -q HEAD || echo main) && '
        'echo "Original branch: $ORIG_BRANCH" && '

        # Detach so the current branch can be moved safely.
        'git checkout --detach "$BASE" && '

        'echo && '
        'echo "Resetting local branches that are descendants of base..." && '

        # Keep local branches, but reset any branch after BASE back to BASE.
        'git for-each-ref --format="%(refname)" refs/heads | while read -r ref; do '
        'tip="$(git rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)"; '
        '[ -z "$tip" ] && continue; '
        'if [ "$tip" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$tip"; then '
        'echo "reset $ref -> $BASE"; '
        'git update-ref "$ref" "$BASE"; '
        'else '
        'echo "keep $ref"; '
        'fi; '
        'done && '

        'echo && '
        'echo "Deleting tags, remote-tracking refs, stash, and other refs after base..." && '

        # Delete non-local-branch refs after BASE.
        'git for-each-ref --format="%(refname)" refs | while read -r ref; do '
        'case "$ref" in refs/heads/*) continue ;; esac; '
        'if git symbolic-ref -q "$ref" >/dev/null 2>&1; then '
        'echo "skip symbolic ref $ref"; '
        'continue; '
        'fi; '
        'tip="$(git rev-parse -q --verify "$ref^{commit}" 2>/dev/null || true)"; '
        '[ -z "$tip" ] && continue; '
        'if [ "$tip" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$tip"; then '
        'echo "delete $ref"; '
        'git update-ref -d "$ref"; '
        'else '
        'echo "keep $ref"; '
        'fi; '
        'done && '

        'echo && '
        'echo "Removing temporary Git operation refs..." && '

        'git_dir=$(git rev-parse --git-dir) && '
        'rm -f "$git_dir"/FETCH_HEAD "$git_dir"/ORIG_HEAD '
        '"$git_dir"/MERGE_HEAD "$git_dir"/CHERRY_PICK_HEAD '
        '"$git_dir"/REVERT_HEAD "$git_dir"/BISECT_HEAD '
        '"$git_dir"/AUTO_MERGE && '

        'echo && '
        'echo "Expiring reflogs..." && '
        'git reflog expire --expire=now --expire-unreachable=now --all && '

        'echo && '
        'echo "Pruning unreachable objects..." && '
        'git repack -ad && '
        'git prune --expire=now && '
        'git gc --prune=now && '

        'echo && '
        'echo "Checking out $ORIG_BRANCH at base..." && '
        'git checkout -B "$ORIG_BRANCH" "$BASE" && '

        'echo && '
        'echo "Done." && '
        'echo "Local branches were preserved." && '
        'echo "Tags/remotes/other refs after base were deleted."'
    )

    action = CmdRunAction(command=deep_reset_cmd)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code != 0:
        # If the per-ref iteration hit its 600 s timeout, the loop is still
        # running in tmux and would block the fallback. Kill it first.
        if obs.exit_code == -1:
            _interrupt_stuck_command(runtime)

        # Fallback: the careful per-ref pass timed out or otherwise failed.
        # This happens reliably on monorepos with thousands of refs (e.g.
        # DataDog/datadog-agent has 5,000+ release tags whose per-ref
        # iteration overhead exceeds the 600 s timeout). Drop the careful
        # keep-some-tags discrimination and batch-delete every non-current
        # ref in two `git update-ref --stdin` calls — microseconds total
        # regardless of ref count. Functionally equivalent for the agent
        # since it only needs HEAD pinned at base_commit.
        logger.info(
            f'deep_reset_cmd failed (exit_code={obs.exit_code}); '
            'falling back to batch-delete of all non-current refs.',
            extra={'msg_type': 'ACTION'},
        )
        nuclear_reset_cmd = (
            f'BASE=$(git rev-parse --verify {base_commit}^{{commit}}) && '
            'ORIG_BRANCH=$(git symbolic-ref --short -q HEAD || echo main) && '
            'echo "Base commit: $BASE, restoring branch: $ORIG_BRANCH" && '

            # Detach so we can freely delete every branch.
            'git checkout --detach "$BASE" && '

            # One git invocation deletes all tags/remotes/stash/notes.
            'echo "Batch-deleting all tags, remote-tracking refs, stash, notes..." && '
            'git for-each-ref --format="delete %(refname)" '
            'refs/tags refs/remotes refs/stash refs/notes 2>/dev/null '
            '| git update-ref --stdin && '

            # One git invocation deletes all local branches.
            'echo "Batch-deleting all local branches..." && '
            'git for-each-ref --format="delete %(refname)" refs/heads '
            '| git update-ref --stdin && '

            # Strip transient operation refs.
            'git_dir=$(git rev-parse --git-dir) && '
            'rm -f "$git_dir"/FETCH_HEAD "$git_dir"/ORIG_HEAD '
            '"$git_dir"/MERGE_HEAD "$git_dir"/CHERRY_PICK_HEAD '
            '"$git_dir"/REVERT_HEAD "$git_dir"/BISECT_HEAD '
            '"$git_dir"/AUTO_MERGE && '

            # Now nothing reaches past base; reflog + gc to reclaim space.
            'git reflog expire --expire=now --expire-unreachable=now --all && '
            'git repack -ad && '
            'git prune --expire=now && '
            'git gc --prune=now && '

            # Recreate the original branch at base and check it out.
            'git checkout -B "$ORIG_BRANCH" "$BASE" && '
            'echo "Nuclear cleanup done. Only $ORIG_BRANCH at base remains."'
        )
        action = CmdRunAction(command=nuclear_reset_cmd)
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if obs.exit_code == -1:
            _interrupt_stuck_command(runtime)

    # 4) Expire reflog so past HEAD positions cannot be walked.
    action = CmdRunAction(
        command=(
            'git reflog expire --expire=now --all && '
            'git reflog expire --expire-unreachable=now --all'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if obs.exit_code == -1:
        _interrupt_stuck_command(runtime)
    # assert_and_raise(obs.exit_code == 0, f'Failed to expire reflog: {str(obs)}')

    # 5) Prune dangling objects. After steps 1-4 every commit past
    # base_commit is unreachable; `git gc --prune=now` deletes them from
    # the object database so `git cat-file -p <sha>` and
    # `git fsck --lost-found` cannot resurrect them.
    action = CmdRunAction(command='git gc --prune=now')
    action.set_hard_timeout(900)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if obs.exit_code == -1:
        _interrupt_stuck_command(runtime)
    # assert_and_raise(obs.exit_code == 0, f'Failed to git gc --prune=now: {str(obs)}')


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
    metadata: EvalMetadata,
) -> bytes | None:
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    workspace_path = _get_workspace_path(instance, workspace_dir_name)
    obs: CmdOutputObservation

    initial_setup_cmd = f"""
echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \
echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && \
echo "alias git='git --no-pager'" >> ~/.bashrc && \
git config --global core.pager "" && \
git config --global diff.binary false && \
export USER=$(whoami) && \
echo "USER=$USER" && \
mkdir -p /swe_util/eval_data/instances && \
source ~/.bashrc
"""
    action = CmdRunAction(command=initial_setup_cmd.strip())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to run initial setup: {str(obs)}',
    )

    # inject the init script
    script_dir = os.path.dirname(__file__)

    swe_instance_json_name = 'swe-bench-instance.json'
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the desired file name within the temporary directory
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        # Write to the file with the desired name within the temporary directory
        with open(temp_file_path, 'w') as f:
            if not isinstance(instance, dict):
                json.dump([instance.to_dict()], f)
            else:
                json.dump([instance], f)

        # Copy the file to the desired location
        runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        if DATASET_TYPE == 'SWE-bench-Live':
            entry_script_path = 'instance_swe_entry_live.sh'
        elif DATASET_TYPE == 'SWE-rebench':
            entry_script_path = 'instance_swe_entry_rebench.sh'
        elif DATASET_TYPE == 'nv-internal-1':
            entry_script_path = 'instance_swe_entry_nv_internal.sh'
        elif DATASET_TYPE == 'swe-bench-ext':
            entry_script_path = 'instance_swe_entry_swe_ext.sh'
        elif DATASET_TYPE == 'SWE-rebench-V2':
            entry_script_path = 'instance_swe_entry_nv_internal.sh'
        elif DATASET_TYPE == 'R2E-Gym':
            entry_script_path = 'instance_swe_entry_r2e.sh'
        else:
            entry_script_path = 'instance_swe_entry.sh'
        runtime.copy_to(
            str(os.path.join(script_dir, f'scripts/setup/{entry_script_path}')),
            '/swe_util/',
        )

    # These dataset types operate directly in their repo dir, skipping the copy-to-workspace step.
    SKIP_ENTRY_SCRIPT_TYPES = ('nv-internal-1', 'SWE-rebench-V2', 'SWE-Gym', 'R2E-Gym', 'swe-bench-ext', 'deepswe')

    if DATASET_TYPE not in SKIP_ENTRY_SCRIPT_TYPES:
        action = CmdRunAction(command=f'source /swe_util/{entry_script_path}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to source /swe_util/{entry_script_path}: {str(obs)}',
        )
    elif DATASET_TYPE == 'SWE-Gym':
        # SWE-Gym: activate conda testbed environment
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "if [ -d /opt/miniconda3 ]; then "
                ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; "
                "fi; which python"
            )
        )
        action.set_hard_timeout(120)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to activate SWE-Gym conda environment: {str(obs)}",
        )
    elif DATASET_TYPE == 'R2E-Gym':
        # R2E-Gym: activate the venv bundled in the container
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "source /testbed/.venv/bin/activate; which python"
            )
        )
        action.set_hard_timeout(120)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to activate R2E-Gym venv: {str(obs)}",
        )
    elif instance.get("repo_language", instance.get("language", "")).lower() == "python":
        # nv-internal-1 / SWE-rebench-V2: use base python in the container
        action = CmdRunAction(
            command=(
                "deactivate >/dev/null 2>&1 || true; unset VIRTUAL_ENV; "
                "export PATH=/usr/bin:/bin:/usr/local/bin:$PATH; which python"
            )
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={"msg_type": "ACTION"})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        assert_and_raise(
            obs.exit_code == 0,
            f"Failed to deactivate and which python: {str(obs)}",
        )

    action = CmdRunAction(command=f'cd {workspace_path}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to {workspace_path}: {str(obs)}',
    )

    if DATASET_TYPE != "swe-bench-ext":
        # Deep-clean the git state so the agent cannot inspect commits past
        # the task's base_commit via local refs, tags, reflog, or dangling
        # objects. Remote access is blocked by removing remotes (below) and
        # by runtime command blacklist rules that forbid --all/--branches/
        # --remotes/--tags, explicit SHAs, origin/*, etc.
        base_commit = instance['base_commit']

        # 1) Discard uncommitted working-tree changes.
        action = CmdRunAction(command='git reset --hard')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

        _deep_reset_to_base_commit(runtime, base_commit)
    else:
        raw_base_commit = instance.get('base_commit', '')
        base_commit = (
            raw_base_commit.strip() if isinstance(raw_base_commit, str) else ''
        )

        # swe-bench-ext containers typically copy flat source into the
        # workspace with no Git history. Bootstrap a local repo when needed,
        # then snapshot the effective pristine HEAD as `swebench_baseline` so
        # we can diff the agent's changes against it in complete_runtime.
        #
        # The baseline_cmd below echoes a marker line so we can detect
        # whether the repo was freshly initialized. If it was, then
        # `instance['base_commit']` (an upstream SHA) does NOT exist in
        # the local repo and passing it to _deep_reset_to_base_commit
        # would fail rev-parse. In that case, the just-created HEAD is
        # already the correct baseline.
        baseline_cmd = (
            f'git config --global --add safe.directory {workspace_path} && '
            f'cd {workspace_path} && '
            # `-e` recognizes both normal `.git` directories and linked
            # worktree `.git` files without inheriting a parent repository.
            'if [ ! -e .git ]; then '
            '  git init -q && '
            "  git config user.email 'eval@openhands.local' && "
            "  git config user.name 'OpenHands Eval' && "
            '  git add -A && '
            "  git commit -q --allow-empty -m 'swe-bench-ext baseline' && "
            '  echo "__SWEBENCH_EXT_FRESH_INIT__"; '
            'else '
            '  echo "__SWEBENCH_EXT_PRE_EXISTING__"; '
            'fi'
        )
        action = CmdRunAction(command=baseline_cmd)
        action.set_hard_timeout(1800)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to set up swebench_baseline git repo: {str(obs)}',
        )
        freshly_initialized = '__SWEBENCH_EXT_FRESH_INIT__' in obs.content

        # Only deep-reset to instance['base_commit'] when the container
        # shipped a pre-existing git history that actually contains that
        # SHA. For freshly initialized repos, our synthetic baseline IS
        # the only commit, and the upstream SHA is unreachable.
        if base_commit and not freshly_initialized:
            _deep_reset_to_base_commit(runtime, base_commit)

        # Always create the tag after the optional deep reset. In particular,
        # some OTS images ship a pre-existing repository but an empty
        # base_commit; without this tag completion has no safe diff anchor.
        # `-f` also replaces a stale image-provided tag with the pristine HEAD.
        action = CmdRunAction(command='git tag -f swebench_baseline HEAD')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
            f'Failed to tag swebench_baseline at the effective HEAD: {str(obs)}',
        )

    if metadata.details['mode'] == 'swt-ci':
        # set up repo
        setup_commands = []
        if instance['repo'] in MAP_REPO_TO_INSTALL:
            setup_commands.append(MAP_REPO_TO_INSTALL[instance['repo']])

        # Run pre-install set up if provided
        install = MAP_VERSION_TO_INSTALL.get(instance['repo'], {}).get(
            instance['version'], []
        )
        if 'pre_install' in install:
            for pre_install in install['pre_install']:
                setup_commands.append(pre_install)

        if 'install' in install:
            setup_commands.append(install['install'])

        for command in setup_commands:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = _run_harness_action(runtime, action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if DATASET_TYPE not in ('Multimodal', 'SWE-bench-Live', 'nv-internal-1', 'SWE-rebench', 'SWE-rebench-V2', 'SWE-bench_Multilingual', 'swe-bench-ext'):
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets, SWE-bench-Live, nv-internal-1, and SWE-rebench are not using the testbed environment
        action = CmdRunAction(command='which python')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0 and 'testbed' in obs.content,
            f'Expected to find python interpreter from testbed, but got: {str(obs)}',
        )

    untracked_baseline: bytes | None = None
    if DATASET_TYPE != 'SWE-bench-Live':
        # Git represents an untracked embedded repository as only `dir/`.
        # Remove its metadata first so the baseline records every contained
        # file exactly and completion cannot expose hidden fixture contents.
        remove_nested_git_dirs(
            lambda command, timeout: _run_patch_export_command(
                runtime, workspace_path, command, timeout
            ),
            _ensure_patch_export_success,
        )
        # Capture pre-existing untracked paths before the agent starts, move the
        # NUL-delimited snapshot to evaluator-host memory, then remove the
        # runtime copy. Completion authenticates a fresh ephemeral upload.
        untracked_baseline = capture_host_held_baseline(
            runtime,
            workspace_path,
            lambda command, timeout: _run_patch_export_command(
                runtime, workspace_path, command, timeout
            ),
            _ensure_patch_export_success,
            report_cleanup_error=_report_suppressed_patch_cleanup,
        )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)
    return untracked_baseline


def _get_workspace_path(
    instance: pd.Series, workspace_dir_name: Optional[str] = None
) -> str:
    """Return the absolute workspace path expected by runtime commands."""
    if workspace_dir_name is None:
        workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    if DATASET_TYPE in ("nv-internal-1", "deepswe"):
        # nv-internal-1 and deepswe (Harbor) images clone the repo at /app and the
        # agent operates there directly instead of /workspace.
        return "/app"
    if DATASET_TYPE == "swe-bench-ext":
        return "/workspace/repo"
    if DATASET_TYPE == "SWE-rebench-V2":
        # SWE-rebench-V2 containers have repos directly at /{repo_name}
        repo = instance.get("repo", "")
        repo_name = repo.split("/")[1] if "/" in repo else repo
        return f"/{repo_name}"
    if DATASET_TYPE in ("SWE-Gym", "R2E-Gym"):
        return "/testbed"
    return f"/workspace/{workspace_dir_name}"


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
    untracked_baseline: bytes,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    workspace_path = _get_workspace_path(instance, workspace_dir_name)

    action = CmdRunAction(command=f'cd {workspace_path}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = _run_harness_action(runtime, action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running — try C-c (SIGINT).
        # is_input=True is required so the signal reaches the tmux pane
        # instead of being rejected by the "previous command still running" guard.
        logger.info('The previous command is still running, sending C-c (is_input=True)...')
        action = CmdRunAction(command='C-c', is_input=True)
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command=f'cd {workspace_path}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # C-c didn't work — try sending C-c twice (some processes like pytest
        # need two SIGINTs: the first triggers graceful shutdown, the second aborts).
        logger.info('C-c failed, sending C-c twice...')
        action = CmdRunAction(command='C-c', is_input=True)
        _run_harness_action(runtime, action)
        action = CmdRunAction(command='C-c', is_input=True)
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command=f'cd {workspace_path}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # Multiple C-c didn't work — try C-z (SIGTSTP) to suspend.
        # If the process suspends, the shell returns a prompt and we can proceed.
        logger.info('Multiple C-c failed, sending C-z (is_input=True) to suspend...')
        action = CmdRunAction(command='C-z', is_input=True)
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command=f'cd {workspace_path}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = _run_harness_action(runtime, action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to {workspace_path}: {str(obs)}',
    )

    obs = _run_patch_export_command(
        runtime,
        workspace_path,
        'git config --global core.pager ""',
        600,
    )
    _ensure_patch_export_success(obs, 'Failed to git config --global core.pager ""')

    # Check whether the workspace is actually a git repository. For
    # swe-bench-ext (and any other dataset where the container ships flat
    # source), initialize_runtime sets up a fresh repo and tags the baseline
    # commit as `swebench_baseline`. If neither a pre-existing .git nor that
    # baseline tag is present, there's nothing we can diff against and we
    # return an empty patch instead of crashing the whole instance.
    obs = _run_patch_export_command(
        runtime,
        workspace_path,
        'git rev-parse --is-inside-work-tree',
        600,
    )
    is_git_repo = (
        isinstance(obs, CmdOutputObservation)
        and obs.exit_code == 0
        and obs.content.strip().endswith('true')
    )
    if not is_git_repo:
        logger.warning(
            f'Workspace {workspace_path} is not a git repository; '
            'skipping git operations and returning an empty patch.'
        )
        logger.info('-' * 30)
        logger.info('END Runtime Completion Fn')
        logger.info('-' * 30)
        return {'git_patch': ''}

    # Prefer the swebench_baseline tag (set up by initialize_runtime for
    # datasets that don't ship a real git history, e.g. swe-bench-ext). Fall
    # back to instance['base_commit'] for normal datasets where the container
    # already has the repo checked out at that SHA.
    obs = _run_patch_export_command(
        runtime,
        workspace_path,
        'git rev-parse --verify refs/tags/swebench_baseline',
        600,
    )
    if (
        isinstance(obs, CmdOutputObservation)
        and obs.exit_code == 0
        and obs.content.strip()
    ):
        diff_base_ref = 'swebench_baseline'
        logger.info(
            f'Using swebench_baseline tag as diff reference '
            f'(resolved to {obs.content.strip()}).'
        )
    else:
        base_commit = instance.get('base_commit', '')
        assert_and_raise(
            isinstance(base_commit, str) and bool(base_commit.strip()),
            'Failed to determine patch diff base: swebench_baseline is missing '
            'and instance base_commit is empty.',
        )
        diff_base_ref = base_commit.strip()

    # Preserve the existing completion cleanup for nested repositories created
    # by the agent after the initialization baseline was captured.
    remove_nested_git_dirs(
        lambda command, timeout: _run_patch_export_command(
            runtime, workspace_path, command, timeout
        ),
        _ensure_patch_export_success,
    )

    # For SWE-bench_Multilingual: run language-specific gitignore script to exclude
    # compilation artifacts (e.g., .class, .o, target/) from the final patch.
    # Scripts sourced from MSWE-agent: https://github.com/multi-swe-bench/MSWE-agent
    if DATASET_TYPE == 'SWE-bench_Multilingual':
        gitignore_script_path = os.path.join(
            os.path.dirname(__file__), f'scripts/gitignores/{LANGUAGE}.sh'
        )
        if os.path.exists(gitignore_script_path):
            # Copy the script into the runtime and execute it
            runtime.copy_to(gitignore_script_path, '/tmp/')
            script_name = os.path.basename(gitignore_script_path)
            obs = _run_patch_export_command(
                runtime,
                workspace_path,
                f'chmod +x /tmp/{script_name} && /tmp/{script_name}',
                600,
            )
            _ensure_patch_export_success(
                obs,
                f'Failed to run gitignore script ({LANGUAGE}.sh)',
            )
        else:
            logger.info(
                f"No gitignore script found for language '{LANGUAGE}'. Skipping."
            )

    # Stage tracked changes and files created during the agent run, then remove
    # paths that were already untracked when the runtime was initialized.
    dispatch_authenticated_staging(
        runtime,
        workspace_path,
        diff_base_ref,
        untracked_baseline,
        lambda command, timeout: _run_patch_export_command(
            runtime, workspace_path, command, timeout
        ),
        _ensure_patch_export_success,
        report_cleanup_error=_report_suppressed_patch_cleanup,
    )

    # Remove binary files from git staging
    obs = _run_patch_export_command(
        runtime,
        workspace_path,
        remove_binary_files_from_git(),
        600,
    )
    _ensure_patch_export_success(obs, 'Failed to remove binary files')

    n_retries = 0
    git_patch = None
    # For multilingual, exclude .gitignore from the diff since we modified it
    # with the language-specific gitignore script
    gitignore_exclude = " ':!.gitignore'" if DATASET_TYPE == 'SWE-bench_Multilingual' else ''
    while n_retries < 5:
        obs = _run_patch_export_command(
            runtime,
            workspace_path,
            f'git diff --no-color --cached {diff_base_ref}{gitignore_exclude} > patch.diff',
            max(300 + 100 * n_retries, 600),
        )
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # Read the patch file
                action = FileReadAction(
                    path=os.path.join(workspace_path, 'patch.diff')
                )
                action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = _run_harness_action(runtime, action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                if isinstance(obs, FileReadObservation):
                    git_patch = obs.content
                    break
                elif isinstance(obs, ErrorObservation):
                    # Fall back to cat "patch.diff" to get the patch
                    assert 'File could not be decoded as utf-8' in obs.content
                    obs = _run_patch_export_command(
                        runtime,
                        workspace_path,
                        'cat patch.diff',
                        max(300 + 100 * n_retries, 600),
                    )
                    _ensure_patch_export_success(
                        obs,
                        'Failed to read non-UTF-8 patch with cat',
                    )
                    git_patch = obs.content
                    break
                else:
                    assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    # Remove binary diffs from the patch
    git_patch = remove_binary_diffs(git_patch)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def _has_existing_result(eval_output_dir: str, instance_id: str) -> tuple[bool, dict | None]:
    completions_dir = os.path.join(eval_output_dir, 'llm_completions', instance_id)
    has_completions = False
    if os.path.exists(completions_dir):
        json_files = [f for f in os.listdir(completions_dir) if f.endswith('.json')]
        has_completions = len(json_files) > 0

    if not has_completions:
        return False, None

    output_file = os.path.join(eval_output_dir, 'output.jsonl')
    existing_result = None
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if result.get('instance_id') == instance_id:
                            git_patch = result.get('test_result', {}).get('git_patch', '')
                            if git_patch and git_patch.strip():
                                existing_result = result
                                break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f'Error reading output file for existing result: {e}')

    if has_completions:
        return True, existing_result

    return False, None


def _cleanup_eval_sessions(sessions_dir: str, pre_existing: set[str]) -> None:
    """Remove session directories created during a single eval run."""
    if not os.path.isdir(sessions_dir):
        return
    try:
        for entry in os.listdir(sessions_dir):
            if entry not in pre_existing:
                entry_path = os.path.join(sessions_dir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                    logger.debug(f'Cleaned up eval session dir: {entry_path}')
    except Exception as e:
        logger.warning(f'Failed to clean up eval session files: {e}')


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:

    should_skip, existing_result = _has_existing_result(metadata.eval_output_dir, instance.instance_id)
    if should_skip:
        if existing_result:
            return EvalOutput(
                instance_id=existing_result.get('instance_id', instance.instance_id),
                instruction=existing_result.get('instruction', ''),
                instance=existing_result.get('instance', instance.to_dict()),
                test_result=existing_result.get('test_result', {}),
                metadata=metadata,
                history=existing_result.get('history', []),
                metrics=existing_result.get('metrics', {}),
                error=existing_result.get('error'),
            )
        else:
            return EvalOutput(
                instance_id=instance.instance_id,
                instruction='',
                instance=instance.to_dict(),
                test_result={
                    'git_patch': '',
                    'skipped': True,
                    'skip_reason': 'completions_exist_no_result',
                },
                metadata=metadata,
                history=[],
                metrics={},
                error=None,
            )

    config = get_config(instance, metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource_factor with increasing attempt_id
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )
    start_time = time.perf_counter()
    runtime = create_runtime(config)
    end_time = time.perf_counter()
    update_metrics({"create_runtime_time": end_time - start_time})
    print(f"create runtime: {end_time - start_time} seconds", flush = True)
    start_time = time.perf_counter()
    call_async_from_sync(runtime.connect)
    end_time = time.perf_counter()
    update_metrics({"connect_to_runtime_time": end_time - start_time})
    print(f"connect to runtime: {end_time - start_time} seconds", flush = True)

    # Snapshot existing session dirs so we can clean up the new one afterward
    eval_sessions_dir = os.path.join(config.file_store_path, 'sessions')
    pre_existing_sessions = set(os.listdir(eval_sessions_dir)) if os.path.isdir(eval_sessions_dir) else set()

    try:
        start_time = time.perf_counter()
        untracked_baseline = initialize_runtime(runtime, instance, metadata)
        end_time = time.perf_counter()
        update_metrics({"initialize_runtime_time": end_time - start_time})
        print(f"init runtime: {end_time - start_time} seconds", flush = True)
        message_action = get_instruction(instance, metadata)

        # Check if we have replay data for this instance
        replay_events = None
        if instance.instance_id in REPLAY_DATA:
            replay_events, replay_initial_action = REPLAY_DATA[instance.instance_id]
            message_action = replay_initial_action
            logger.info(
                f'Replay mode: replaying {len(replay_events)} events for {instance.instance_id}'
            )

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        start_time = time.perf_counter()
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                exit_on_message=(
                    metadata.agent_class
                    in AGENT_CLASSES_THAT_STOP_WITHOUT_TOOL_CALL
                ),
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                    metadata.agent_class
                ),
                replay_events=replay_events,
            )
        )
        end_time = time.perf_counter()
        print(f"run controller: {end_time - start_time} seconds", flush = True)

        # if fatal error, throw EvalError to trigger re-run
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # ======= THIS IS SWE-Bench specific =======
        # Get git patch
        def complete_live_patch() -> dict[str, Any]:
            from evaluation.benchmarks.swe_bench.live_utils import (
                complete_runtime as complete_runtime_fn,
            )

            return complete_runtime_fn(runtime, instance)

        return_val = dispatch_patch_completion(
            DATASET_TYPE,
            untracked_baseline,
            complete_live=complete_live_patch,
            complete_normal=lambda baseline: complete_runtime(
                runtime, instance, baseline
            ),
            missing_baseline_error=lambda: EvalException(
                'Missing host-held pre-existing untracked paths baseline'
            ),
        )
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()
        # _cleanup_eval_sessions(eval_sessions_dir, pre_existing_sessions)
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'git_patch': git_patch,
    }

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    # NOTE: this is NO LONGER the event stream, but an agent history that includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    # Calculate action execution times from history
    metrics['action_execution_latencies'] = get_action_execution_latencies(state.history)

    # Save the output
    instruction = message_action.content
    if message_action.image_urls:
        instruction += (
            '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
        )
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def get_action_execution_latencies(history: list) -> list[dict]:
    """Extract execution latencies from observations in the history."""
    latencies = []
    for event in history:
        if isinstance(event, Observation):
            execution_latency = getattr(event, '_execution_latency', None)
            if execution_latency is None:
                execution_latency = getattr(event, 'execution_latency', None)
            if execution_latency is not None:
                latencies.append({
                    'observation_type': type(event).__name__,
                    'observation_id': str(event.id),
                    'latency': float(execution_latency),
                    'message': event.message,
                    'timestamp': event.timestamp,
                })
    return latencies


def filter_dataset(
    dataset: pd.DataFrame,
    filter_column: str,
    selected_ids: list[str] | None = None,
) -> pd.DataFrame:
    # First check command-line args, then fall back to config.toml
    if selected_ids is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = toml.load(file)
                if 'selected_ids' in data:
                    selected_ids = data['selected_ids']

    if selected_ids:
        logger.info(f'Filtering {len(selected_ids)} tasks from selected_ids...')
        subset = dataset[dataset[filter_column].isin(selected_ids)]
        logger.info(f'Retained {subset.shape[0]} tasks after filtering')
        return subset

    skip_ids = os.environ.get('SKIP_IDS', '').split(',')
    if len(skip_ids) > 0 and skip_ids[0]:
        logger.info(f'Filtering {len(skip_ids)} tasks from SKIP_IDS...')
        return dataset[~dataset[filter_column].isin(skip_ids)]
    return dataset


if __name__ == '__main__':
    # Declare globals at the start of the block
    global SYSTEM_PROMPT_PATH, SYSTEM_PROMPT_LONG_HORIZON_PATH, INCLUDE_TURNS_REMAINING_REMINDER

    # Module-level variables to store custom prompt paths
    SYSTEM_PROMPT_PATH = None
    SYSTEM_PROMPT_LONG_HORIZON_PATH = None
    INCLUDE_TURNS_REMAINING_REMINDER = False

    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )
    parser.add_argument(
        '--selected-id',
        type=str,
        default=None,
        help='Single instance ID to evaluate (e.g., "django__django-12345")',
    )
    parser.add_argument(
        '--instance-dict-path',
        type=str,
        default=None,
        help='Path to a JSON file containing instance data to use instead of loading from HuggingFace (e.g., \'{"instance_id": "...", "repo": "...", ...}\')',
    )
    parser.add_argument(
        '--instruction-template-path',
        type=str,
        default=None,
        help='Path to a custom instruction template file (overrides swe_default.j2)',
    )
    parser.add_argument(
        '--system-prompt-path',
        type=str,
        default=None,
        help='Path to a custom system_prompt.j2 file',
    )
    parser.add_argument(
        '--system-prompt-long-horizon-path',
        type=str,
        default=None,
        help='Path to a custom system_prompt_long_horizon.j2 file',
    )
    parser.add_argument(
        '--replay-messages-path',
        type=str,
        default=None,
        help='Path to a JSON file with raw LLM messages (OpenAI chat format) to replay before continuing.',
    )
    parser.add_argument(
        '--include-turns-remaining-reminder',
        action='store_true',
        default=False,
        help='If set, append "ENVIRONMENT REMINDER: You have X turns left to complete the task." to every observation shown to the LLM.',
    )

    args, _ = parser.parse_known_args()
    INCLUDE_TURNS_REMAINING_REMINDER = args.include_turns_remaining_reminder

    maybe_base_profile_dir = os.environ.get("NG_PROFILING_DIR")
    should_profile = maybe_base_profile_dir is not None
    if should_profile:
        profiler = Profiler(
            name="openhands", base_profile_dir=Path(maybe_base_profile_dir)
        )
        profiler.start()

    # Validate nv-internal-1 and SWE-rebench-V2 require instance_dict_path
    if any(dataset_type in args.dataset.lower() for dataset_type in ('nv-internal-1', 'swe-rebench-v2', 'swe-bench-ext')):
        if not args.instance_dict_path or not args.selected_id:
            raise ValueError(
                f'{args.dataset} dataset requires both --instance-dict-path and --selected-id arguments. '
                'This dataset does not support HuggingFace dataset loading.'
            )

    # Check if instance data is provided directly
    if args.instance_dict_path and args.selected_id:
        logger.info(f"Loading instance from --instance-dict-path: {args.instance_dict_path}")
        with open(args.instance_dict_path, "r") as f:
            instance_data = [json.loads(line) for line in f]
            instance_data = [
                instance_dict
                for instance_dict in instance_data
                if instance_dict["instance_id"] == args.selected_id
            ][0]

        # delete the instance_dict_path to avoid information leakage
        # os.remove(args.instance_dict_path)
        swe_bench_tests = pd.DataFrame([instance_data])
        logger.info(f"Loaded instance from --instance-dict-path: {instance_data.get('instance_id', 'unknown')}")
        set_dataset_type(args.dataset)
    else:
        # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
        # so we don't need to manage file uploading to OpenHands's repo
        dataset = load_dataset(args.dataset, split=args.split)

        # Set the global dataset type based on dataset name
        set_dataset_type(args.dataset)

        # Parse selected_id from args
        selected_ids = [args.selected_id] if args.selected_id else None

        swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id', selected_ids)
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )

    # Insert dummy columns that OpenHands expects to be present
    if "PASS_TO_PASS" not in swe_bench_tests:
        swe_bench_tests["PASS_TO_PASS"] = "[]"
    if "FAIL_TO_PASS" not in swe_bench_tests:
        swe_bench_tests["FAIL_TO_PASS"] = "[]"
    if "version" not in swe_bench_tests:
        swe_bench_tests["version"] = "1.0"

    # if DATASET_TYPE == 'SWE-Gym':
    #     with open(
    #         os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             'split',
    #             'swegym_verified_instances.json',
    #         ),
    #         'r',
    #     ) as f:
    #         swegym_verified_instances = json.load(f)
    #         swe_bench_tests = swe_bench_tests[
    #             swe_bench_tests['instance_id'].isin(swegym_verified_instances)
    #         ]
    #     logger.info(
    #         f'{len(swe_bench_tests)} tasks left after filtering for SWE-Gym verified instances'
    #     )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Get condenser config from environment variable
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        # If no specific condenser config is provided via env var, default to NoOpCondenser
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    # Set up custom system prompt paths if provided
    if args.system_prompt_path:
        if os.path.isfile(args.system_prompt_path):
            SYSTEM_PROMPT_PATH = args.system_prompt_path
            logger.info(f'Using custom system_prompt.j2: {SYSTEM_PROMPT_PATH}')
        else:
            raise ValueError(f'System prompt file does not exist: {args.system_prompt_path}')

    if args.system_prompt_long_horizon_path:
        if os.path.isfile(args.system_prompt_long_horizon_path):
            SYSTEM_PROMPT_LONG_HORIZON_PATH = args.system_prompt_long_horizon_path
            logger.info(f'Using custom system_prompt_long_horizon.j2: {SYSTEM_PROMPT_LONG_HORIZON_PATH}')
        else:
            raise ValueError(f'System prompt long horizon file does not exist: {args.system_prompt_long_horizon_path}')

    # Build details dict with custom prompt paths
    details = {'mode': args.mode}
    if args.instruction_template_path:
        details['instruction_template_path'] = args.instruction_template_path
        logger.info(f'Custom instruction template path: {args.instruction_template_path}')

    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        agent_config=agent_config,
        condenser_config=condenser_config,
    )

    # Load replay messages if provided
    if args.replay_messages_path:
        if not args.selected_id:
            raise ValueError('--replay-messages-path requires --selected-id')
        with open(args.replay_messages_path, 'r') as f:
            replay_messages = json.load(f)
        replay_events, replay_initial_action = messages_to_replay_events(
            replay_messages, metadata.agent_class
        )
        REPLAY_DATA[args.selected_id] = (replay_events, replay_initial_action)
        logger.info(
            f'Loaded replay history for {args.selected_id}: '
            f'{len(replay_events)} events to replay'
        )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')

    # Run evaluation in iterative mode:
    # If a rollout fails to output AgentFinishAction, we will try again until it succeeds OR total 3 attempts have been made.
    ITERATIVE_EVAL_MODE = (
        os.environ.get('ITERATIVE_EVAL_MODE', 'false').lower() == 'true'
    )
    ITERATIVE_EVAL_MODE_MAX_ATTEMPTS = int(
        os.environ.get('ITERATIVE_EVAL_MODE_MAX_ATTEMPTS', '3')
    )

    if not ITERATIVE_EVAL_MODE:
        # load the dataset
        instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)
        if len(instances) > 0 and not isinstance(
            instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
        ):
            for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                instances[col] = instances[col].apply(lambda x: str(x))

        run_evaluation(
            instances,
            metadata,
            output_file,
            args.eval_num_workers,
            process_instance,
            timeout_seconds=8
            * 60
            * 60,  # 8 hour PER instance should be more than enough
            max_retries=MAX_RETRIES,
        )
    else:
        critic = AgentFinishedCritic()

        def get_cur_output_file_path(attempt: int) -> str:
            return (
                f'{output_file.removesuffix(".jsonl")}.critic_attempt_{attempt}.jsonl'
            )

        eval_ids = None
        for attempt in range(1, ITERATIVE_EVAL_MODE_MAX_ATTEMPTS + 1):
            cur_output_file = get_cur_output_file_path(attempt)
            logger.info(
                f'Running evaluation with critic {critic.__class__.__name__} for attempt {attempt} of {ITERATIVE_EVAL_MODE_MAX_ATTEMPTS}.'
            )

            # For deterministic eval, we set temperature to 0.1 for (>1) attempt
            # so hopefully we get slightly different results
            if attempt > 1 and metadata.llm_config.temperature == 0:
                logger.info(
                    f'Detected temperature is 0 for (>1) attempt {attempt}. Setting temperature to 0.1...'
                )
                metadata.llm_config.temperature = 0.1

            # Load instances - at first attempt, we evaluate all instances
            # On subsequent attempts, we only evaluate the instances that failed the previous attempt determined by critic
            instances = prepare_dataset(
                swe_bench_tests, cur_output_file, args.eval_n_limit, eval_ids=eval_ids
            )
            if len(instances) > 0 and not isinstance(
                instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
            ):
                for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                    instances[col] = instances[col].apply(lambda x: str(x))

            # Run evaluation - but save them to cur_output_file
            logger.info(
                f'Evaluating {len(instances)} instances for attempt {attempt}...'
            )
            run_evaluation(
                instances,
                metadata,
                cur_output_file,
                args.eval_num_workers,
                process_instance,
                timeout_seconds=8
                * 60
                * 60,  # 8 hour PER instance should be more than enough
                max_retries=MAX_RETRIES,
            )

            # When eval is done, we update eval_ids to the instances that failed the current attempt
            instances_failed = []
            logger.info(
                f'Use critic {critic.__class__.__name__} to check {len(instances)} instances for attempt {attempt}...'
            )
            with open(cur_output_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    try:
                        history = [
                            event_from_dict(event) for event in instance['history']
                        ]
                        critic_result = critic.evaluate(
                            history, instance['test_result'].get('git_patch', '')
                        )
                        if not critic_result.success:
                            instances_failed.append(instance['instance_id'])
                    except Exception as e:
                        logger.error(
                            f'Error loading history for instance {instance["instance_id"]}: {e}'
                        )
                        instances_failed.append(instance['instance_id'])
            logger.info(
                f'{len(instances_failed)} instances failed the current attempt {attempt}: {instances_failed}'
            )
            eval_ids = instances_failed

            # If no instances failed, we break
            if len(instances_failed) == 0:
                break

        # Then we should aggregate the results from all attempts into the original output file
        # and remove the intermediate files
        logger.info(
            'Aggregating results from all attempts into the original output file...'
        )
        fout = open(output_file, 'w')
        added_instance_ids = set()
        for attempt in reversed(range(1, ITERATIVE_EVAL_MODE_MAX_ATTEMPTS + 1)):
            cur_output_file = get_cur_output_file_path(attempt)
            if not os.path.exists(cur_output_file):
                logger.warning(
                    f'Intermediate output file {cur_output_file} does not exist. Skipping...'
                )
                continue

            with open(cur_output_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    # Also make sure git_patch is not empty - otherwise we fall back to previous attempt (empty patch is worse than anything else)
                    if (
                        instance['instance_id'] not in added_instance_ids
                        and instance['test_result'].get('git_patch', '').strip()
                    ):
                        fout.write(line)
                        added_instance_ids.add(instance['instance_id'])
            logger.info(
                f'Aggregated instances from {cur_output_file}. Total instances added so far: {len(added_instance_ids)}'
            )
        fout.close()
        logger.info(
            f'Done! Total {len(added_instance_ids)} instances added to {output_file}'
        )
        # Check if any instances reached maximum retries
        check_maximum_retries_exceeded(metadata.eval_output_dir)

    if should_profile:
        profiler.stop()
