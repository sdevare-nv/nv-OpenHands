from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
)


@dataclass
class CmdRunAction(Action):
    command: (
        str  # When `command` is empty, it will be used to print the current tmux window
    )
    is_input: bool = False  # if True, the command is an input to the running process
    thought: str = ''
    blocking: bool = False  # if True, the command will be run in a blocking manner, but a timeout must be set through _set_hard_timeout
    is_static: bool = False  # if True, runs the command in a separate process
    cwd: str | None = None  # optional working directory for the command
    login: bool | None = None  # whether to use login-shell semantics, when supported
    hidden: bool = (
        False  # if True, this command does not go through the LLM or event stream
    )
    # Trusted-harness escape hatch for the command blacklist. Defaults to
    # False (fail-closed): the anti-cheat blacklist is enforced on every
    # command unless the harness itself explicitly opts a command out. The
    # agent can never set this — its actions are built by the function-call
    # parser, which only populates `command`/`thought`/etc. — so this cannot
    # be used to bypass the blacklist during the agent rollout. It exists so
    # the harness's own setup/reset commands (e.g. the deep-reset's
    # `git merge-base --is-ancestor`) are not blocked by rules meant to
    # constrain the agent.
    bypass_blacklist: bool = False
    action: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f'Running command: {self.command}'

    def __str__(self) -> str:
        ret = f'**CmdRunAction (source={self.source}, is_input={self.is_input})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret


@dataclass
class IPythonRunCellAction(Action):
    code: str
    thought: str = ''
    include_extra: bool = (
        True  # whether to include CWD & Python interpreter in the output
    )
    action: str = ActionType.RUN_IPYTHON
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN
    kernel_init_code: str = ''  # code to run in the kernel (if the kernel is restarted)

    def __str__(self) -> str:
        ret = '**IPythonRunCellAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CODE:\n{self.code}'
        return ret

    @property
    def message(self) -> str:
        return f'Running Python code interactively: {self.code}'
