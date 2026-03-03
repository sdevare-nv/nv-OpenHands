"""Terminus-2 action classes for keystroke-based terminal interaction.

Terminus-2 sends raw keystrokes to a terminal session (tmux-style) and captures
the resulting screen state, rather than running commands and collecting stdout.
"""

from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class Terminus2CmdRunAction(Action):
    """Sends raw keystrokes to a terminal session.

    Keystrokes are sent verbatim to the terminal. Commands should end with
    '\\n' to execute. Special key sequences use tmux-style escapes:
      - C-c for Ctrl+C
      - C-d for Ctrl+D

    Attributes:
        keystrokes: The exact keystrokes to send to the terminal.
        duration: Seconds to wait for the command to complete before
            capturing output (default 1.0). Cap at 60s.
    """

    keystrokes: str
    duration: float = 1.0
    thought: str = ''
    action: str = ActionType.TERMINUS_2_CMD_RUN
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        ks = self.keystrokes.replace('\n', '\\n')
        if len(ks) > 60:
            ks = ks[:57] + '...'
        return f'Sending keystrokes: {ks} (wait {self.duration}s)'
