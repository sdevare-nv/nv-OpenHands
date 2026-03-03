"""Terminus-2 observation classes for terminal screen capture output."""

from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class Terminus2CmdOutputObservation(Observation):
    """Observation containing the terminal screen state after keystroke execution.

    Unlike CmdOutputObservation which carries stdout/stderr, this observation
    carries the full terminal screen capture, preserving tmux-style semantics.

    Attributes:
        terminal_state: The captured terminal screen content after execution.
        timed_out: Whether the command timed out before completing.
        command_keystrokes: The keystrokes that were sent (for reference).
    """

    terminal_state: str = ''
    timed_out: bool = False
    command_keystrokes: str = ''
    observation: str = ObservationType.TERMINUS_2_CMD_OUTPUT

    @property
    def message(self) -> str:
        ks = self.command_keystrokes.replace('\n', '\\n')
        if len(ks) > 60:
            ks = ks[:57] + '...'
        suffix = ' (timed out)' if self.timed_out else ''
        return f'Terminal output after: {ks}{suffix}'
