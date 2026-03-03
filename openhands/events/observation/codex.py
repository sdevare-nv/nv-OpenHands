"""Codex-specific observation classes."""

from dataclasses import dataclass, field

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class CodexApplyPatchObservation(Observation):
    """Observation for Codex ApplyPatch action execution.

    Attributes:
        files_changed: List of files that were modified by the patch.
        success: Whether the patch was successfully applied.
    """

    files_changed: list[str] = field(default_factory=list)
    success: bool = True
    observation: str = ObservationType.CODEX_APPLY_PATCH

    @property
    def message(self) -> str:
        count = len(self.files_changed)
        if self.success:
            return f"Applied patch to {count} file{'s' if count != 1 else ''}"
        return "Failed to apply patch"


@dataclass
class CodexUpdatePlanObservation(Observation):
    """Observation for Codex UpdatePlan action execution.

    Attributes:
        plan: The current plan items after the update.
        success: Whether the plan was successfully updated.
    """

    plan: list[dict] = field(default_factory=list)
    success: bool = True
    observation: str = ObservationType.CODEX_UPDATE_PLAN

    @property
    def message(self) -> str:
        count = len(self.plan)
        if self.success:
            return f"Plan updated with {count} step{'s' if count != 1 else ''}"
        return "Failed to update plan"
