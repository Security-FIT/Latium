"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EditOutcome:
    success: bool = True
    probe_vector: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    modified_weights: dict[str, tuple[int, ...] | None] = field(default_factory=dict)
    restorations: dict[str, Any] = field(default_factory=dict, repr=False)


class EditMethod(Protocol):
    identifier: str
    description: str

    def apply(self, handler: Any, case: dict[str, Any]) -> EditOutcome:
        """Apply an edit, rolling it back before raising and returning restoration state on success."""
        ...

    def evaluate(
        self,
        handler: Any,
        case: dict[str, Any],
        outcome: EditOutcome,
    ) -> dict[str, Any]:
        """Evaluate an applied edit while the edited model is still resident."""
        ...
