# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable reward rubrics for the Hospital Council environment."""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import WeightedSum
from openenv.core.rubrics.base import Rubric


def _score(observation: Any, key: str, default: float = 0.0) -> float:
    metadata = getattr(observation, "metadata", {}) or {}
    subscores = metadata.get("subscores", {}) or {}
    try:
        return float(subscores.get(key, default))
    except (TypeError, ValueError):
        return default


class MilestoneRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "milestone")


class CoalitionRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "coalition")


class SafetyRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "safety")


class EfficiencyRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "efficiency")


class TerminalSuccessRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "terminal")


class TaskGraphRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        return _score(observation, "task_graph")


class HospitalCouncilRubric(WeightedSum):
    """Reward container aligned with the judging criteria story."""

    def __init__(self) -> None:
        super().__init__(
            [
                MilestoneRubric(),
                CoalitionRubric(),
                SafetyRubric(),
                EfficiencyRubric(),
                TerminalSuccessRubric(),
                TaskGraphRubric(),
            ],
            weights=[0.30, 0.15, 0.20, 0.10, 0.10, 0.15],
        )
