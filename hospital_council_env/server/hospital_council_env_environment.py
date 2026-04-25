# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv server implementation for the Hospital Council environment."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import HospitalCouncilAction, HospitalCouncilObservation, HospitalCouncilState
    from ..rubrics import HospitalCouncilRubric
    from ..simulator import MIMICCouncilSimulator, STAKEHOLDER_DESCRIPTIONS
except ImportError:
    from models import HospitalCouncilAction, HospitalCouncilObservation, HospitalCouncilState
    from rubrics import HospitalCouncilRubric
    from simulator import MIMICCouncilSimulator, STAKEHOLDER_DESCRIPTIONS


_REPO_DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "physionet.org" / "files" / "mimiciv" / "3.1"
)
DEFAULT_DATA_ROOT = os.environ.get(
    "MIMIC_DATA_ROOT",
    str(_REPO_DEFAULT_DATA_ROOT if _REPO_DEFAULT_DATA_ROOT.exists() else "physionet.org/files/mimiciv/3.1"),
)


class HospitalCouncilEnvironment(
    Environment[HospitalCouncilAction, HospitalCouncilObservation, HospitalCouncilState]
):
    """Long-horizon multi-agent hospital coordination environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _shared_simulator: MIMICCouncilSimulator | None = None

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        max_steps: int = 6,
        sample_size: int = 3000,
    ) -> None:
        super().__init__(rubric=HospitalCouncilRubric())
        if HospitalCouncilEnvironment._shared_simulator is None:
            HospitalCouncilEnvironment._shared_simulator = MIMICCouncilSimulator(
                data_root=data_root,
                max_steps=max_steps,
                sample_size=sample_size,
            )
        self.simulator = HospitalCouncilEnvironment._shared_simulator
        self._snapshot = None
        self._state = HospitalCouncilState(
            episode_id=str(uuid4()),
            scenario_id="",
            scenario_type="diagnostic_ambiguity",
            difficulty="medium",
            max_steps=max_steps,
        )

    def _available_actions(self) -> list[str]:
        return [
            "consult <stakeholder> to gather hidden information or beliefs",
            "propose <category> to set directional intent",
            "delegate <stakeholder> with a concrete follow-up task",
            "resolve <stakeholder> to reduce coalition friction",
            "commit <category> when the council is ready to execute",
        ]

    def _build_observation(
        self,
        subscores: dict[str, float] | None = None,
    ) -> HospitalCouncilObservation:
        if self._snapshot is None:
            return HospitalCouncilObservation(
                mission_brief="Environment not initialized.",
                phase_name="idle",
                available_actions=self._available_actions(),
            )

        stage = self._snapshot.stages[min(self._snapshot.step_count, len(self._snapshot.stages) - 1)]
        scoreboard = {
            "coalition_support": round(
                sum(item.alignment for item in self._snapshot.stakeholders.values())
                / max(1, len(self._snapshot.stakeholders)),
                4,
            ),
            "steps_remaining": float(max(0, self.simulator.max_steps - self._snapshot.step_count)),
            "diagnostic_clarity": 1.0 if self._snapshot.diagnostic_clarity else 0.0,
            "medication_started": 1.0 if self._snapshot.medication_started else 0.0,
            "discharge_ready": 1.0 if self._snapshot.discharge_ready else 0.0,
            "task_graph_score": round(float(self._snapshot.task_graph.get("score", 0.0)), 4),
            "task_graph_loss": round(float(self._snapshot.task_graph.get("loss", 0.0)), 4),
            "web_evidence_count": float(self._snapshot.web_augmentation.get("evidence_count", 0)),
        }
        metadata = {
            "subscores": subscores or {},
            "phase_rationale": stage.rationale,
            "stakeholder_descriptions": dict(STAKEHOLDER_DESCRIPTIONS),
            "task_graph": dict(self._snapshot.task_graph),
            "web_augmentation": dict(self._snapshot.web_augmentation),
        }
        return HospitalCouncilObservation(
            mission_brief=self._snapshot.mission_brief,
            scenario_type=self._snapshot.scenario_type,
            difficulty=self._snapshot.difficulty,
            phase_name=stage.phase_name,
            patient_snapshot=self._snapshot.patient_snapshot,
            stakeholder_updates=self._snapshot.message_log[-4:],
            visible_conflicts=list(self._snapshot.visible_conflicts),
            retrieved_analogies=list(self._snapshot.retrieved_analogies),
            task_graph=dict(self._snapshot.task_graph),
            web_augmentation=dict(self._snapshot.web_augmentation),
            available_actions=self._available_actions(),
            long_horizon_goals=list(self._snapshot.long_horizon_goals),
            scoreboard=scoreboard,
            last_outcome=self._snapshot.last_outcome,
            done=self._snapshot.done,
            reward=0.0,
            metadata=metadata,
        )

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.name = "HospitalCouncilEnvironment"
        metadata.description = (
            "Long-horizon multi-agent hospital operations simulator built from MIMIC-derived encounter seeds."
        )
        metadata.version = "0.2.0"
        metadata.author = "Baibhav Sureka"
        return metadata

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> HospitalCouncilObservation:
        difficulty = str(kwargs.get("difficulty", "medium"))
        scenario_type = kwargs.get("scenario_type")
        self._snapshot = self.simulator.reset(
            seed=seed,
            episode_id=episode_id or str(uuid4()),
            scenario_type=scenario_type,
            difficulty=difficulty,
        )
        self._state = HospitalCouncilState(
            episode_id=self._snapshot.scenario_id,
            **self.simulator.export_state(self._snapshot),
        )
        self._reset_rubric()
        return self._build_observation()

    def step(
        self,
        action: HospitalCouncilAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> HospitalCouncilObservation:
        if self._snapshot is None:
            raise RuntimeError("Call reset() before step().")

        advance = self.simulator.advance(self._snapshot, action.model_dump())
        evaluation = advance["evaluation"]
        observation = self._build_observation(subscores=advance["scoreboard"])
        observation.reward = self._apply_rubric(action, observation)
        observation.done = evaluation.done
        observation.last_outcome = evaluation.outcome_text
        observation.stakeholder_updates = self._snapshot.message_log[-4:]
        observation.visible_conflicts = list(self._snapshot.visible_conflicts)
        observation.retrieved_analogies = list(self._snapshot.retrieved_analogies)
        observation.scoreboard.update(
            {
                "milestone": advance["scoreboard"]["milestone"],
                "safety": advance["scoreboard"]["safety"],
                "efficiency": advance["scoreboard"]["efficiency"],
                "terminal": advance["scoreboard"]["terminal"],
                "task_graph": advance["scoreboard"]["task_graph"],
                "task_graph_loss": advance["scoreboard"]["task_graph_loss"],
                "web_evidence_count": advance["scoreboard"]["web_evidence_count"],
            }
        )
        self._state = HospitalCouncilState(
            episode_id=self._snapshot.scenario_id,
            **self.simulator.export_state(self._snapshot),
        )
        return observation

    @property
    def state(self) -> HospitalCouncilState:
        return self._state
