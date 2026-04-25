# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the Hospital Council OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


StakeholderName = Literal[
    "attending_physician",
    "triage_nurse",
    "pharmacist",
    "bed_manager",
    "family_liaison",
]
ActionType = Literal["consult", "propose", "delegate", "resolve", "commit"]
CategoryName = Literal["diagnosis", "medication", "discharge", "no_action"]
ScenarioType = Literal[
    "diagnostic_ambiguity",
    "medication_alignment",
    "conservative_monitoring",
    "discharge_negotiation",
]
DifficultyName = Literal["easy", "medium", "hard"]


class HospitalCouncilAction(Action):
    """Action emitted by the LLM coordinator."""

    action_type: ActionType = Field(
        ...,
        description="Type of council move: consult, propose, delegate, resolve, or commit.",
    )
    target: Optional[StakeholderName] = Field(
        default=None,
        description="Stakeholder to address for consult/delegate/resolve actions.",
    )
    category: Optional[CategoryName] = Field(
        default=None,
        description="Clinical category the coordinator is steering toward.",
    )
    medication: Optional[str] = Field(
        default=None,
        description="Medication name when the plan involves treatment.",
    )
    message: str = Field(
        default="",
        description="Natural-language plan, question, or negotiation message.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent self-reported confidence in the move.",
    )


class HospitalCouncilObservation(Observation):
    """What the coordinator sees after each step."""

    mission_brief: str = Field(
        default="",
        description="High-level partially observed brief for the current episode.",
    )
    scenario_type: ScenarioType = Field(
        default="diagnostic_ambiguity",
        description="Scenario family for the current episode.",
    )
    difficulty: DifficultyName = Field(
        default="medium",
        description="Difficulty level for this episode.",
    )
    phase_name: str = Field(default="", description="Current long-horizon phase name.")
    patient_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Visible patient facts available to the coordinator.",
    )
    stakeholder_updates: List[str] = Field(
        default_factory=list,
        description="Most recent visible stakeholder notes and reactions.",
    )
    visible_conflicts: List[str] = Field(
        default_factory=list,
        description="Open tensions or unresolved issues among stakeholders.",
    )
    retrieved_analogies: List[str] = Field(
        default_factory=list,
        description="Similar historical council moments shown after weak actions.",
    )
    task_graph: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic phase graph with active node, expected edges, and monitoring loss.",
    )
    web_augmentation: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM-simulated retrieval signals mapped to the current query-action pair.",
    )
    context_observation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context LLM Manager output with classification, correction, and next-step guidance.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Text description of legal or high-value next moves.",
    )
    long_horizon_goals: List[str] = Field(
        default_factory=list,
        description="Persistent goals that matter across the full episode.",
    )
    scoreboard: Dict[str, float] = Field(
        default_factory=dict,
        description="Visible progress metrics such as coalition support and safety.",
    )
    last_outcome: str = Field(
        default="",
        description="Natural-language feedback about the effect of the last move.",
    )


class HospitalCouncilState(State):
    """Full debug state, including hidden targets and beliefs."""

    scenario_id: str = Field(default="", description="Unique scenario identifier.")
    scenario_type: ScenarioType = Field(
        default="diagnostic_ambiguity",
        description="Scenario family for the episode.",
    )
    difficulty: DifficultyName = Field(
        default="medium",
        description="Difficulty level.",
    )
    step_count: int = Field(default=0, ge=0, description="Current episode step index.")
    max_steps: int = Field(default=8, ge=1, description="Maximum episode length.")
    coalition_support: Dict[str, float] = Field(
        default_factory=dict,
        description="Hidden stakeholder alignment values in [0, 1].",
    )
    consulted_stakeholders: List[str] = Field(
        default_factory=list,
        description="Stakeholders consulted so far.",
    )
    diagnostic_clarity: bool = Field(
        default=False,
        description="Whether enough information has been collected for confident diagnosis.",
    )
    medication_started: bool = Field(
        default=False,
        description="Whether treatment has been started.",
    )
    discharge_ready: bool = Field(
        default=False,
        description="Whether the council is actually ready for discharge/handoff.",
    )
    hidden_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step hidden milestone plan used for scoring.",
    )
    task_graph: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latest task-graph monitoring state.",
    )
    web_augmentation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latest web augmentation state used by the reasoning payload.",
    )
    context_observation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latest actionable context observation used to guide the next action.",
    )
    archived_trajectory_size: int = Field(
        default=0,
        ge=0,
        description="How many historical trajectory entries are available for retrieval.",
    )
