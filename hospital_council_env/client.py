# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Hospital Council OpenEnv environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import HospitalCouncilAction, HospitalCouncilObservation, HospitalCouncilState


class HospitalCouncilEnv(
    EnvClient[HospitalCouncilAction, HospitalCouncilObservation, HospitalCouncilState]
):
    """Persistent WebSocket client for the long-horizon hospital council environment."""

    def _step_payload(self, action: HospitalCouncilAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[HospitalCouncilObservation]:
        observation = HospitalCouncilObservation(**payload.get("observation", {}))
        observation.reward = payload.get("reward", observation.reward)
        observation.done = payload.get("done", observation.done)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HospitalCouncilState:
        return HospitalCouncilState(**payload)
