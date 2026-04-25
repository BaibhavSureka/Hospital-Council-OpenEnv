"""Minimal TRL GRPO training script for Hospital Council.

This script is intentionally small and Colab-friendly. It uses TRL's
`environment_factory` integration, where public methods become tools.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from hospital_council_env import HospitalCouncilAction, HospitalCouncilEnv


def reward_func(environments, **kwargs) -> list[float]:
    return [float(getattr(env, "reward", 0.0)) for env in environments]


class HospitalCouncilToolEnv:
    """Tool-facing wrapper around the OpenEnv client for GRPO training."""

    def __init__(self):
        base_url = os.environ.get("HOSPITAL_COUNCIL_ENV_URL", "http://localhost:8000")
        self.client = HospitalCouncilEnv(base_url=base_url).sync()
        self.reward = 0.0
        self.done = False
        self._last_observation = None

    def reset(self, **kwargs) -> Optional[str]:
        self.reward = 0.0
        self.done = False
        self._last_observation = self.client.reset(**kwargs).observation
        return (
            f"{self._last_observation.mission_brief}\n"
            f"Visible conflicts: {self._last_observation.visible_conflicts}\n"
            f"Goals: {self._last_observation.long_horizon_goals}\n"
            f"Available actions: {self._last_observation.available_actions}"
        )

    def _run_action(self, action: HospitalCouncilAction) -> str:
        if self.done:
            raise ValueError("Episode already ended.")
        result = self.client.step(action)
        self._last_observation = result.observation
        self.reward = float(result.reward or 0.0)
        self.done = bool(result.done)
        return (
            f"{self._last_observation.last_outcome}\n"
            f"Stakeholder updates: {self._last_observation.stakeholder_updates}\n"
            f"Scoreboard: {self._last_observation.scoreboard}"
        )

    def consult(self, stakeholder: str, question: str) -> str:
        """
        Ask one stakeholder for information or a constraint.

        Args:
            stakeholder: One of attending_physician, triage_nurse, pharmacist, bed_manager, family_liaison.
            question: Short natural-language question for that stakeholder.

        Returns:
            The environment's visible response and updated scoreboard.
        """
        return self._run_action(
            HospitalCouncilAction(
                action_type="consult",
                target=stakeholder,
                message=question,
            )
        )

    def propose(self, category: str, rationale: str, medication: str = "") -> str:
        """
        Propose the council's current direction before committing.

        Args:
            category: diagnosis, medication, discharge, or no_action.
            rationale: Why this direction fits the current situation.
            medication: Optional medication name when category is medication.

        Returns:
            The environment's visible response and updated scoreboard.
        """
        return self._run_action(
            HospitalCouncilAction(
                action_type="propose",
                category=category,
                medication=medication or None,
                message=rationale,
            )
        )

    def delegate(self, stakeholder: str, category: str, task: str, medication: str = "") -> str:
        """
        Delegate a follow-up task to a stakeholder.

        Args:
            stakeholder: Stakeholder to involve.
            category: diagnosis, medication, discharge, or no_action.
            task: The task or coordination request.
            medication: Optional medication name.

        Returns:
            The environment's visible response and updated scoreboard.
        """
        return self._run_action(
            HospitalCouncilAction(
                action_type="delegate",
                target=stakeholder,
                category=category,
                medication=medication or None,
                message=task,
            )
        )

    def resolve(self, stakeholder: str, strategy: str, category: str = "no_action") -> str:
        """
        Resolve conflict with a stakeholder to improve coalition support.

        Args:
            stakeholder: Stakeholder whose friction should be reduced.
            strategy: The coordination or negotiation strategy.
            category: Optional category context for the resolution step.

        Returns:
            The environment's visible response and updated scoreboard.
        """
        return self._run_action(
            HospitalCouncilAction(
                action_type="resolve",
                target=stakeholder,
                category=category,
                message=strategy,
            )
        )

    def commit(self, category: str, decision: str, medication: str = "") -> str:
        """
        Commit the council to a concrete decision.

        Args:
            category: diagnosis, medication, discharge, or no_action.
            decision: Final natural-language commitment.
            medication: Optional medication name.

        Returns:
            The environment's visible response and updated scoreboard.
        """
        return self._run_action(
            HospitalCouncilAction(
                action_type="commit",
                category=category,
                medication=medication or None,
                message=decision,
            )
        )


def build_dataset(size: int = 256) -> Dataset:
    prompt = (
        "You are the coordinator of a hospital council. Use the available tools to gather "
        "information, align stakeholders, and choose safe long-horizon actions. Do not commit "
        "too early when the situation is still unclear."
    )
    return Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * size})


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO trainer for Hospital Council")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/grpo_hospital_council")
    parser.add_argument("--dataset-size", type=int, default=256)
    args = parser.parse_args()

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=build_dataset(args.dataset_size),
        reward_funcs=reward_func,
        args=GRPOConfig(
            output_dir=args.output_dir,
            max_completion_length=2048,
            num_generations=4,
            log_completions=True,
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=HospitalCouncilToolEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
