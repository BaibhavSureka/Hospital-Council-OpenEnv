"""Minimal TRL GRPO training script for Hospital Council.

This script is intentionally small and Colab-friendly. It uses TRL's
`environment_factory` integration, where public methods become tools.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from datasets import Dataset
from packaging.version import Version
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

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


def ensure_environment_factory_support() -> None:
    minimum_version = Version("5.2.0")
    current_version = Version(transformers_version)
    if current_version < minimum_version:
        raise ImportError(
            "TRL environment_factory requires transformers>=5.2.0. "
            f"Found transformers=={transformers_version}. "
            "Install the Hugging Face main branch first with "
            '`pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"`.'
        )


def load_policy_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        low_cpu_mem_usage=False,
        dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GRPO trainer for Hospital Council")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/grpo_hospital_council")
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    ensure_environment_factory_support()
    use_cpu = not torch.cuda.is_available()
    batch_size = max(2, args.num_generations)
    model, tokenizer = load_policy_model_and_tokenizer(args.model)

    trainer = GRPOTrainer(
        model=model,
        train_dataset=build_dataset(args.dataset_size),
        reward_funcs=reward_func,
        processing_class=tokenizer,
        args=GRPOConfig(
            output_dir=args.output_dir,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            log_completions=True,
            chat_template_kwargs={"enable_thinking": False},
            max_steps=args.max_steps,
            use_cpu=use_cpu,
            bf16=not use_cpu,
            fp16=False,
            per_device_train_batch_size=batch_size,
        ),
        environment_factory=HospitalCouncilToolEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
