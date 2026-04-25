"""Demo runner for MIMICDecisionEnv.

Usage (from workspace root):
python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 3
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict

from mimic_openenv import (
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    MIMICDecisionEnv,
    evaluate_next_step_predictions,
    format_metrics,
)


def tester_policy(_obs, info: Dict[str, Any]):
    """Policy that simulates tester input and noisy agent responses.

    It intentionally injects occasional medication mismatch
    (tester says vitamin, agent says minerals) to validate dynamicity.
    """
    gt = int(info.get("ground_truth_id", CATEGORY_TO_ID["no_action"]))

    if gt == CATEGORY_TO_ID["medication"]:
        if random.random() < 0.5:
            return {
                "category": "medication",
                "tester_prompt": "give vitamin",
                "agent_response": "give minerals",
            }
        return {
            "category": "medication",
            "tester_prompt": "give vitamin",
            "agent_response": "give vitamin",
        }

    if gt == CATEGORY_TO_ID["diagnosis"]:
        return {
            "category": "diagnosis",
            "tester_prompt": "advice diagnosis",
            "agent_response": "advice diagnosis",
        }

    if gt == CATEGORY_TO_ID["discharge"]:
        return {
            "category": "discharge",
            "tester_prompt": "discharge patient",
            "agent_response": "discharge patient",
        }

    return {
        "category": "no_action",
        "tester_prompt": "no action",
        "agent_response": "no action",
    }


def run_single_episode(env: MIMICDecisionEnv) -> float:
    obs, info = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    print("\n=== New Episode ===")
    print(
        {
            "hadm_id": info.get("hadm_id"),
            "subject_id": info.get("subject_id"),
            "initial_ground_truth": info.get("ground_truth_category"),
        }
    )

    while not (terminated or truncated):
        action = tester_policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        print(
            {
                "step": info.get("step_index"),
                "ground_truth": info.get("ground_truth_category"),
                "decision": info.get("decision_category"),
                "reward": round(reward, 4),
                "medication_lookup": info.get("medication_lookup"),
                "web_search_status": info.get("web_search", {}).get("status"),
                "llm_top": max(info.get("llm_probabilities", {}), key=info.get("llm_probabilities", {}).get)
                if info.get("llm_probabilities")
                else "",
            }
        )

    print(f"Episode total_reward={total_reward:.4f}")
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIMIC Open Gym decision environment demo")
    parser.add_argument(
        "--data-root",
        default="physionet.org/files/mimiciv/3.1",
        help="Path containing hosp/ and icu/ folders (or parent folder)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of demo episodes")
    parser.add_argument("--sample-size", type=int, default=3000, help="Admission sample size")
    args = parser.parse_args()

    env = MIMICDecisionEnv(
        data_root=args.data_root,
        max_steps=6,
        sample_size=args.sample_size,
        trajectory_size=30,
    )

    rewards = []
    for _ in range(max(1, args.episodes)):
        rewards.append(run_single_episode(env))

    print("\n=== Evaluation ===")
    metrics = evaluate_next_step_predictions(
        env=env,
        policy_fn=tester_policy,
        episodes=max(3, args.episodes),
    )
    print(format_metrics(metrics))
    print(f"\nDemo average reward: {sum(rewards)/max(1, len(rewards)):.4f}")


if __name__ == "__main__":
    main()
