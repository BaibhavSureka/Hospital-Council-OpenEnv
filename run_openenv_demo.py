"""Demo runner for the dynamic MIMICDecisionEnv loop.

Usage (from workspace root):
python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 3
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict

from mimic_openenv import (
    CATEGORY_TO_ID,
    MIMICDecisionEnv,
    evaluate_next_step_predictions,
    format_metrics,
)


def build_tester_agent(env: MIMICDecisionEnv, rng: random.Random):
    def tester_agent(_obs, info: Dict[str, Any]) -> Dict[str, Any]:
        case = env.generate_tester_query(env.current_record, env.step_idx)
        if (
            case["ground_truth_category"] == "medication"
            and rng.random() < 0.25
            and env.current_record
            and len(env.current_record.meds) > 1
        ):
            case["query"] = "there may be something worth starting, but it is not obvious yet"
        elif case["ground_truth_category"] == "diagnosis" and rng.random() < 0.30:
            case["query"] = "we still do not really know what explains this; what would you do first"
        elif case["ground_truth_category"] == "discharge" and rng.random() < 0.20:
            case["query"] = "this may be close to the end of the stay; what next"
        elif case["ground_truth_category"] == "no_action" and rng.random() < 0.20:
            case["query"] = "nothing is screaming for intervention; what would you do now"
        case.setdefault("metadata", {})
        case["metadata"]["seed_query"] = info.get("tester_query", "")
        return case

    return tester_agent


def build_primary_agent(env: MIMICDecisionEnv, rng: random.Random):
    def primary_agent(tester_case: Dict[str, Any], obs, info: Dict[str, Any]) -> Dict[str, Any]:
        default_action = env.default_primary_agent(tester_case, obs, info)
        state = dict(info.get("state", {}))
        step_progress = float(info.get("step_progress", 0.0))

        if state.get("med_count", 0) and rng.random() < 0.12:
            return {
                "category": "medication",
                "agent_response": "give minerals",
            }
        if (
            state.get("diag_count", 0) == 0
            and step_progress < 0.8
            and rng.random() < 0.06
        ):
            return {
                "category": "no_action",
                "agent_response": "continue monitoring",
            }
        return default_action

    return primary_agent


def build_eval_policy(env: MIMICDecisionEnv, tester_agent, primary_agent):
    def policy(obs, info: Dict[str, Any]):
        tester_case = tester_agent(obs, info)
        primary_case = env._prepare_primary_agent_case(tester_case)
        primary_context = env.build_primary_agent_context(info)
        action = primary_agent(primary_case, obs, primary_context)
        return env._compose_action_payload(tester_case, action)

    return policy


def run_single_episode(
    env: MIMICDecisionEnv,
    tester_agent,
    primary_agent,
) -> float:
    interactions = env.run_agent_episode(
        tester_agent=tester_agent,
        primary_agent=primary_agent,
    )
    total_reward = sum(float(step.get("reward", 0.0)) for step in interactions)

    print("\n=== New Episode ===")
    if interactions:
        first = interactions[0]
        print(
            {
                "hadm_id": first.get("hadm_id"),
                "subject_id": first.get("subject_id"),
                "initial_ground_truth": first.get("ground_truth_category"),
            }
        )

    for step in interactions:
        print(
            {
                "step": step.get("step_index"),
                "query": step.get("tester_query"),
                "ground_truth": step.get("ground_truth_action"),
                "decision": step.get("decision_action"),
                "classification": step.get("reasoning_classification"),
                "confidence": round(float(step.get("reasoning_confidence", 0.0)), 4),
                "db_exists": step.get("database_validation", {}).get("exists"),
                "retrieved": len(step.get("retrieved_trajectories", [])),
                "web_status": step.get("web_search", {}).get("status"),
                "reward": round(float(step.get("reward", 0.0)), 4),
            }
        )

    print(f"Episode total_reward={total_reward:.4f}")
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIMIC dynamic agent-environment demo")
    parser.add_argument(
        "--data-root",
        default="physionet.org/files/mimiciv/3.1",
        help="Path containing hosp/ and icu/ folders (or parent folder)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of demo episodes")
    parser.add_argument("--sample-size", type=int, default=3000, help="Admission sample size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible demo behavior")
    args = parser.parse_args()

    demo_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 101)
    env = MIMICDecisionEnv(
        data_root=args.data_root,
        max_steps=6,
        sample_size=args.sample_size,
        trajectory_size=30,
        seed=args.seed,
    )
    tester_agent = build_tester_agent(env, demo_rng)
    primary_agent = build_primary_agent(env, demo_rng)
    eval_policy = build_eval_policy(
        env,
        build_tester_agent(env, eval_rng),
        build_primary_agent(env, eval_rng),
    )

    rewards = []
    for _ in range(max(1, args.episodes)):
        rewards.append(run_single_episode(env, tester_agent, primary_agent))

    print("\n=== Continuous Loop Summary ===")
    loop_summary = env.run_continuous_loop(
        tester_agent=tester_agent,
        primary_agent=primary_agent,
        episodes=max(1, args.episodes),
    )
    print(
        {
            "episodes": len(loop_summary["episodes"]),
            "avg_reward": round(float(loop_summary["avg_reward"]), 4),
            "memory_size": loop_summary["memory_size"],
        }
    )

    print("\n=== Evaluation ===")
    metrics = evaluate_next_step_predictions(
        env=env,
        policy_fn=eval_policy,
        episodes=max(10, args.episodes * 4),
    )
    print(format_metrics(metrics))
    print(f"\nDemo average reward: {sum(rewards)/max(1, len(rewards)):.4f}")


if __name__ == "__main__":
    main()
