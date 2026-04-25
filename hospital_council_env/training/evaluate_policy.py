"""Local evaluator for Hospital Council policies."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Callable, Dict

from hospital_council_env.models import HospitalCouncilAction
from hospital_council_env.server.hospital_council_env_environment import HospitalCouncilEnvironment
from hospital_council_env.training.policies import baseline_policy, random_policy


PolicyFn = Callable[[object], HospitalCouncilAction]


def choose_policy(name: str, seed: int) -> PolicyFn:
    if name == "baseline":
        return baseline_policy
    if name == "random":
        rng = random.Random(seed)
        return lambda obs: random_policy(obs, rng)
    raise ValueError(f"Unknown policy: {name}")


def evaluate_policy(
    env: HospitalCouncilEnvironment,
    policy_fn: PolicyFn,
    episodes: int,
) -> Dict[str, object]:
    total_rewards = []
    episode_successes = 0
    phase_hits = 0
    category_hits = 0
    category_steps = 0
    milestone_hits = 0
    coalition_hits = 0
    task_graph_losses = []
    web_augmented_steps = 0
    context_confidences = []
    guided_replace_steps = 0
    total_steps = 0
    scenario_counts = Counter()
    scenario_rewards = defaultdict(list)
    scenario_success = Counter()

    for _ in range(max(1, episodes)):
        obs = env.reset()
        scenario = obs.scenario_type
        scenario_counts[scenario] += 1
        episode_reward = 0.0

        while not obs.done:
            current_step = env.state.step_count
            target = env.state.hidden_targets[current_step]
            action = policy_fn(obs)
            if action.action_type == target["expected_action_type"]:
                phase_hits += 1
            if action.category is not None:
                category_steps += 1
                if action.category == target["expected_category"]:
                    category_hits += 1

            obs = env.step(action)
            total_steps += 1
            episode_reward += float(obs.reward or 0.0)

            milestone = float(obs.metadata.get("subscores", {}).get("milestone", 0.0))
            terminal = float(obs.metadata.get("subscores", {}).get("terminal", 0.0))
            coalition = float(obs.scoreboard.get("coalition_support", 0.0))
            task_graph_losses.append(float(obs.scoreboard.get("task_graph_loss", 0.0)))
            web_augmented_steps += int(str(obs.web_augmentation.get("status", "")) == "llm_simulated_search")
            context_confidences.append(float(obs.context_observation.get("confidence", 0.0)))
            guided_replace_steps += int(str(obs.context_observation.get("next_step_guidance", "")) == "replace")
            if milestone >= 0.85:
                milestone_hits += 1
            if coalition >= 0.58:
                coalition_hits += 1
            if obs.done and terminal >= 0.99:
                episode_successes += 1
                scenario_success[scenario] += 1

        total_rewards.append(episode_reward)
        scenario_rewards[scenario].append(episode_reward)

    summary = {
        "episodes": len(total_rewards),
        "avg_reward": mean(total_rewards) if total_rewards else 0.0,
        "success_rate": episode_successes / max(1, len(total_rewards)),
        "phase_action_accuracy": phase_hits / max(1, total_steps),
        "category_accuracy": category_hits / max(1, category_steps),
        "milestone_hit_rate": milestone_hits / max(1, total_steps),
        "coalition_threshold_rate": coalition_hits / max(1, total_steps),
        "avg_task_graph_loss": mean(task_graph_losses) if task_graph_losses else 0.0,
        "web_augmented_step_rate": web_augmented_steps / max(1, total_steps),
        "avg_context_confidence": mean(context_confidences) if context_confidences else 0.0,
        "guided_replace_rate": guided_replace_steps / max(1, total_steps),
        "total_steps": total_steps,
        "scenario_breakdown": {
            scenario: {
                "episodes": scenario_counts[scenario],
                "avg_reward": mean(rewards) if rewards else 0.0,
                "success_rate": scenario_success[scenario] / max(1, scenario_counts[scenario]),
            }
            for scenario, rewards in scenario_rewards.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hospital Council locally")
    parser.add_argument("--data-root", default="physionet.org/files/mimiciv/3.1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=["baseline", "random"], default="baseline")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    env = HospitalCouncilEnvironment(
        data_root=args.data_root,
        sample_size=args.sample_size,
    )
    metrics = evaluate_policy(env, choose_policy(args.policy, args.seed), args.episodes)

    if args.as_json:
        print(json.dumps(metrics, indent=2))
        return

    print("=== Hospital Council Evaluation ===")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Total Steps: {metrics['total_steps']}")
    print(f"Average Reward: {metrics['avg_reward']:.4f}")
    print(f"Success Rate: {metrics['success_rate']:.4f}")
    print(f"Phase-Action Accuracy: {metrics['phase_action_accuracy']:.4f}")
    print(f"Category Accuracy: {metrics['category_accuracy']:.4f}")
    print(f"Milestone Hit Rate: {metrics['milestone_hit_rate']:.4f}")
    print(f"Coalition Threshold Rate: {metrics['coalition_threshold_rate']:.4f}")
    print(f"Average Task-Graph Loss: {metrics['avg_task_graph_loss']:.4f}")
    print(f"Web-Augmented Step Rate: {metrics['web_augmented_step_rate']:.4f}")
    print(f"Average Context Confidence: {metrics['avg_context_confidence']:.4f}")
    print(f"Guided Replace Rate: {metrics['guided_replace_rate']:.4f}")
    print("")
    print("Per Scenario:")
    for scenario, values in metrics["scenario_breakdown"].items():
        print(
            f"- {scenario}: episodes={values['episodes']}, "
            f"avg_reward={values['avg_reward']:.4f}, "
            f"success_rate={values['success_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
