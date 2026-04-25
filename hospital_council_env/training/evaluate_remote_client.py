"""Remote client smoke test for the Hospital Council OpenEnv server."""

from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean

from hospital_council_env import HospitalCouncilEnv
from hospital_council_env.training.policies import baseline_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hospital Council through the OpenEnv client")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    rewards = []
    scenario_counter = Counter()
    milestone_scores = []
    terminal_scores = []

    with HospitalCouncilEnv(base_url=args.base_url).sync() as env:
        for _ in range(max(1, args.episodes)):
            result = env.reset()
            obs = result.observation
            scenario_counter[obs.scenario_type] += 1
            episode_reward = 0.0

            while not obs.done:
                action = baseline_policy(obs)
                result = env.step(action)
                obs = result.observation
                episode_reward += float(result.reward or 0.0)
                milestone_scores.append(float(obs.scoreboard.get("milestone", 0.0)))
                terminal_scores.append(float(obs.scoreboard.get("terminal", 0.0)))

            rewards.append(episode_reward)

    print("=== Remote Client Smoke Test ===")
    print(f"Episodes: {len(rewards)}")
    print(f"Average Reward: {mean(rewards):.4f}")
    print(f"Average Milestone Score: {mean(milestone_scores) if milestone_scores else 0.0:.4f}")
    print(f"Average Terminal Score: {mean(terminal_scores) if terminal_scores else 0.0:.4f}")
    print(f"Scenario Mix: {dict(scenario_counter)}")


if __name__ == "__main__":
    main()
