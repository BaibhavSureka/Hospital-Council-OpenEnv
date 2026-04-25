"""Local demo runner for the Hospital Council environment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from hospital_council_env.server.hospital_council_env_environment import HospitalCouncilEnvironment
from hospital_council_env.training.policies import baseline_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Hospital Council demo")
    parser.add_argument("--data-root", default="physionet.org/files/mimiciv/3.1")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--record-path",
        default="",
        help="Optional JSONL path for demo-video narration and rollout events.",
    )
    args = parser.parse_args()

    env = HospitalCouncilEnvironment(
        data_root=args.data_root,
        sample_size=args.sample_size,
    )

    episode_rewards = []
    scenario_counter = Counter()
    recording = []
    for episode_idx in range(max(1, args.episodes)):
        observation = env.reset()
        total_reward = 0.0
        print(f"\n=== Episode {episode_idx + 1} ===")
        print({"scenario": observation.scenario_type, "brief": observation.mission_brief})
        recording.append(
            {
                "event": "episode_start",
                "episode": episode_idx + 1,
                "scenario": observation.scenario_type,
                "brief": observation.mission_brief,
                "task_graph": observation.task_graph,
                "web_augmentation": observation.web_augmentation,
            }
        )
        while not observation.done:
            action = baseline_policy(observation)
            observation = env.step(action)
            total_reward += float(observation.reward or 0.0)
            event = {
                "event": "step",
                "episode": episode_idx + 1,
                "phase": observation.phase_name,
                "action_type": action.action_type,
                "target": action.target,
                "category": action.category,
                "reward": round(float(observation.reward or 0.0), 4),
                "outcome": observation.last_outcome,
                "scoreboard": observation.scoreboard,
                "active_task": observation.task_graph.get("active_node", ""),
                "task_graph_loss": observation.scoreboard.get("task_graph_loss", 0.0),
                "web_status": observation.web_augmentation.get("status", ""),
                "web_valid_cases": observation.web_augmentation.get("valid_use_cases", [])[:2],
                "context_classification": observation.context_observation.get("classification", ""),
                "next_step_guidance": observation.context_observation.get("next_step_guidance", ""),
                "correction_signal": observation.context_observation.get("correction_signal", {}),
            }
            print(event)
            recording.append(event)
        scenario_counter[observation.scenario_type] += 1
        episode_rewards.append(total_reward)
        print({"episode_reward": round(total_reward, 4)})
        recording.append(
            {
                "event": "episode_end",
                "episode": episode_idx + 1,
                "episode_reward": round(total_reward, 4),
            }
        )

    print("\n=== Summary ===")
    summary = {
        "episodes": len(episode_rewards),
        "avg_reward": round(mean(episode_rewards), 4),
        "scenario_mix": dict(scenario_counter),
    }
    print(summary)
    recording.append({"event": "summary", **summary})

    if args.record_path:
        record_path = Path(args.record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=True) for item in recording) + "\n",
            encoding="utf-8",
        )
        print({"record_path": str(record_path)})


if __name__ == "__main__":
    main()
