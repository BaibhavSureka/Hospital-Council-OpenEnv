"""Generate reward/loss plots from a GRPO training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def find_trainer_state(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {output_dir}")
    return candidates[0]


def extract_series(log_history: list[dict[str, object]]) -> tuple[list[int], list[float], list[int], list[float]]:
    loss_steps: list[int] = []
    loss_values: list[float] = []
    reward_steps: list[int] = []
    reward_values: list[float] = []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue

        if "loss" in row:
            loss_steps.append(step_int)
            loss_values.append(float(row["loss"]))
        if "reward" in row:
            reward_steps.append(step_int)
            reward_values.append(float(row["reward"]))

    return loss_steps, loss_values, reward_steps, reward_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward/loss curves from a GRPO run")
    parser.add_argument("--output-dir", required=True, help="Directory containing trainer_state.json")
    parser.add_argument(
        "--artifacts-dir",
        default="docs/evidence/training_run",
        help="Where plots and summaries should be written",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trainer_state_path = find_trainer_state(output_dir)
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = trainer_state.get("log_history", [])

    loss_steps, loss_values, reward_steps, reward_values = extract_series(log_history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if loss_steps and loss_values:
        axes[0].plot(loss_steps, loss_values, marker="o", color="#1768ac")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "No loss values logged", ha="center", va="center")
        axes[0].set_axis_off()

    if reward_steps and reward_values:
        axes[1].plot(reward_steps, reward_values, marker="o", color="#2e8b57")
        axes[1].set_title("Reward Trace")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("reward")
        axes[1].grid(alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No reward values logged", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    plot_path = artifacts_dir / "grpo_training_curves.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "trainer_state_path": str(trainer_state_path),
        "logged_rows": len(log_history),
        "loss_points": len(loss_values),
        "reward_points": len(reward_values),
        "latest_loss": loss_values[-1] if loss_values else None,
        "latest_reward": reward_values[-1] if reward_values else None,
        "plot_path": str(plot_path),
    }
    summary_path = artifacts_dir / "training_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_lines = [
        "# GRPO Training Run Summary",
        "",
        f"- Trainer state: `{trainer_state_path}`",
        f"- Logged rows: `{summary['logged_rows']}`",
        f"- Loss points: `{summary['loss_points']}`",
        f"- Reward points: `{summary['reward_points']}`",
        f"- Latest loss: `{summary['latest_loss']}`",
        f"- Latest reward: `{summary['latest_reward']}`",
        f"- Plot: `{plot_path}`",
    ]
    (artifacts_dir / "training_run_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
