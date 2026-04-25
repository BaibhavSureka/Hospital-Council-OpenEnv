# Hospital Council OpenEnv

This repo now centers on `hospital_council_env/`, a real OpenEnv package built for the Meta x Hugging Face OpenEnv Hackathon rather than a one-off single-agent demo.

The core idea is stronger than the earlier baseline: the model is not just predicting a label, it is acting as a hospital council coordinator across a long-horizon, partially observed episode. It has to manage conflicting incentives from an attending physician, triage nurse, pharmacist, bed manager, and family liaison while steering a MIMIC-seeded patient case toward a safe outcome.

## Why this fits the judging criteria

- **Environment Innovation**: this is a multi-agent hospital operations world, not a grid clone or a simple classifier loop.
- **Storytelling**: every episode has a clear narrative arc: sensemaking, alignment, execution, conflict resolution, handoff.
- **Improvement in Rewards**: the environment exposes dense rubric scores plus a terminal success component, so reward curves and before/after rollouts are easy to show.
- **Reward & Training Pipeline**: the reward is rubric-composed, scenario-balanced, and paired with a minimal TRL training script.

## Themes Covered

- **Theme 1: Multi-Agent Interactions**
  The coordinator negotiates with multiple stakeholders whose incentives diverge.
- **Theme 2: Long-Horizon Planning**
  Episodes unfold across several phases with delayed terminal credit.
- **Theme 3: World Modeling**
  The hidden patient trajectory depends on actions, coalition support, and safety constraints.
- **Theme 4: Dynamic Monitoring**
  The environment exposes an active task graph and monitoring loss so rollouts show whether each action is reducing phase-level planning error.
- **Theme 5: External Knowledge**
  Optional Serper search enriches query-action pairs with external evidence and maps those signals back onto prior trajectories.

## What Changed

- `hospital_council_env/`: new official OpenEnv package scaffold and implementation.
- `hospital_council_env/server/hospital_council_env_environment.py`: OpenEnv server environment.
- `hospital_council_env/simulator.py`: MIMIC-derived scenario sampler, long-horizon phase plan, stakeholder dynamics, historical retrieval.
- `hospital_council_env/augmentation.py`: optional Serper web-search augmentation and trajectory overlap signals.
- `hospital_council_env/rubrics.py`: composable reward logic using OpenEnv rubrics.
- `hospital_council_env/training/hf_trl_grpo_minimal.py`: minimal Hugging Face TRL script.
- `run_openenv_demo.py`: top-level local demo entrypoint for the new environment.

`mimic_openenv.py` is kept as the earlier baseline simulator and reference point, but it is no longer the main submission shape.

## Environment Summary

Each episode samples a MIMIC-derived encounter and maps it into one of four balanced scenario families:

- `diagnostic_ambiguity`
- `medication_alignment`
- `conservative_monitoring`
- `discharge_negotiation`

The agent acts through structured moves:

- `consult`
- `propose`
- `delegate`
- `resolve`
- `commit`

The observation is only partially observed and includes:

- mission brief
- visible patient snapshot
- stakeholder updates
- visible conflicts
- retrieved analogies from historical failures
- task-graph monitoring loss
- web augmentation signals
- long-horizon goals
- scoreboard metrics

The hidden state tracks:

- per-step milestone targets
- stakeholder alignment
- diagnostic clarity
- medication progress
- discharge readiness

## Reward Design

The environment uses OpenEnv rubrics, not one monolithic score.

Subscores:

- `milestone`: did the action fit the current long-horizon phase?
- `coalition`: are stakeholders aligning or drifting apart?
- `safety`: did the agent avoid risky premature actions?
- `efficiency`: is it making progress without redundant loops?
- `terminal`: did the episode land in a strong final state?
- `task_graph`: did the action reduce the active task-graph monitoring loss?

These are combined in `HospitalCouncilRubric` with a weighted sum.

## Dynamic Augmentation

Every step now appends two signals to the structured observation:

- `task_graph`: active phase node, expected action/category/targets, and `task_graph_loss`.
- `web_augmentation`: action description, valid use cases, supporting evidence, and overlap with archived trajectories.

Set `SERPER_API_KEY` to enable live Serper search. Without a key, the environment emits deterministic offline evidence from the internal stage graph so the demo remains reproducible.

## Quick Start

Use the repo venv:

```bash
.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 4 --sample-size 1000
```

Record a JSONL rollout for the demo video:

```bash
.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 2 --sample-size 1000 --record-path artifacts/demo_rollout.jsonl
```

Run full local metrics:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy baseline
```

That prints:

- average reward
- success rate
- phase-action accuracy
- category accuracy
- milestone hit rate
- coalition threshold rate
- average task-graph loss
- web-augmented step rate
- per-scenario averages

Validate the environment package:

```bash
.\.venv\Scripts\openenv.exe validate hospital_council_env -v
```

Run the local OpenEnv server:

```bash
cd hospital_council_env
..\.venv\Scripts\uv.exe run server
```

If PowerShell dislikes the spacing, use:

```bash
Set-Location hospital_council_env
..\.venv\Scripts\python.exe -m hospital_council_env.server.app
```

Then test the real server-client loop end to end from the repo root:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_remote_client --base-url http://localhost:8000 --episodes 5
```

## Training

Minimal TRL script:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.hf_trl_grpo_minimal --model Qwen/Qwen3-0.6B
```

That script follows the official `environment_factory` pattern from TRL’s OpenEnv integration docs and exposes environment methods as tools.

## End-to-End Test Order

If you want the full local test flow in the simplest order:

1. `.\.venv\Scripts\openenv.exe validate hospital_council_env -v`
2. `.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 4 --sample-size 1000`
3. `.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy baseline`
4. Start the server in `hospital_council_env/`
5. `.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_remote_client --base-url http://localhost:8000 --episodes 5`

## Submission Checklist

Before final hackathon submission, fill these in:

- Hugging Face Space URL: `TODO`
- Hugging Face mini-blog URL: `TODO`
- Short demo video URL: `TODO`
- Reward curve image path or WandB link: `TODO`
- Before/after rollout examples: `TODO`

Use [docs/demo_video_script.md](docs/demo_video_script.md) as the recording outline.

## Validation Notes

This package was locally checked with:

- `python -m py_compile` on the environment package
- `openenv validate hospital_council_env -v`
- local demo rollouts via `python -m hospital_council_env.training.run_local_demo`

## References

- OpenEnv docs: https://meta-pytorch.org/OpenEnv/
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
