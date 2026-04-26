---
title: Hospital Council OpenEnv
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - multi-agent
  - long-horizon
---

# Hospital Council OpenEnv

Hospital Council OpenEnv is a long-horizon, multi-agent environment for the Meta x Hugging Face OpenEnv Hackathon. The agent acts as a hospital council coordinator and has to manage partially observed patient state, stakeholder incentives, coalition drift, and late-stage execution pressure across a multi-step episode.

The main submission lives in `hospital_council_env/`. The older `mimic_openenv.py` file is only a legacy baseline reference and is not the environment that should be deployed or judged.

## Why this fits the hackathon

- **Theme 1: Multi-Agent Interactions**. The agent negotiates across an attending physician, triage nurse, pharmacist, bed manager, and family liaison.
- **Theme 2: Long-Horizon Planning**. Episodes have phased structure, delayed credit, and failure modes that appear only after several steps.
- **Theme 3.1: Professional World Modeling**. The environment models changing clinical and operational state, not a static puzzle or label task.
- **Judging fit**. The environment uses dense OpenEnv rubric scores, has a minimal TRL training entrypoint, and exposes enough structure to show measurable reward improvement.

## Environment Summary

Each episode is assigned to one of four scenario families:

- `diagnostic_ambiguity`
- `medication_alignment`
- `conservative_monitoring`
- `discharge_negotiation`

The agent acts through five structured moves:

- `consult`
- `propose`
- `delegate`
- `resolve`
- `commit`

Observations include:

- mission brief
- patient snapshot
- stakeholder updates
- visible conflicts
- retrieved analogies
- task-graph state and loss
- `web_augmentation`
- `context_observation`
- long-horizon goals
- scoreboard metrics

Rewards are composed with OpenEnv rubrics:

- `milestone`
- `coalition`
- `safety`
- `efficiency`
- `terminal`
- `task_graph`

## Data and Deployment Mode

The environment is designed to use licensed MIMIC-IV tables when `MIMIC_DATA_ROOT` is available. For Hugging Face Spaces and other public demos where private MIMIC files cannot be bundled, it now falls back automatically to a synthetic bootstrap encounter set that preserves the same action space, reward flow, and scenario families.

That means:

- local research runs can use real MIMIC-derived seeds
- the public Space can still boot and run without private data
- judges can interact with the environment immediately after deployment

The active data source is exposed in the observation under `patient_snapshot.data_source`.

## Hugging Face Deployment

This repo is now deployable from the repo root as a Docker Space.

Files that matter for Space deployment:

- `README.md`: Hugging Face Space metadata and project overview
- `Dockerfile`: root Docker build for the Space
- `hospital_council_env/openenv.yaml`: OpenEnv manifest
- `hospital_council_env/server/app.py`: FastAPI entrypoint

If you want the Space to use real MIMIC data instead of synthetic bootstrap mode, add a Space secret or runtime variable named `MIMIC_DATA_ROOT` and mount the licensed dataset in that path. Otherwise the Space will run in synthetic mode automatically.

## Local Validation

Use the repo venv:

```bash
.\.venv\Scripts\openenv.exe validate hospital_council_env -v
.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 4 --sample-size 1000
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy baseline
```

Run the server locally:

```bash
Set-Location hospital_council_env
..\.venv\Scripts\python.exe -m hospital_council_env.server.app
```

Then check the real client-server loop:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_remote_client --base-url http://localhost:8000 --episodes 5
```

## Training

Minimal TRL entrypoint:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.hf_trl_grpo_minimal --model Qwen/Qwen3-0.6B
```

This follows the official `environment_factory` pattern from the Hugging Face TRL OpenEnv docs.

## Evidence

Local evidence tracked in the repo:

- Baseline metrics JSON: [docs/evidence/metrics_baseline.json](docs/evidence/metrics_baseline.json)
- Random metrics JSON: [docs/evidence/metrics_random.json](docs/evidence/metrics_random.json)
- Comparison summary: [docs/evidence/reward_comparison.md](docs/evidence/reward_comparison.md)
- Verified demo rollout: [docs/evidence/demo_rollout_verified.jsonl](docs/evidence/demo_rollout_verified.jsonl)
- Demo script: [docs/demo_video_script.md](docs/demo_video_script.md)
- Compliance audit: [docs/requirement_audit_2026-04-25.md](docs/requirement_audit_2026-04-25.md)

## Final Submission TODOs

These are still manual because they depend on your published assets:

- add the final Hugging Face Space URL
- add the mini-blog URL or short demo video URL
- add reward/loss plots from a real training run
- add trained-vs-untrained rollout evidence once that run is complete

## References

- OpenEnv docs: https://meta-pytorch.org/OpenEnv/
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
