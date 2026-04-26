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

`hospital_council_env` is the deployable OpenEnv package for the Hospital Council submission. It is a long-horizon, multi-agent environment where the model coordinates a hospital council under partial observability.

## Core Idea

The agent manages five stakeholders:

- attending physician
- triage nurse
- pharmacist
- bed manager
- family liaison

Episodes are organized into four scenario families:

- `diagnostic_ambiguity`
- `medication_alignment`
- `conservative_monitoring`
- `discharge_negotiation`

## Why it is useful for training

- **Multi-agent interactions**. Stakeholders carry different incentives and different levels of alignment.
- **Long-horizon planning**. The best move depends on earlier consultation, coalition shaping, and execution timing.
- **World modeling**. Hidden clinical and operational state evolves as actions are taken.
- **Dense supervision**. Task-graph state, rubric scores, and context feedback give a rich learning signal.

## Action Space

The coordinator can make five structured moves:

- `consult`
- `propose`
- `delegate`
- `resolve`
- `commit`

Actions can also specify:

- `target`
- `category`
- `medication`
- `message`
- `confidence`

## Observation Space

Each step returns:

- mission brief
- patient snapshot
- stakeholder updates
- visible conflicts
- retrieved analogies
- `task_graph`
- `web_augmentation`
- `context_observation`
- long-horizon goals
- scoreboard metrics

## Reward

Rewards are built with OpenEnv rubrics and combine:

- milestone fit
- coalition support
- safety
- efficiency
- terminal success
- task-graph score

## Retrieval and Feedback

The environment adds two structured guidance layers:

- `web_augmentation`: a self-contained LLM-style pseudo-search and ranking layer
- `context_observation`: classification, confidence, correction, and next-step guidance

This keeps the environment self-contained while still exposing the kind of reasoning scaffolding that helps with long-horizon agent training.

## Data Mode

When `MIMIC_DATA_ROOT` points to licensed MIMIC-IV data, the simulator samples MIMIC-derived encounter seeds and lab signals from:

- `hosp/labevents.csv.gz`
- `hosp/d_labitems.csv.gz`

When that data is unavailable, the environment falls back automatically to a synthetic bootstrap set so a public Hugging Face Space can still run. The active source is surfaced as `patient_snapshot.data_source`.

## Local Development

Validate the package:

```bash
openenv validate . -v
```

Run the server:

```bash
uv run server
```

Run a local demo rollout:

```bash
python -m hospital_council_env.training.run_local_demo --data-root ../physionet.org/files/mimiciv/3.1
```

Record a rollout for demo evidence:

```bash
python -m hospital_council_env.training.run_local_demo --data-root ../physionet.org/files/mimiciv/3.1 --record-path ../docs/evidence/demo_rollout_verified.jsonl
```

## Minimal TRL Training

The package includes:

```bash
python -m hospital_council_env.training.hf_trl_grpo_minimal
```

This follows the official `environment_factory` pattern from the Hugging Face TRL OpenEnv docs.

## Evidence

- Baseline metrics JSON: [../docs/evidence/metrics_baseline.json](../docs/evidence/metrics_baseline.json)
- Random metrics JSON: [../docs/evidence/metrics_random.json](../docs/evidence/metrics_random.json)
- Comparison summary: [../docs/evidence/reward_comparison.md](../docs/evidence/reward_comparison.md)
- Verified rollout JSONL: [../docs/evidence/demo_rollout_verified.jsonl](../docs/evidence/demo_rollout_verified.jsonl)
- Compliance audit: [../docs/requirement_audit_2026-04-25.md](../docs/requirement_audit_2026-04-25.md)

## Manual Submission Assets

Add these after publishing:

- final Hugging Face Space URL
- blog post URL or short demo video URL
- reward/loss plots from a real training run
