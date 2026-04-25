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

`hospital_council_env` is a long-horizon, multi-agent environment for training LLMs to coordinate a hospital council under partial observability.

The model acts as a coordinator managing five stakeholders:

- attending physician
- triage nurse
- pharmacist
- bed manager
- family liaison

Episodes are seeded from real MIMIC-IV encounter statistics and organized into balanced scenario families:

- `diagnostic_ambiguity`
- `medication_alignment`
- `conservative_monitoring`
- `discharge_negotiation`

## Why this environment is interesting

This environment is designed for the OpenEnv Hackathon themes rather than just being a generic benchmark:

- **Multi-agent interactions**: stakeholders have different incentives and hidden beliefs.
- **Long-horizon planning**: the right move depends on what happened several turns earlier.
- **World modeling**: the patient trajectory, coalition support, and safety state all evolve over time.
- **Dynamic monitoring**: each episode exposes an active task graph and a task-graph loss that changes after every action.
- **External knowledge**: optional Serper search turns web evidence into structured query-action signals.

## Action Space

The coordinator can make five structured moves:

- `consult`
- `propose`
- `delegate`
- `resolve`
- `commit`

Each action can also specify:

- `target`
- `category`
- `medication`
- `message`
- `confidence`

## Observation Space

Each step returns a partially observed view containing:

- mission brief
- visible patient snapshot
- stakeholder updates
- visible conflicts
- retrieved analogies from historical failures
- task-graph monitoring state
- web augmentation signals
- long-horizon goals
- scoreboard metrics

The hidden state tracks:

- milestone plan
- coalition support
- diagnostic clarity
- medication progress
- discharge readiness

## Reward

Rewards are built with OpenEnv rubrics and combine:

- milestone fit
- coalition support
- safety
- efficiency
- terminal success
- task-graph score

This makes the reward signal dense enough for training while still keeping a meaningful delayed component.

## Augmentation

The observation includes `task_graph` and `web_augmentation` fields. `task_graph.loss` monitors whether the current action matches the active phase node. `web_augmentation` contains action descriptions, valid use cases, supporting evidence, and trajectory overlaps.

Set `SERPER_API_KEY` to use live Serper search. Without it, the environment returns deterministic offline evidence from the internal stage graph.

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

Record rollout events for a demo video:

```bash
python -m hospital_council_env.training.run_local_demo --data-root ../physionet.org/files/mimiciv/3.1 --record-path ../artifacts/demo_rollout.jsonl
```

## Minimal TRL Training

The package includes:

```bash
python -m hospital_council_env.training.hf_trl_grpo_minimal
```

This follows the official `environment_factory` pattern from TRL’s OpenEnv integration.

## Submission Assets

Fill these before final submission:

- HF Space URL: `TODO`
- Blog post URL: `TODO`
- 2-minute video URL: `TODO`
- Reward curve artifact: `TODO`
- Demo recording JSONL: `TODO`
