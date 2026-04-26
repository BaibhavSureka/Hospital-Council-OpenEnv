# Hospital Council OpenEnv

## Problem

Large language models often do reasonably well on short medical-style reasoning prompts, but they are weaker when the task becomes multi-agent, long-horizon, and operational. Real hospital coordination is not just diagnosis. It is negotiation, sequencing, coalition management, and safe handoff under partial observability.

This project targets that gap with an OpenEnv environment where the model acts as a hospital council coordinator rather than a single-step classifier.

## Environment

The environment is built around four scenario families:

- `diagnostic_ambiguity`
- `medication_alignment`
- `conservative_monitoring`
- `discharge_negotiation`

The agent interacts through five structured actions:

- `consult`
- `propose`
- `delegate`
- `resolve`
- `commit`

Each episode unfolds across multiple phases and requires the model to coordinate five stakeholders with different incentives:

- attending physician
- triage nurse
- pharmacist
- bed manager
- family liaison

The observation includes partial patient context, visible conflicts, stakeholder updates, task-graph state, retrieval-style augmentation, and contextual corrective feedback.

## Why This Matters

This environment fits the OpenEnv hackathon goals because it teaches more than shallow pattern matching. The model has to:

- track hidden progress over several turns
- avoid premature or unsafe commitments
- maintain alignment across multiple agents
- recover when the current plan starts drifting

That makes it useful for:

- multi-agent interaction training
- long-horizon planning
- professional world modeling

## Reward Design

The reward is rubric-based and intentionally dense enough to support training:

- `milestone`
- `coalition`
- `safety`
- `efficiency`
- `terminal`
- `task_graph`

This gives the model feedback on both local correctness and episode-level outcome quality.

## Data Strategy

The local research version can use licensed MIMIC-IV-derived encounter seeds and lab signals. The public Hugging Face Space cannot bundle private clinical data, so the deployed environment automatically falls back to a synthetic bootstrap mode with the same task structure and action flow.

That keeps the Space runnable for judges while preserving the intended environment behavior.

## Current Evidence

Tracked evidence in this Space includes:

- baseline metrics
- random-policy metrics
- reward comparison summary
- verified rollout JSONL

In the current evaluation snapshot, the baseline policy clearly outperforms a random policy in reward, success rate, and task-graph alignment.

## What Judges Should Look At

1. Open the app and inspect the multi-step interaction flow.
2. Review the evidence files under `docs/evidence/`.
3. Compare baseline vs random behavior.
4. Check how the environment exposes phase structure, coalition shaping, and contextual guidance rather than only terminal pass/fail scoring.

## Limitations and Next Step

The main remaining improvement area is stronger post-training evidence from a real RL run, especially on the `discharge_negotiation` scenario family, which is the weakest case in the current baseline.

Even so, the environment itself is already complete, deployable, OpenEnv-compliant, and suitable for long-horizon multi-agent training experiments.
