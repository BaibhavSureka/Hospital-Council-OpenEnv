# Demo Video Script

Target length: 90 to 120 seconds.

## Opening

Hospital Council OpenEnv is a long-horizon, multi-agent hospital coordination environment seeded from MIMIC-IV encounter statistics. The agent is not just predicting a label. It has to coordinate an attending physician, triage nurse, pharmacist, bed manager, and family liaison across a partially observed episode.

## Dynamic Environment Moment

Show one rollout step and point to three live signals:

- the current phase in the task graph
- the task-graph loss, which drops when the action matches the active milestone
- the coalition, safety, and terminal reward components

This makes the environment feel dynamic because the reward is tied to progress through an evolving graph, not a static one-step classifier.

## Web Augmentation Moment

Show the `web_status` and `web_valid_cases` fields in the recorded rollout. With `SERPER_API_KEY` configured, the environment calls Serper for external evidence about the query-action pair. Without a key, it still emits deterministic offline evidence from the internal stage graph so the demo is reproducible.

## Training Story

Run the evaluator and show:

- average reward
- success rate
- phase-action accuracy
- category accuracy
- average task-graph loss
- web-augmented step rate

## Commands To Record

```bash
.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 2 --sample-size 1000 --record-path artifacts/demo_rollout.jsonl
```

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy baseline
```

Optional live web evidence:

```bash
$env:SERPER_API_KEY="your_key_here"
.\.venv\Scripts\python.exe run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 1 --sample-size 1000 --record-path artifacts/demo_rollout_serper.jsonl
```
