# MIMIC Open Gym Decision Environment

This repository provides a Gym-style environment for next-step clinical decision simulation on MIMIC-IV data.

Core files:

- `mimic_openenv.py`: environment, reward logic, metrics, helper utilities.
- `run_openenv_demo.py`: demo runner and evaluation example.

The environment is designed for fast iteration workflows:

- Uses sampled admissions for lightweight episodes.
- Accepts both discrete and structured action payloads.
- Validates medication actions against medication text seen in MIMIC tables.
- Builds LLM-ready step payloads and accepts a pluggable observer callback.

## Decision Space

The action categories are:

1. `diagnosis` (id 0)
2. `medication` (id 1)
3. `discharge` (id 2)
4. `no_action` (id 3)

`action_space = Discrete(4)`.

Actions can be either:

- Integer action id (`0..3`), or
- Dictionary payload, for example:

```python
{
    "category": "medication",
    "tester_prompt": "give vitamin",
    "agent_response": "give vitamin"
}
```

Supported dictionary keys include `category`/`category_id`, `tester_prompt`, `agent_response`, and `action_text`.

## Observation Space

`observation_space = Box(shape=(16,), dtype=float32)`.

Feature layout:

1. `los_hours / 24`
2. `in_icu`
3. `expired_flag`
4. `diag_count / 20`
5. `med_count / 20`
6. `proc_count / 20`
7. `transfer_count / 20`
8. `step_idx / max_steps`
9. `last_reward`
10. `invalid_medication_last_step`
11. `last_action_id / 3`
12. `age / 100`
13. `P(diagnosis)`
14. `P(medication)`
15. `P(discharge)`
16. `P(no_action)`

## Reward Logic

Per-step reward is category alignment plus medication consistency checks:

- `+1.0` if predicted category equals step ground truth, else `-0.5`.
- Additional medication bonus/penalty when action is `medication`:
  - `+0.4` if medication term is found in MIMIC medication index.
  - `-0.4` if not found.
- If `tester_prompt` implies a different medication term than the action term: `-0.2`.
- Step cost: `-0.01`.

Episode terminates when:

- `step_idx >= max_steps`, or
- action category is `discharge`.

## Data Requirements

Pass `data_root` as either:

- The folder containing `hosp/` and `icu/`, or
- A parent folder (the environment recursively searches for a valid root).

Minimum required table:

- `hosp/admissions.csv.gz` (must exist and be readable).

Additional tables enrich state and checks:

- `hosp/patients.csv.gz`
- `hosp/diagnoses_icd.csv.gz`
- `hosp/procedures_icd.csv.gz`
- `hosp/transfers.csv.gz`
- `icu/icustays.csv.gz`
- `hosp/prescriptions.csv.gz`
- `hosp/pharmacy.csv.gz`
- `hosp/emar.csv.gz`

The loader auto-detects true gzip files by signature, so files named `.csv.gz` that are plain-text CSV are also handled.

## Installation

Python 3.9+ recommended.

Install dependencies:

```bash
pip install numpy pandas requests gymnasium
```

If you use classic Gym instead of Gymnasium:

```bash
pip install gym
```

## Quick Start

Run the demo from repository root:

```bash
python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 3
```

Useful options:

- `--episodes`: number of demo episodes.
- `--sample-size`: admission sample size used to build records.
- `--data-root`: path to dataset root or its parent.

## Programmatic Usage

```python
from mimic_openenv import MIMICDecisionEnv

env = MIMICDecisionEnv(
    data_root="physionet.org/files/mimiciv/3.1",
    max_steps=6,
    sample_size=3000,
    trajectory_size=30,
)

obs, info = env.reset()

action = {
    "category": "medication",
    "tester_prompt": "give vitamin",
    "agent_response": "give vitamin"
}

next_obs, reward, terminated, truncated, info = env.step(action)
```

## LLM Observer Integration

You can pass `llm_observer` to override the heuristic observer.

The callback receives a payload with:

- Ground truth category for the current step.
- Current action content (`category`, `raw_action`, `tester_prompt`).
- Action latency (`time_taken_ms`).
- Recent trajectory (`fetch_trajectory`).
- Medication lookup results.
- Optional web search context.
- Current encounter state snapshot.

Expected callback return:

```python
{
    "observation": "text summary",
    "probabilities": {
        "diagnosis": 0.2,
        "medication": 0.5,
        "discharge": 0.2,
        "no_action": 0.1,
    },
}
```

Probabilities are normalized in the environment before being used.

## Evaluation Helpers

`mimic_openenv.py` includes:

- `evaluate_next_step_predictions(env, policy_fn, episodes=10)`
- `format_metrics(metrics)`

These compute:

- Accuracy
- Per-category precision/recall/F1/support
- Confusion matrix
- Average reward across episodes

The demo runner prints these metrics after episode rollouts.

## Troubleshooting

- `Path not found` or missing `hosp/` and `icu/`:
  - Verify `--data-root` points to the MIMIC folder or its parent.
- `No admissions data found`:
  - Ensure `hosp/admissions.csv.gz` exists and is readable.
- Slow loading on large extracts:
  - Lower `sample_size` and/or set `table_row_limit` in `MIMICDecisionEnv`.
- Web search skipped:
  - Install `requests` or provide a custom `web_search_fn`.

## Notes

- This project is a simulation environment for experimentation, not a clinical decision support system.
- Ground-truth labels are heuristic and generated from encounter-level signals.
