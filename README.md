# MIMIC-IV to SQLite

This repo contains a small loader that builds a single SQLite database from MIMIC-IV CSV exports.

It is aimed at the workflow from your chat:

- Kaggle demo dataset: https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2
- Full PhysioNet dataset: https://physionet.org/content/mimiciv/3.1/

The loader follows the official MIT-LCP MIMIC-IV Postgres table order and adapts it to SQLite so Kaggle can preview the resulting `.sqlite` file directly.

Source references:

- Official create script: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/buildmimic/postgres/create.sql
- Official load script: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/buildmimic/postgres/load.sql

## Quick Start

Run this in a Kaggle notebook cell after adding the dataset:

```bash
!python build_mimic_iv_sqlite.py \
  --input /kaggle/input/mimic-iv-clinical-database-demo-2-2 \
  --output /kaggle/working/mimiciv_demo.sqlite \
  --create-indexes \
  --overwrite
```

For a full PhysioNet export, point `--input` to the folder that contains `hosp/` and `icu/`:

```bash
!python build_mimic_iv_sqlite.py \
  --input /kaggle/input/mimiciv-3-1 \
  --output /kaggle/working/mimiciv_3_1.sqlite \
  --create-indexes \
  --overwrite
```

The script accepts both plain `.csv` files and compressed `.csv.gz` files.

If your Kaggle dataset path contains one extra nesting level, that is fine. The script will search below the input folder until it finds a directory with both `hosp/` and `icu/`.

## What It Creates

Each source CSV becomes one SQLite table.

Examples:

- `hosp/admissions.csv` -> `mimiciv_hosp_admissions`
- `icu/chartevents.csv.gz` -> `mimiciv_icu_chartevents`
- `demo_subject_id.csv` -> `mimiciv_meta_demo_subject_id`

The script also creates `mimiciv_load_log` with the loaded source path, row count, and elapsed time for each table.

## Handy Options

Use a smaller smoke test first:

```bash
!python build_mimic_iv_sqlite.py \
  --input /kaggle/input/mimic-iv-clinical-database-demo-2-2 \
  --output /kaggle/working/mimiciv_demo.sqlite \
  --limit-rows 10000 \
  --overwrite
```

Important flags:

- `--create-indexes`: adds a small set of common lookup indexes after loading.
- `--overwrite`: replaces an existing SQLite file.
- `--batch-size 20000`: useful if memory is tight.
- `--sample-size 5000`: samples more rows before inferring SQLite affinities.

## Query Example

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("/kaggle/working/mimiciv_demo.sqlite")

pd.read_sql_query(
    "SELECT COUNT(*) AS patients FROM mimiciv_hosp_patients",
    conn,
)
```

## Notes

- SQLite does not support Postgres schemas, so the schema name is folded into the table name.
- This is for raw table loading, not derived MIMIC concepts.
- The script is intended for exploration and notebook workflows, especially on Kaggle where a single SQLite artifact is easy to inspect.

## Open Gym Environment (Next Step Decision)

This workspace now includes a schema-aware Open Gym style environment for next-step clinical decision making:

- `mimic_openenv.py`
- `run_openenv_demo.py`

Decision categories:

1. `diagnosis`
2. `medication`
3. `discharge`
4. `no_action`

Environment capabilities:

- Loads data from MIMIC `hosp/` and `icu/` tables.
- Accepts action payloads from a tester/agent (`category`, `tester_prompt`, `agent_response`).
- Validates medication terms against MIMIC medication fields (`prescriptions`, `pharmacy`, `emar`).
- Tracks trajectory history for each step.
- Performs optional web lookup for medication context.
- Builds a single payload that includes:
  - ground truth
  - current action and time taken
  - fetched trajectory
  - web search information
- Sends that payload to an observer function (LLM-compatible) to compute observations/probabilities.

Run demo:

```bash
python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 3
```

Notes:

- The default observer is heuristic so the environment runs without API keys.
- You can pass your own `llm_observer` callback to `MIMICDecisionEnv` for model-based probabilities.
