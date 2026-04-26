---
title: Hospital Council OpenEnv
sdk: docker
app_port: 8000
---

# Hospital Council OpenEnv

Hospital Council OpenEnv is a long-horizon, multi-agent environment for the Meta x Hugging Face OpenEnv India Hackathon 2026. The agent plays the role of a hospital council coordinator and must manage partial information, stakeholder incentives, coalition drift, and late execution pressure across a multi-step episode.

The hackathon submission environment lives in `hospital_council_env/`. The older `mimic_openenv.py` file is only a legacy baseline reference and is not the artifact that should be deployed or judged.

## Submission Links

Public submission URLs:

- Hugging Face Space: `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv`
- Public training notebook: `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv/blob/main/Hospital_Council_OpenEnv_Colab_Training.ipynb`
- Public blog post: `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv/blob/main/Blog.md`

## Why This Fits The Hackathon

- Multi-agent interaction: the agent negotiates across an attending physician, triage nurse, pharmacist, bed manager, and family liaison.
- Long-horizon planning: episodes have phased structure, delayed credit, and failure modes that appear only after several steps.
- Professional world modeling: the environment models changing clinical and operational state rather than a static puzzle.
- RL training fit: the environment exposes dense OpenEnv rubric scores and a minimal Hugging Face TRL GRPO entrypoint.

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

## Data Mode

When `MIMIC_DATA_ROOT` is available, local research runs can use licensed MIMIC-IV-derived data. In public deployments such as Hugging Face Spaces, the environment automatically falls back to a synthetic bootstrap encounter set with the same action space, reward flow, and scenario families.

That gives us:

- local runs that can use licensed clinical data
- a public Space that can boot without private files
- a judge-friendly environment that is immediately runnable after deployment

The active source is surfaced in `patient_snapshot.data_source`.

## Hugging Face Space Deployment

This repo is set up to be pushed directly as a Docker Space from the repo root.

Files that matter most:

- `README.md`
- `Dockerfile`
- `hospital_council_env/openenv.yaml`
- `hospital_council_env/server/app.py`

Useful endpoints after deployment:

- `/` shows the Space landing page
- `/web` shows the same landing page for Space path probes
- `/status`
- `/health`

If you want the Space to use licensed MIMIC data instead of synthetic mode, configure `MIMIC_DATA_ROOT` as a Space runtime variable or secret and mount the dataset there. Otherwise the public Space works in synthetic mode automatically.

## Local Validation

Use the repo virtual environment:

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

Then validate the real client-server loop:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.evaluate_remote_client --base-url http://localhost:8000 --episodes 5
```

## Training

Minimal Hugging Face TRL GRPO entrypoint:

```bash
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m hospital_council_env.training.hf_trl_grpo_minimal --model Qwen/Qwen3-0.6B
```

The Windows UTF-8 environment variable is important because the current TRL plus `transformers` tool-calling stack reads template files that otherwise break under the default Windows codepage.

To generate compact training evidence artifacts after a run:

```bash
.\.venv\Scripts\python.exe -m hospital_council_env.training.plot_training_run --output-dir outputs/grpo_hospital_council --artifacts-dir docs/evidence/training_run
```

Expected evidence outputs:

- `docs/evidence/training_run/grpo_training_curves.png`
- `docs/evidence/training_run/training_run_summary.json`
- `docs/evidence/training_run/training_run_summary.md`

## Evidence

Current repo evidence:

- Project write-up: [Blog.md](Blog.md)
- Training notebook: [Hospital_Council_OpenEnv_Colab_Training.ipynb](Hospital_Council_OpenEnv_Colab_Training.ipynb)
- Baseline metrics: [docs/evidence/metrics_baseline.json](docs/evidence/metrics_baseline.json)
- Random metrics: [docs/evidence/metrics_random.json](docs/evidence/metrics_random.json)
- Reward comparison: [docs/evidence/reward_comparison.md](docs/evidence/reward_comparison.md)
- Verified demo rollout: [docs/evidence/demo_rollout_verified.jsonl](docs/evidence/demo_rollout_verified.jsonl)
- Demo script: [docs/demo_video_script.md](docs/demo_video_script.md)
- Submission audit: [docs/requirement_audit_2026-04-25.md](docs/requirement_audit_2026-04-25.md)
- Form answer template: [docs/submission_form_answers.md](docs/submission_form_answers.md)

The only missing evidence artifact in-git is the final GRPO reward/loss plot export from a completed training run. The plotting pipeline is now included in the repo, and the generated files should be committed under `docs/evidence/training_run/` once the run finishes.

## Submission Checklist

- OpenEnv-based environment included in `hospital_council_env/`
- Working RL training script in `hospital_council_env/training/hf_trl_grpo_minimal.py`
- Public re-runnable notebook in `Hospital_Council_OpenEnv_Colab_Training.ipynb`
- Public blog option prepared in `Blog.md`
- Docker Space deployment path included at repo root
- README contains the three submission URL placeholders and supporting artifact links
- Real reward/loss plots can be generated with `plot_training_run.py` after the GRPO run completes

## Final Manual Steps

Only three external publication steps remain before form submission:

1. Confirm the Space opens publicly at `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv`.
2. Confirm the notebook file is visible publicly at `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv/blob/main/Hospital_Council_OpenEnv_Colab_Training.ipynb`.
3. Confirm `Blog.md` is visible publicly at `https://huggingface.co/spaces/BAIBHAV1234/hospital-council-openenv/blob/main/Blog.md`.

After those links are public, submit the same three URLs in the Google Form.

## References

- OpenEnv docs: https://meta-pytorch.org/OpenEnv/
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
