# OpenEnv Hackathon Compliance Audit

Last refreshed: `2026-04-26`

This audit reflects the repo state after the final deployment-cleanup pass.

## Verification Commands Run

- `python -m py_compile` on the environment modules
- `openenv validate hospital_council_env -v`
- `python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 1 --sample-size 100`
- `python run_openenv_demo.py --data-root physionet.org/files/mimiciv/3.1 --episodes 1 --sample-size 600 --record-path docs/evidence/demo_rollout_verified.jsonl`
- `python -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy baseline --json`
- `python -m hospital_council_env.training.evaluate_policy --data-root physionet.org/files/mimiciv/3.1 --episodes 20 --sample-size 1000 --policy random --json`
- synthetic fallback smoke test via `HospitalCouncilEnvironment(data_root="missing-mimic-root")`

## Minimum Requirements Status

- OpenEnv usage: PASS
  - Evidence: [hospital_council_env/openenv.yaml](../hospital_council_env/openenv.yaml)
- OpenEnv latest release requirement: PASS as of `2026-04-26`
  - Project pins `openenv-core[core]==0.2.3`
  - PyPI showed `0.2.3` as the latest `openenv-core` release on `2026-04-26`
- Deployable Hugging Face Space: READY
  - Root `README.md` now has Space metadata
  - Root `Dockerfile` now builds the environment directly
  - Public Space publish step is still manual
- Minimal training script using HF TRL or Unsloth: PASS
  - Evidence: [hospital_council_env/training/hf_trl_grpo_minimal.py](../hospital_council_env/training/hf_trl_grpo_minimal.py)
- README linking local materials: PASS
  - Evidence links now resolve to real files under [docs/evidence](evidence)
- Evidence of measurable improvement: PASS for baseline-vs-random
  - Trained-vs-untrained from a real RL run is still recommended for stronger judging performance
- Public explainer assets: MANUAL
  - Space URL, blog link, and short video link still need to be added after publication

## Important Deployment Note

Because licensed MIMIC-IV data cannot be bundled into a public Space, the environment now supports two runtime modes:

- `mimic_bootstrap`
  - uses real MIMIC-derived seeds when `MIMIC_DATA_ROOT` is available
- `synthetic_bootstrap`
  - automatically activates when no dataset mount is present, so the Space still boots and remains interactive

This keeps the public deployment runnable without breaking the private-data-backed research workflow.

## Judging Criteria Coverage

- Environment Innovation (40%): STRONG
  - Multi-agent stakeholders, long-horizon phase plan, coalition dynamics, retrieval-style augmentation, and contextual correction signals
- Storytelling (30%): STRONG
  - Root README, package README, and demo script now tell a cleaner and more accurate submission story
- Showing Improvement in Rewards (20%): GOOD
  - Clear baseline-vs-random separation is included
  - A real post-training curve would still strengthen the score further
- Reward and Training Pipeline (10%): STRONG
  - OpenEnv rubric composition plus a minimal TRL environment-factory entrypoint

## Quantitative Evidence Snapshot

See:

- [docs/evidence/metrics_baseline.json](evidence/metrics_baseline.json)
- [docs/evidence/metrics_random.json](evidence/metrics_random.json)
- [docs/evidence/reward_comparison.md](evidence/reward_comparison.md)
- [docs/evidence/demo_rollout_verified.jsonl](evidence/demo_rollout_verified.jsonl)

Highlights from 20 episodes each:

- Baseline average reward: `4.3860`
- Random average reward: `3.4400`
- Baseline success rate: `0.7500`
- Random success rate: `0.0000`
- Baseline phase-action accuracy: `1.0000`
- Random phase-action accuracy: `0.1795`
- Baseline category accuracy: `1.0000`
- Random category accuracy: `0.2717`
- Baseline average task-graph loss: `0.0342`
- Random average task-graph loss: `0.4754`

## Current Risks

1. The repo is deployable, but the final public Space, video, and blog URLs still need to be inserted after publishing.
2. The baseline remains weakest on `discharge_negotiation`; that is the main environment-quality tuning target left.
3. For maximum judging strength, add plots from a real training run instead of relying only on baseline-vs-random evidence.
