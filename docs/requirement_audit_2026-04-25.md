# OpenEnv Hackathon Compliance Audit (2026-04-25)

This audit evaluates the current codebase against the hackathon minimum requirements, judging criteria, and theme fit.

## Verification Commands Run

- `python -m compileall -q mimic_openenv.py run_openenv_demo.py hospital_council_env`
- `openenv validate hospital_council_env -v`
- `python -m hospital_council_env.training.evaluate_policy --episodes 20 --sample-size 1000 --policy baseline --json`
- `python -m hospital_council_env.training.evaluate_policy --episodes 20 --sample-size 1000 --policy random --json`
- `python run_openenv_demo.py --episodes 1 --sample-size 600 --record-path artifacts/demo_rollout_verified.jsonl`

## Minimum Requirements Status

- OpenEnv usage: PASS
  - Evidence: [hospital_council_env/openenv.yaml](../hospital_council_env/openenv.yaml)
  - Validation output: ready for multi-mode deployment.
- OpenEnv-compliant environment hosted on Hugging Face Spaces: BLOCKED (manual)
  - README still needs final Space URL.
- Minimal training script using HF TRL or Unsloth: PASS
  - Evidence: [hospital_council_env/training/hf_trl_grpo_minimal.py](../hospital_council_env/training/hf_trl_grpo_minimal.py)
- Mini-blog or <2 minute demo video: BLOCKED (manual)
  - README placeholders still need public links.
- README linking all materials: PARTIAL
  - Internal materials are linked; external final URLs are pending.
- Evidence of measurable improvement: PARTIAL
  - Baseline vs random policy metrics exist.
  - Real training curve from a training run is still needed for full compliance.

## Judging Criteria Coverage

- Environment Innovation (40%): STRONG
  - Multi-agent stakeholder model, phase graph, conflict handling, and rubric-composed rewards.
- Storytelling (30%): STRONG
  - Root README + package README + demo script provide clear narrative and demo flow.
- Showing Improvement in Rewards (20%): PARTIAL
  - Quantitative baseline vs random gap is available.
  - Trained-vs-untrained curves from an actual training run should be added.
- Reward and Training Pipeline (10%): STRONG
  - Composable rubrics + TRL environment-factory training script.

## Theme Fit

- Theme 1 (Multi-Agent Interactions): PASS
- Theme 2 ((Super) Long-Horizon Planning): PASS
- Theme 3.1 (Professional World Modeling): PASS
- Theme 3.2 (Personalized Tasks): NOT TARGETED
- Theme 4 (Self-Improvement): PARTIAL
- Theme 5 (Wild Card): OPTIONAL

## Quantitative Evidence Snapshot

See:

- [docs/evidence/metrics_baseline.json](evidence/metrics_baseline.json)
- [docs/evidence/metrics_random.json](evidence/metrics_random.json)
- [docs/evidence/reward_comparison.md](evidence/reward_comparison.md)
- [docs/evidence/demo_rollout_verified.jsonl](evidence/demo_rollout_verified.jsonl)

Highlights from 20 episodes each:

- Baseline average reward: 4.8400
- Random average reward: 3.2852
- Baseline success rate: 1.0000
- Random success rate: 0.0000
- Baseline task-graph loss: 0.1126
- Random task-graph loss: 0.5583

## Blocking Items Before Final Submission

1. Add real public links for:
   - Hugging Face Space
   - mini-blog (HF) or video (YouTube, <2 min)
   - reward curve image or run dashboard
2. Run at least one actual training session and export reward/loss curves.
3. Add trained-vs-untrained behavior comparison in README using those artifacts.
4. Confirm Space URL is final and publicly accessible before submission deadline.
