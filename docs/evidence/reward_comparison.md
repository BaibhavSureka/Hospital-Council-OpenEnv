# Reward Comparison

This snapshot compares the shipped heuristic baseline against a random policy over 20 episodes each on `2026-04-26`.

## Headline metrics

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

## Readout

The environment clearly separates purposeful policy behavior from noisy behavior. The baseline follows the intended phase plan almost perfectly and reaches successful terminal states in most runs, while the random policy fails to complete any successful episode.

The weakest current scenario remains `discharge_negotiation`, where the baseline still underperforms the other three scenario families. That is now a tuning problem inside coalition shaping and late-stage handoff behavior rather than a packaging or environment-validity problem.
