# MCTS parameters

See [mcts_experiment.toml](../configs/mcts_experiment.toml) for a complete example
and the [configuration reference](configuration.md) for shared fields. This page
covers single-tree MCTS through the main `run` entry point:

```sh
python -m cosmosalpha run --config configs/mcts_experiment.toml --run-id mcts-001 --preflight-only
python -m cosmosalpha run --config configs/mcts_experiment.toml --run-id mcts-001
```

Preflight does not call the LLM. A full run uses the configured data and LLM
service. These parameters do not control the legacy `--mcts-live` or
`--mcts-multi` entry points.

## Five search parameters

You may omit the entire `[mcts]` section or supply only the fields you want to
override. The loader fills in the following defaults and saves the complete
resolved settings in the run's configuration snapshot.

| Setting | Default when omitted | Meaning and constraints |
|---|---|---|
| `mcts.max_iterations` | `50` | Maximum outer search rounds, an integer ≥1. Each round selects a parent and attempts multiple candidates; this is not a cap on total factors or API requests. |
| `mcts.exploration_weight` | `1.0` | Weight of the UCT exploration term, a nonnegative finite float. Higher values favor less-visited branches. |
| `mcts.initial_budget` | `3` | Dynamic budget at the start of search after the root is established, an integer ≥1. |
| `mcts.budget_increment` | `3` | Budget added when the breakthrough condition is met, a nonnegative integer. `0` disables only this type of budget replenishment. |
| `mcts.breakthrough_relax` | `0.95` | Breakthrough ratio, a finite float in `(0,1]`. Replenishes the budget when a candidate's overall score reaches the historical best multiplied by this ratio. Lower ratios make the condition easier to meet. |

These values retain the code defaults of the main single-tree entry point before
migration. They do not use the separate experimental settings of the legacy
multi-tree live entry point. The example provides a baseline run configuration,
not a standardized experiment budget or tuned result.

## Rounds, candidates, and budget

The root is generated first, followed by at most `max_iterations` outer search
rounds. Normal rounds attempt two candidates; strong-factor branches or deep-dive
rounds may attempt three. Search may end early when the budget is exhausted.

For a normal candidate that successfully creates a child without a strong-factor
bonus, budget accounting works as follows:

```text
No breakthrough: 3 - 1 = 2
Breakthrough:   3 - 1 + budget_increment(3) = 5
```

The breakthrough condition uses `overall_score`; it does not compare RankIC
directly with `0.95`. The historical best never decreases. A candidate reaching
95% of that best can also replenish the budget.

The controller also retains strong-factor branch bonuses. When a parent's or
child's absolute RankIC exceeds the internal threshold of `0.04`, additional
budget may be granted, and some branches do not deduct the normal candidate
cost. Failed candidates are not charged as successful children either. Therefore,
`budget_increment = 0` does not guarantee a strictly decreasing budget, and
`initial_budget` is not a limit on LLM calls. `max_iterations` still bounds the
outer rounds.

## Configuration example

This snippet limits search to at most 10 rounds and disables normal breakthrough
budget replenishment:

```toml
[mcts]
max_iterations = 10
exploration_weight = 1.0
initial_budget = 3
budget_increment = 0
breakthrough_relax = 0.95
```

`workflow.max_cycles` does not control MCTS search length, and
`runtime.max_workers` does not parallelize expansion in the current single-tree
controller. Root seed constraints, scoring weights, strong-factor rules, and
deep-dive rules remain internal; these five fields do not switch them.

MCTS starts a new search on each invocation rather than continuing from a saved
tree; use a new `--run-id`. The saved tree and configuration snapshot can be used
to inspect the run. The current LightGBM integration accepts only Graph runs.

Implementation references: [configuration](../src/cosmosalpha/core/config.py),
[runtime parameter wiring](../src/cosmosalpha/pipeline/runtime.py), and
[MCTS controller](../src/cosmosalpha/pipeline/mcts/controller.py).
