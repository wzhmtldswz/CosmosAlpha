# Graph parameters

See [graph_experiment.toml](../configs/graph_experiment.toml) for a complete example
and the [configuration reference](configuration.md) for shared fields. Every
`[graph]` field on this page must be supplied explicitly. An example value does
not imply that the field is optional; fixed fields cannot be changed under the
current protocol.

## Initialization

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.schema_version` | `graph-config/2.0` | Fixed: Graph configuration protocol version. |
| `graph.initialization.seed_mode` | `cold` | Tunable: `cold`, `warm`, or `auto`, as described below. |
| `graph.initialization.seed_pool` | `huatai_v1` | Alias for a pool of fixed seed expressions. The public built-in pool accepts `huatai_v1` or `huatai`. Cold starts do not use it to generate initial nodes. |
| `graph.initialization.initial_node_count` | `2` | Tunable: target number of initial nodes, a positive integer. The actual count depends on generation, deduplication, and technical admission. |
| `graph.initialization.random_seed` | `20260823` | Tunable: nonnegative integer for internal Graph choices such as seed sampling and tie handling. It does not guarantee reproducible LLM responses. |

- `cold`: generate initial factors with the LLM, without using existing candidates as initial nodes.
- `warm`: prefer compatible existing candidates, then try the configured fixed seed expressions if needed. Fails if no usable node is available.
- `auto`: prefer compatible existing candidates, then use the LLM to fill any shortfall.

Built-in fixed seeds are recalculated, evaluated, and technically validated.
`seed_pool` does not automatically import old run results. Existing candidate
pools must be supplied by the runtime; the external high-quality pool is empty by
default.

## Search size and concurrency

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.search.parents_per_round` | `4` | Tunable: maximum parents selected per round, a positive integer. Also limited by eligible nodes and the remaining global budget. |
| `graph.search.max_expansion_attempts` | `100` | Tunable: global limit on parent expansion attempts, a positive integer. Counts requests, including failed attempts; initialization is counted separately. |
| `graph.search.max_expansions_per_node` | `5` | Tunable: maximum times one node can be expanded as a parent, a positive integer. |

`parents_per_round` sets the planned number of expansions in a batch. The shared
`runtime.max_workers` setting limits how many can run concurrently.
`workflow.max_cycles` does not control Graph search length. Search stops when the
global budget is reached, all nodes reach their expansion limit, or no eligible
parent remains. Expansion attempts are not the number of successful factors or
LLM requests.

Use this snippet to reduce the scale of an existing configuration:

```toml
[graph.search]
parents_per_round = 2
max_expansion_attempts = 10
max_expansions_per_node = 3
```

## Structural and correlation neighborhoods

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.structure.edge_threshold` | `0.2` | Tunable: minimum Jaccard similarity between canonical expression subtree sets, a float in `[0,1]`. Raising it generally reduces structural edges. |
| `graph.structure.max_neighbors` | `30` | Tunable: number of local top-K structural neighbors selected per node, a positive integer. |
| `graph.correlation.edge_threshold` | `0.1` | Tunable: minimum absolute correlation between factor values, a float in `[0,1]`. Uses daily cross-sectional Pearson correlations and checks valid-date coverage. |
| `graph.correlation.max_neighbors` | `30` | Tunable: number of local top-K correlation neighbors selected per node, a positive integer. |

Local top-K neighborhoods are merged symmetrically, so a node's final undirected
degree may exceed `max_neighbors`. Reducing neighborhood size does not eliminate
all underlying correlation or structural calculations. These thresholds construct
the search graph; they are separate from factor-library acceptance thresholds
and LightGBM correlation filtering.

## Novelty reference pool

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.average_distance.knn_k` | `10` | Tunable: number of nearest reference neighbors used for novelty, a positive integer. Uses the available count when fewer than K exist. |
| `graph.average_distance.minimum_abs_rank_ic` | `0.02` | Tunable: absolute RankIC threshold for reference-pool factors, a nonnegative finite float. Independent of the shared acceptance threshold. |
| `graph.average_distance.external_pool_capacity` | `30` | Tunable: capacity of the external high-quality reference pool, a positive integer. Does not create or download external factors automatically. |

AverageDistance is the mean of the structural KNN distance and correlation KNN
distance to the reference set. Structural distance is `1 - Jaccard`; correlation
distance is `1 - abs(correlation)`. Larger values indicate greater difference from
the reference set. A component with no usable reference neighbors contributes 0.

## Fixed quality normalization

Absolute RankIC, RankIC standard deviation, and Top20 return are clipped and
normalized using fixed anchors, then averaged into SelfQuality. Lower RankIC
standard deviation yields higher quality, reversing the direction used for
returns and absolute RankIC. All fields below are fixed by the current v2
protocol; the anchors cannot be changed through configuration.

| Setting | Fixed value | Meaning |
|---|---|---|
| `graph.quality.normalization_version` | `graph-quality-normalization/2.0` | Normalization protocol identifier. |
| `graph.quality.rank_ic.lower` | `0.0` | Lower anchor for the RankIC quality component. |
| `graph.quality.rank_ic.upper` | `0.05` | Upper anchor for the RankIC quality component. |
| `graph.quality.rank_ic_std.lower` | `0.05` | Better-end anchor for RankIC standard deviation. |
| `graph.quality.rank_ic_std.upper` | `0.09` | Worse-end anchor for RankIC standard deviation. |
| `graph.quality.top20_return.lower` | `0.0` | Lower anchor for the Top20 return component. |
| `graph.quality.top20_return.upper` | `0.0008` | Upper anchor for Top20 return, expressed as a decimal: `0.0008` equals `0.08%`. |

Technical admission to the Graph differs from factor-library `accepted` status.
A `rejected` factor with complete metrics and sufficient coverage may still
become a search node.

## Fertility and Flow

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.fertility.top_q` | `3` | Fixed: use the Top-3 Mean of a parent's historical expansion contributions. |
| `graph.fertility.use_birth_time_delta_hv` | `true` | Fixed: use each child's hypervolume increment relative to the quality archive at its birth. Later archive changes do not recompute this historical contribution. |
| `graph.flow.gamma` | `0.75` | Tunable: neighbor propagation weight, a float satisfying `0 ≤ gamma < 1`. `0.0` retains only each node's own reward. |
| `graph.flow.iterations` | `100` | Tunable: maximum numerical iterations per Flow solve, a positive integer. This is not the number of mining expansions. |
| `graph.flow.convergence_tolerance` | `1e-8` | Tunable: stopping threshold for the largest elementwise change between consecutive Flow iterations, a nonnegative finite float. |
| `graph.flow.modality_weights.structure` | `0.3333333333333333` | Fixed: mixture weight for the structural neighborhood. |
| `graph.flow.modality_weights.correlation` | `0.3333333333333333` | Fixed: mixture weight for the correlation neighborhood. |
| `graph.flow.modality_weights.generation` | `0.3333333333333333` | Fixed: mixture weight for generation relationships. |

The three weights must be equal and sum to 1. Each modality is row-normalized
separately before forming the transition matrix `P`. Flow solves each reward
component in `(SelfQuality, Fertility, AverageDistance)`:

```text
F_next = (1 - gamma) * R + gamma * P * F
```

`iterations` and `convergence_tolerance` control the accuracy and cost of this
numerical iteration. They do not limit factor generation.

## Parent selection

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `graph.selector.name` | `flow_guided_graph` | Fixed: identifier of the current selector implementation. |
| `graph.selector.use_pareto_fronts` | `true` | Fixed: rank nodes into Pareto fronts using propagated reward vectors. |
| `graph.selector.tie_breaker` | `crowding_distance` | Fixed: crowding-distance ordering within a front. |
| `graph.selector.crowding_epsilon` | `1e-12` | Tunable: numerical stability threshold for crowding-distance calculation, a nonnegative finite float. |

Example values match this repository's public Graph configuration and are not
automatically tuned. Use a new run ID after changing parameters. With the same
configuration and inputs, `--resume` recovers from committed rounds.

Implementation references: [configuration validation](../src/cosmosalpha/pipeline/graph/config.py),
[controller](../src/cosmosalpha/pipeline/graph/controller.py),
[rewards](../src/cosmosalpha/pipeline/graph/reward.py),
[Flow](../src/cosmosalpha/pipeline/graph/flow.py), and
[selector](../src/cosmosalpha/pipeline/graph/selector.py).
