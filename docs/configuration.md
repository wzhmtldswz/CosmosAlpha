# Configuration reference

This page explains the shared settings for all four mining architectures, along
with the controls for Chain and Parallel Branches. Architecture-specific settings
are documented in [Graph](graph.md), [MCTS](mcts.md), and [LightGBM](lightgbm.md).

| Architecture | Complete example | Main budget |
|---|---|---|
| Graph | [graph_experiment.toml](../configs/graph_experiment.toml) | `[graph.search]` |
| Chain | [chain_experiment.toml](../configs/chain_experiment.toml) | `workflow.max_cycles` |
| Parallel Branches | [parallel_branches_experiment.toml](../configs/parallel_branches_experiment.toml) | Number of directions × cycles per direction |
| MCTS, single tree | [mcts_experiment.toml](../configs/mcts_experiment.toml) | `[mcts]` |

## Reading and editing the configuration

The example values in the tables come from the public TOML files; they are not
claimed to be optimal. Shared fields are required unless explicitly described as
optional. All Graph-specific fields must also be supplied. See the MCTS and
LightGBM pages for their defaulting rules.

**Tunable** means the main entry point reads the field. **Declaration** means it
records a data or calculation convention. **Reserved** means the main entry point
does not currently use it. Keep declaration and reserved fields at their example
values: changing their names or numbers does not switch algorithms. Validators
may also restrict fixed protocol fields to a single value.

Unknown fields cause an error. Integer and floating-point types are checked
strictly; for example, `0.0` and `0` are not freely interchangeable. The TOML
snippets below replace the matching fields in an existing complete configuration.
Relative paths are resolved from the working directory; run from the repository
root. You can save local copies as Git-ignored `*.local.toml` files. Use a new
`--run-id` after changing the configuration or data. Each run saves a configuration
snapshot with secrets removed.

## Architecture and version

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `schema_version` | `experiment-config/1.6` | Configuration schema version, not an algorithm hyperparameter. The loader can fill in the new structure for older `1.5` TOML files. |
| `architecture.name` | `graph` | Tunable: `graph`, `chain`, `parallel_branches`, or `mcts`. Selects the controller used by the main `run` entry point. |
| `architecture.directions` | `[]` | Research directions for parallel branches. That architecture requires at least two distinct, nonempty strings; the other three architectures require `[]`. |

## Data and universe

Dates use `YYYY-MM-DD`, with both endpoints included. All four mining examples
use the same research interval: 2010-04-01 through 2019-05-31.

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `data.provider` | `qlib` | Data adapter. Use `qlib` for runs with real data; `mock` is for offline checks with fixed fixtures. |
| `data.dataset_version` | `qlib-cn-market-v2` | Dataset version declaration, also checked for consistency between mining and modeling. Renaming it does not transform the underlying data. |
| `data.price_adjustment` | `post_adjusted` | Price adjustment convention. Data must already follow it; changing this field does not readjust prices. |
| `data.start_date` | `2010-04-01` | Tunable: start of the calculation and evaluation interval. |
| `data.end_date` | `2019-05-31` | Tunable: end of the interval, no earlier than the start. When used for modeling, mining must not extend into the model test period. |
| `data.warmup_start_date` | `2010-01-04` | Tunable: start of the additional history read to warm up rolling operators; no later than `start_date`. |
| `data.market` | `cn` | Market identifier. The public workflow currently targets Chinese A-shares. |
| `data.frequency` | `1d` | Data frequency. The public factor pipeline currently uses daily data. |
| `data.trading_calendar_version` | `qlib-cn-market-v2-calendar` | Calendar version declaration for provenance and consistency checks. Calendar contents come from the provider. |
| `universe.name` | `csi300` | Tunable: universe name, corresponding to `instruments/<name>.txt` in each provider. It must match the mask data. |
| `universe.new_stock_min_age_years` | `1` | Minimum listing age policy, a nonnegative integer. Preflight checks the policy declared by the mask data; changing this value does not rebuild the masks. |

Each provider needs `calendars/`, `features/`, and `instruments/` directories. The
exposure directory also requires a completed `cosmosalpha_industry_manifest.json`.
The current industry data lacks strict point-in-time (PIT) certification;
limitations from missing first-publication timestamps and revision history still
apply.

## Return and preprocessing conventions

These fields record the fixed return and preprocessing conventions. Keep their
example values. For configurable model-side preprocessing, see
[LightGBM configuration](lightgbm.md).

| Setting | Example value | Current meaning |
|---|---|---|
| `returns.single_factor` | `close_t1_over_close_t` | Declaration: return from the current close to the next available close. |
| `returns.suspension_compression` | `clip_10_percent` | Declaration: clip returns across gaps in market data to ±10%. This clipping does not apply to adjacent valid dates. |
| `returns.horizon_days` | `1` | Declaration: daily forward-label convention. Changing it to another positive integer does not change the calculation horizon. |
| `preprocessing.protocol` | `gplearn_neutralized_v1` | Declaration: eligible-sample handling, MAD clipping, industry and style neutralization, and cross-sectional standardization. |
| `preprocessing.missing_policy` | `field_specific` | Declaration: missing values are handled by field and operator; this does not mean global zero-filling. |
| `preprocessing.rolling_min_fraction` | `0.6666666667` | Declaration: built-in rolling calculations require approximately two-thirds valid observations. This field does not dynamically change that convention. |
| `preprocessing.industry_field` | `industry_id` | Declaration: industry code field used for neutralization. |
| `preprocessing.continuous_exposures` | `["log_mkt_cap", "ret20", "turn20", "vol20"]` | Declaration: market capitalization and continuous style exposures derived from market data. |
| `preprocessing.mad_multiplier` | `5.0` | Declaration: mining preprocessing currently uses its own default of 5 times MAD. Changing this field does not change that parameter. |
| `preprocessing.rolling_std_ddof` | `1` | Declaration: rolling sample standard deviation convention. Cross-sectional standardization has its own fixed convention. |

## Operators and factor generation

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `operators.operator_set` | `OPERATOR_SET_BASE` | The main preflight check accepts only this built-in operator set. |
| `operators.registry_version` | `operator-registry/1.0` | Registry version declaration. The built-in registry is always used; changing the string does not load another operator set. |
| `operators.allowed_windows` | `[5, 10, 20]` | Built-in window declaration. The operator catalog is not rebuilt from this field. |
| `factor_generation.max_proposals` | `2` | Tunable: maximum candidates per design call; only `1` or `2` is allowed. Graph parent expansion also enforces its own single-candidate limit. |
| `factor_generation.max_expression_depth` | `15` | Reserved: this field is not passed through by the main entry point. The workflow uses the default depth of `8`; changing this setting does not increase the expression depth. |
| `factor_generation.max_validation_repair_attempts` | `2` | Tunable: repair attempts after expression validation fails, an integer from `0` to `2`. `0` disables repair. |
| `factor_generation.duplicate_policy` | `reject_existing_expression` | Deduplication convention. The workflow and architecture controllers perform deduplication; this field does not switch strategies. |

## Single-factor evaluation and acceptance

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `evaluation.protocol` | `rankic_neutralized_v1` | Evaluation alias accepted by the main preflight check. Keep the example value. |
| `evaluation.metrics` | RankIC mean, IR, standard deviation, and Top20 return | Metric declaration. The evaluator computes a fixed set; this list does not dynamically select metrics. |
| `evaluation.min_effective_sample_count` | `100` | Tunable: minimum cumulative valid sample count for acceptance, a positive integer. |
| `evaluation.min_cross_section_samples` | `100` | Tunable: minimum valid cross-sectional sample count per day, an integer ≥2. Passed to both mining preprocessing and evaluation. |
| `evaluation.acceptance_metric` | `rank_ic_mean` | Tunable: metric used for acceptance; it must exist in the evaluation results. The comparison checks a lower bound and does not automatically reverse direction based on the metric name. |
| `evaluation.acceptance_threshold` | `0.02` | Tunable: threshold for the main metric, a finite float. Acceptance also checks sample count and valid-date coverage. |
| `evaluation.acceptance_use_absolute_value` | `true` | Tunable: compare the absolute value of the main metric, allowing sufficiently strong negative correlations to pass. |
| `evaluation.min_daily_effective_stock_ratio` | `0.8` | Tunable: minimum ratio of valid samples to eligible stocks on each date, a float in `[0,1]`. |
| `evaluation.min_effective_date_ratio` | `0.8` | Tunable: minimum fraction of evaluation dates that are valid, a float in `[0,1]`. Graph technical admission and correlation coverage checks also use this value. |

`accepted` and `rejected` are factor-library acceptance states. Graph node
usability has separate technical requirements, so a `rejected` factor may still
enter the search graph. Modeling uses both states by default; see
[LightGBM factor admission](lightgbm.md).

## Cycles and prompts

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `workflow.max_cycles` | `1` | Tunable: total serial cycles for Chain, or cycles per direction for Parallel Branches; an integer ≥1. Graph and MCTS do not use it to control search length. |
| `workflow.max_retries` | `1` | Reserved: the main entry point does not currently use this field for step retries. It does not control LLM request retries. |
| `prompts.manifest` | `builtin:v2` | Built-in prompt manifest, located within the installed package independently of the working directory. Explicit external manifest paths are also supported. |
| `prompts.max_input_tokens` | `32000` | Tunable: input budget for rendering agent prompts, a positive integer. Uses the project's token estimate rather than the provider's exact billing count. |
| `prompts.max_history_items` | `4` | Tunable: maximum historical lessons included across Chain or branch cycles, a nonnegative integer. `0` includes none. Graph and MCTS use their own search context. |
| `prompts.max_proposals_per_call` | `2` | Budget declaration: only `1` or `2` is allowed, and it must not exceed `factor_generation.max_proposals`. It is not an independent control over the generated candidate count. |
| `prompts.max_format_retries` | `2` | Tunable: maximum retries for malformed live LLM output, a nonnegative integer. This is separate from expression repair. |

## LLM service and retries

See the [README](../README.md) for minimal examples of switching services.
Set `model_alias` to a model ID accepted by the service, and keep `provider` as the
compatible adapter alias.

| Setting | Example value or omission behavior | Meaning and constraints |
|---|---|---|
| `llm.provider` | `openai-compatible` | Supported aliases are `openai-compatible`, `openai`, and offline `mock`. The service must support Chat Completions and JSON mode. |
| `llm.model_alias` | `deepseek-flash` | Tunable: model ID sent as the API's `model` parameter. |
| `llm.base_url` | `https://api.deepseek.com` | Tunable: API base URL. The client appends `/v1/chat/completions`, or only `/chat/completions` if the URL already ends in `/v1`. |
| `llm.base_url_env` | `COSMOSALPHA_LLM_BASE_URL` | Name of the URL override environment variable. A nonempty value takes precedence over `base_url`. |
| `llm.organization` | `unused` | Placeholder for the optional organization header. No header is sent while this is `unused`; the field itself is still required. |
| `llm.api_key` | `unused` | API key placeholder. A real key supplied here takes precedence over the environment variable. Keep the placeholder and supply the key through the environment. |
| `llm.api_key_env` | `COSMOSALPHA_LLM_API_KEY` | Name of the environment variable containing the API key. `.env` files are not loaded automatically. |
| `llm.timeout_seconds` | `60` | Tunable: timeout in seconds for one HTTP request, a positive integer. |
| `llm.temperature` | `0.0` | Tunable: sampling temperature, a nonnegative finite float. The provider may impose additional constraints. |
| `llm.max_output_tokens` | `12000` | Tunable: output budget for pipeline agents, a positive integer. Graph parent design uses the larger of this value and the template budget; auxiliary MCTS calls may use internal budgets. |
| `llm.max_timeout_retries` | `2` | Tunable: maximum retries for timeout errors, a nonnegative integer. |
| `llm.max_rate_limit_retries` | `3` | Tunable: maximum retries for rate-limit errors, a nonnegative integer. |
| `llm.max_server_retries` | `2` | Tunable: maximum retries for server errors; also used for transport errors and the default empty-response budget. |
| `llm.retry_initial_backoff_seconds` | `2.0` | Tunable: initial retry backoff in seconds, a nonnegative float. |
| `llm.retry_max_backoff_seconds` | `10.0` | Tunable: backoff cap, at least the initial value and at most 60 seconds. Actual delays include random jitter. |
| `llm.replay_only` | `false` | `true` uses fixed replay responses without calling the live service. It does not automatically replay an old run or supply market data. |
| `llm.reasoning_effort` | Optional; omitted from the request by default | Optional string: `none`, `low`, `medium`, `high`, `xhigh`, or `max`. Set it only if the provider supports it. |
| `llm.max_empty_retries` | Optional; inherits `max_server_retries` | Optional nonnegative integer overriding the empty-response retry limit. |

All error categories also share a total retry cap: the sum of the timeout,
rate-limit, server, and format retry limits. Increasing a category's budget does
not bypass that cap. LLM request retries are controlled by these settings and
`prompts.max_format_retries`, independently of `workflow.max_retries`.

## Storage and concurrency

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `storage.backend` | `local_sqlite` | Stores local artifacts, objects, and SQLite records for normal runs. `local_mock` is for offline fixtures. |
| `storage.root` | `runs` | Tunable: root directory for runs. Outputs go to `<root>/<run-id>/`. |
| `runtime.max_workers` | Graph: `4`; Parallel Branches: `2`; others: `1` | Tunable: concurrency limit for Graph batches and parallel branches, a positive integer. It does not parallelize serial Chain or the current single-tree MCTS. |
| `runtime.random_seed` | `20260823` | Reserved: the main runtime does not currently use this field to seed global randomness. Graph uses its own `graph.initialization.random_seed`. |

Higher concurrency increases memory use and concurrent LLM requests. Branch
concurrency cannot exceed the number of directions. These settings do not control
LightGBM threads or model-side factor workers; see [model configuration](lightgbm.md).

## The three Qlib providers

| Setting | Example value | Meaning and constraints |
|---|---|---|
| `qlib.enabled` | `true` | Must be enabled when using Qlib data. |
| `qlib.region` | `cn` | Qlib region parameter; it must match the market. |
| `qlib.instruments` | `[]` | An empty list uses the universe file. A nonempty list explicitly overrides the stock codes read, without replacing mask eligibility constraints. |
| `qlib.market.provider_uri` | `~/data/cn_data/qlib_bin` | Market data provider path. |
| `qlib.exposure.provider_uri` | `~/data/cn_data/qlib_exposure_bin` | Industry and market capitalization exposure provider path. |
| `qlib.mask.provider_uri` | `~/data/cn_data/qlib_mask_bin` | Provider path for universe, eligibility, and tradability masks. |
| `qlib.market.fields[].field_name` | For example, `close` | Internal market field name; it must match the built-in field catalog. |
| `qlib.market.fields[].expression` | For example, `$close` | Qlib expression paired with the internal field name in the same array item. |
| `qlib.exposure.fields[].field_name` | `industry_id`, `log_mkt_cap` | Internal exposure field name. |
| `qlib.exposure.fields[].expression` | For example, `$industry_id` | Qlib expression read from the exposure provider. |
| `qlib.mask.fields[].field_name` | For example, `eligible_mask` | Internal mask field name. |
| `qlib.mask.fields[].expression` | For example, `$eligible_mask` | Qlib expression read from the mask provider. |

`fields` is a nonempty array; each item must supply `field_name` and `expression`.
The market example contains open, high, low, close, volume, vwap, turn, free_turn,
and returns. The exposure example contains industry_id and log_mkt_cap; the other
style exposures are derived from market data. The masks are universe_mask,
eligible_mask, entry_tradable_mask, and exit_tradable_mask. On Windows, use paths
such as `C:/data/qlib_bin` to avoid backslash escaping.

## Chain and Parallel Branches examples

One Chain cycle covers hypothesis generation, factor design, expression
generation, calculation, evaluation, and feedback. Replace these fields in an
existing Chain configuration to run five serial cycles:

```toml
[workflow]
max_cycles = 5
max_retries = 1
```

For parallel branches, the list length determines the number of directions.
Cycles within each direction still run serially:

```toml
[architecture]
name = "parallel_branches"
directions = ["Short-term price reversal", "Medium-term price momentum"]

[workflow]
max_cycles = 3
max_retries = 1

[runtime]
max_workers = 2
random_seed = 20260823
```

This example has two directions with three cycles each and runs at most two
branches concurrently. Cycle count is not the final factor count or LLM request
count: candidates may fail, duplicate existing expressions, or be rejected. See
the Graph and MCTS pages for their search budgets.

See [core/config.py](../src/cosmosalpha/core/config.py) for parameter types and
loading, and [pipeline/runtime.py](../src/cosmosalpha/pipeline/runtime.py) for how
settings are passed through and used.
