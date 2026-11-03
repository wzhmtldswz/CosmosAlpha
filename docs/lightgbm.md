# Daily LightGBM configuration

See [lightgbm_integration.example.toml](../configs/lightgbm_integration.example.toml)
for a complete example. This page covers `train-lightgbm`, which reads a single
Graph mining run, recalculates its factors, then trains and evaluates the model.
See the [shared configuration reference](configuration.md) for data directories
and field mappings.

The model, preprocessing, filtering, label, backtest, and parallel settings in the
tables have loader defaults. Data, universe, source location, output location,
and Qlib configuration must still be supplied in full. Write floating-point
parameters as `0.0`, `1.0`, and so on; do not replace all floats with integers.

## Factor source and default admission

| Setting | Example or default value | Meaning and constraints |
|---|---|---|
| `schema_version` | `lightgbm-integration-config/1.2` | Fixed: daily model configuration schema version. |
| `factor_source.storage_root` | `runs` | Required: mining results root. Input is read from `<storage_root>/<run_id>/`. |
| `factor_source.run_id` | `replace-with-graph-run-id` | Required: replace with an actual Graph run ID, or override with the CLI's `--source-run-id`. |
| `factor_source.include_rejected_factors` | `true`, optional | Read both accepted and rejected factors. `false` uses accepted factors only. Failed factors never enter training. |

Correlation filtering is disabled by default. All usable factors are included
in stable `factor_id` order, without admission checks based on RankIC, factor
correlation, valid-date ratios, or RankIC coverage. A factor is excluded only if
recalculation fails, no feature is produced, or no eligible training sample has
a finite feature value after preprocessing. Each exclusion reason is saved. An
explicit error is raised if no factors are usable; uncomputed selection RankIC
values are recorded as `null`.

The source must be a Graph run. Mining must end no later than the end of model
validation and before model testing starts. The source and model must agree on
dataset declarations, market, frequency, price adjustment, calendar, universe,
field mappings, region, and explicit stock list. Provider paths may differ when
data has been relocated.

## Time segments and data

All date ranges are inclusive. Training, validation, and testing must occur in
that order without overlap.

| Setting | Example value | Meaning |
|---|---|---|
| `segments.train.start_date` | `2010-04-01` | Training start, also the start of the factor admission interval. |
| `segments.train.end_date` | `2019-05-31` | Training end. |
| `segments.valid.start_date` | `2019-06-01` | Validation start; validation selects the number of rounds through early stopping. |
| `segments.valid.end_date` | `2020-12-31` | Validation end. |
| `segments.test.start_date` | `2021-01-01` | Start of out-of-sample evaluation. |
| `segments.test.end_date` | `2025-12-31` | End of out-of-sample evaluation. |
| `data.start_date` | `2010-04-01` | Must equal the training start. |
| `data.end_date` | `2025-12-31` | Must equal the test end. |
| `data.warmup_start_date` | `2010-01-04` | Start of historical warmup for rolling factor calculations, no later than the training start. |

Other `data.*`, `universe.*`, and `qlib.*` fields follow the data declarations,
policies, and field mappings in the [shared configuration reference](configuration.md).
The model entry point requires `provider = "qlib"`, `market = "cn"`,
`frequency = "1d"`, and `qlib.enabled = true`.

Training and validation rows require an available label and at least one finite
feature. Missing values in individual features are handled by the model. Test
labels do not affect factor admission, fitting, or early stopping. Test rows with
usable features can still receive predictions when their labels are missing.

## Model feature preprocessing

By default, each day's features follow this sequence: raw expression
recalculation → eligible-sample handling → MAD clipping → industry and continuous
style neutralization → cross-sectional Z-score → model feature matrix.

| Setting | Default value | Meaning and constraints |
|---|---|---|
| `feature_preprocessing.neutralization_enabled` | `true` | Tunable: whether to neutralize features before modeling. `false` still applies clipping and standardization. |
| `feature_preprocessing.mad_multiplier` | `5.0` | Tunable: cross-sectional MAD clipping multiplier, a finite float >0. This model-side field is passed to the preprocessing function. |
| `feature_preprocessing.min_cross_section_samples` | `100` | Tunable: minimum valid cross-sectional samples per day for preprocessing, an integer ≥2. Still applies when filtering is disabled. |
| `feature_preprocessing.protocol` | Selected automatically from the neutralization switch | Usually omitted. Uses `model_feature_neutralized_v1` when enabled and `model_feature_mad_zscore_v1` when disabled. An explicit value must agree with the switch. |
| `feature_preprocessing.industry_field` | `industry_id` | Fixed: industry field, omitted from the public example. |
| `feature_preprocessing.continuous_exposures` | `["log_mkt_cap", "ret20", "turn20", "vol20"]` | Fixed: continuous exposure list, omitted from the public example. |

Neutralization regresses on industry fixed effects and continuous exposures,
then standardizes the residuals. Insufficient samples or incomplete exposures
produce missing features, without falling back to raw, unneutralized values.
Disabling neutralization does not remove the entry point's exposure and mask
data prerequisites.

## Optional correlation filtering

| Setting | Default value | Meaning and constraints |
|---|---|---|
| `correlation_filter.enabled` | `false` | Tunable: enable RankIC ranking and greedy correlation filtering within the training interval. |
| `correlation_filter.threshold` | `0.8` | Only active when filtering is enabled: exclude a factor when its absolute mean daily Pearson correlation with a selected factor reaches this value, a float in `(0,1]`. |
| `correlation_filter.min_daily_effective_stock_ratio` | `0.8` | Only active when filtering is enabled: minimum daily valid-stock ratio for RankIC and correlation calculations, a float in `[0,1]`. |
| `correlation_filter.min_effective_date_ratio` | `0.8` | Only active when filtering is enabled: minimum valid-date ratio for those statistics, a float in `[0,1]`. |

When enabled, factors are sorted by descending absolute training RankIC, with
factor ID breaking ties, then checked against already selected factors.
Uncomputable statistics or insufficient coverage also produce recorded exclusion
reasons. The three thresholds are omitted from the public example and do not
affect admission while filtering is disabled.

## Fixed label protocol

The entire `[label]` section may be omitted. The loader supplies these fixed
values:

| Setting | Fixed default | Meaning |
|---|---|---|
| `label.return_definition` | `close_t1_over_close_t` | Shared daily forward close-return convention. |
| `label.horizon_days` | `1` | Daily label horizon. Changing this to another integer does not switch the period. |
| `label.normalization` | `qlib_cs_rank_norm_v1` | Qlib cross-sectional rank normalization for learning labels. Raw returns are retained for signal evaluation. |

## Model hyperparameters

Defaults follow the Qlib Alpha158/CSI300 LightGBM benchmark settings, with up to
1000 boosting rounds and 50 rounds of early-stopping patience. Training uses the
mined factors; Alpha158 features are not added.

| Setting | Default value | Meaning and constraints |
|---|---|---|
| `model.loss` | `mse` | Fixed: mean squared error regression objective. |
| `model.colsample_bytree` | `0.8879` | Fraction of features sampled per tree, a float in `(0,1]`. |
| `model.learning_rate` | `0.2` | Learning rate per boosting round, a finite float >0. |
| `model.subsample` | `0.8789` | Row-sampling fraction passed to LightGBM, a float in `(0,1]`. The corresponding sampling frequency is not enabled; see below. |
| `model.lambda_l1` | `205.6999` | L1 regularization strength, a nonnegative finite float. |
| `model.lambda_l2` | `580.9768` | L2 regularization strength, a nonnegative finite float. |
| `model.max_depth` | `8` | Maximum depth per tree, a positive integer. The project configuration does not accept the upstream `-1` convention. |
| `model.num_leaves` | `210` | Maximum leaves per tree, an integer ≥2. LightGBM's own constraints also apply. |
| `model.num_threads` | `4` | Model training threads, a positive integer. |
| `model.early_stopping_rounds` | `50` | Rounds without validation metric improvement before early stopping, a positive integer. |
| `model.num_boost_round` | `1000` | Maximum boosting rounds, a positive integer. Early stopping may reduce the actual count. |

Including all factors describes the input feature set. `colsample_bytree` still
samples columns within each tree. `subsample` is an alias for `bagging_fraction`;
LightGBM requires `bagging_freq > 0` to enable row sampling. The current model
setup does not set that frequency, whose upstream default is `0`. Changing
`subsample` alone therefore does not enable row sampling. The strict
configuration does not currently expose `bagging_freq` either. This behavior
follows the [LightGBM 4.6 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#bagging_freq)
and the project's Qlib model setup.

## Daily backtest and costs

The executor makes daily decisions using TopkDropout. Daily decisions do not
require replacing every holding. Evaluation uses absolute returns: reported net
returns deduct actual trading costs without subtracting a market benchmark.

| Setting | Default value | Meaning and constraints |
|---|---|---|
| `backtest.account` | `100000000` | Initial account balance, a positive integer. |
| `backtest.return_mode` | `absolute` | Fixed: absolute-return evaluation. |
| `backtest.topk` | `50` | Target number of holdings, a positive integer. |
| `backtest.n_drop` | `5` | Planned number of stocks to replace, a positive integer. Actual trades depend on candidates, holdings, and trading constraints. |
| `backtest.hold_thresh` | `1` | Minimum holding period in daily trading steps, a positive integer. Does not change decision frequency. |
| `backtest.only_tradable` | `false` | Whether to construct candidates only from tradable stocks. `false` still respects the trading engine's execution restrictions. |
| `backtest.forbid_all_trade_at_limit` | `true` | Whether to prohibit both buy and sell trades at price limits, enforced by the Qlib strategy and exchange. |
| `backtest.risk_degree` | `0.95` | Target fraction of capital invested, a finite float in `(0,1]`. |
| `backtest.limit_threshold` | `0.095` | Daily return threshold used by Qlib to identify price limits, a float in `[0,1]`. This is not a transaction fee. |
| `backtest.open_cost` | `0.0` | Buy-side cost rate, a nonnegative finite float. |
| `backtest.close_cost` | `0.0` | Sell-side cost rate, a nonnegative finite float. |
| `backtest.min_cost` | `0` | Minimum cost per trade, a nonnegative integer. |
| `backtest.impact_cost` | `0.0` | Market impact cost coefficient, a nonnegative finite float. |
| `backtest.trade_unit` | `100` | Trading lot size, a positive integer. Actual quantities follow Qlib execution rules. |
| `backtest.benchmark` | Empty string | Fixed, omitted from the public example. The entry point uses a zero benchmark to satisfy Qlib's report structure. |
| `backtest.method_sell` | `bottom` | Fixed, omitted from the public example: sell lower-signal stocks first. |
| `backtest.method_buy` | `top` | Fixed, omitted from the public example: buy higher-signal stocks first. |
| `backtest.deal_price` | `open` | Fixed, omitted from the public example: execute at the opening price. |

All four costs are zero in the example, omission defaults, and actual arguments
passed to the engine. Changing `n_drop` or `hold_thresh` does not switch this
entry point to monthly backtesting.

## Parallelism and output

| Setting | Default or example value | Meaning and constraints |
|---|---|---|
| `parallel.factor_workers` | `1`, optional | Factor recalculation workers, a positive integer. Values above 1 use separate processes. |
| `parallel.native_threads` | `1`, fixed and optional | Numerical-library thread limit inside each factor worker. Separate from model training threads. |
| `output.root` | `runs`, required | Root directory for model runs. Results go to `<root>/<model-run-id>/`; an existing directory must not be overwritten. |

More factor workers increase memory use. `model.num_threads` controls model
threads. The mining configuration's `runtime.max_workers` does not control this
model entry point.

## Command-line overrides

CLI overrides are applied after loading the configuration. The resulting values
are saved in the configuration snapshot.

| Argument | Overrides |
|---|---|
| `--source-run-id ID` | `factor_source.run_id` |
| `--correlation-filter enabled/disabled` | `correlation_filter.enabled` |
| `--neutralization enabled/disabled` | Neutralization switch and its matching protocol |
| `--num-threads N` | `model.num_threads` |
| `--factor-workers N` | `parallel.factor_workers` |
| `--include-rejected-factors` | Sets `include_rejected_factors` to `true`. For accepted-only input, set it to `false` in TOML. |
| `--zero-cost` | Forces all four costs to zero; they are already zero by default. |
| `--preflight-only` | Checks local dependencies, source, and configuration without recalculating features or training. |

```sh
python -m cosmosalpha train-lightgbm --config configs/lightgbm_integration.example.toml --source-run-id mine-001 --run-id model-001 --preflight-only
python -m cosmosalpha train-lightgbm --config configs/lightgbm_integration.example.toml --source-run-id mine-001 --run-id model-001
```

Implementation references: [configuration defaults and validation](../src/cosmosalpha/modeling/config.py),
[recalculation, selection, training, and backtesting](../src/cosmosalpha/modeling/lightgbm.py),
and [feature preprocessing](../src/cosmosalpha/evaluation/preprocessing.py).
