# CosmosAlpha

[English](README.md) | [中文](README_zh-CN.md)

**Paper:** [arXiv:2611.04555](https://arxiv.org/abs/2611.04555)

**CosmosAlpha** is an LLM-driven alpha-mining framework that organizes the factor
pool as a **multi-relational graph** of generation lineage, expression-structure
similarity, and cross-sectional exposure correlation. **Graph Flow** propagates
search utilities across the graph, while **Pareto-guided selection** balances
quality, fertility, and diversity to guide exploration. The mined factors are
integrated with LightGBM for predictive and portfolio evaluation.

![CosmosAlpha framework overview](assets/flowchart.png)

## Install

Use Conda to create and activate a Python 3.12 environment:

~~~sh
conda create -n cosmosalpha -c conda-forge python=3.12 pip -y
conda activate cosmosalpha
~~~

If the `cosmosalpha` environment already exists, activate it without recreating
it. From the cloned repository, choose a regular installation:

~~~sh
python -m pip install ".[model]"
~~~

Or an editable installation:

~~~sh
python -m pip install -e ".[model]"
~~~

The model extra supplies Qlib, pandas, and LightGBM. For an offline smoke check
without these optional dependencies, install the core with
`python -m pip install -e .`.

## Data

Download [data.zip](https://zenodo.org/records/23857833/files/data.zip?download=1)
from Zenodo and extract it locally. Point the three provider paths in your mining
and LightGBM configurations to the corresponding extracted directories:

| Provider | Example directory | Required contents |
|---|---|---|
| Market | `~/data/cn_data/qlib_bin` | open, high, low, close, volume, vwap, turn, free_turn, returns |
| Exposure | `~/data/cn_data/qlib_exposure_bin` | industry_id, log_mkt_cap, industry manifest |
| Mask | `~/data/cn_data/qlib_mask_bin` | universe_mask, eligible_mask, entry_tradable_mask, exit_tradable_mask |

The examples use CSI300. See [data requirements and limitations](docs/configuration.md#data-and-universe)
for provider details.

## Run

Activate the `cosmosalpha` environment and run these commands from the repository
root. Choose a mining configuration below, or make ignored `*.local.toml` copies.

Parameter meanings, defaults and scope: [configuration reference](docs/configuration.md).

### 1. Configure the LLM

In your mining configuration, replace the matching fields under `[llm]`
with one of the examples below; keep the remaining settings.

**DeepSeek (default):**

~~~toml
[llm]
provider = "openai-compatible"
model_alias = "deepseek-flash"
base_url = "https://api.deepseek.com"
~~~

**Another OpenAI-compatible service:**

~~~toml
[llm]
provider = "openai-compatible"
model_alias = "your-model-id"
base_url = "https://your-provider.example/v1"
~~~

Use your service's model ID and API base URL. The service must support
Chat Completions and JSON mode.

Set the API key in your shell:

Bash:

~~~sh
export COSMOSALPHA_LLM_API_KEY='your-api-key'
~~~

PowerShell:

~~~powershell
$env:COSMOSALPHA_LLM_API_KEY = 'your-api-key'
~~~

`COSMOSALPHA_LLM_BASE_URL` optionally overrides `base_url`.
`.env` files are not loaded automatically.

### 2. Mine factors

Choose one configuration for the `run --config` argument:

| Architecture | Configuration | Main controls |
|---|---|---|
| Graph | [graph_experiment.toml](configs/graph_experiment.toml) | `[graph]` search and selection settings |
| Chain | [chain_experiment.toml](configs/chain_experiment.toml) | `workflow.max_cycles` |
| Parallel branches | [parallel_branches_experiment.toml](configs/parallel_branches_experiment.toml) | `architecture.directions`, `runtime.max_workers`, cycles per branch |
| MCTS (single tree) | [mcts_experiment.toml](configs/mcts_experiment.toml) | `[mcts]` search rounds, exploration and dynamic budget |

For parallel branches, `workflow.max_cycles` applies to each direction.
MCTS search length is controlled by `[mcts]`.

Check the installation without data, an API key, or network requests:

~~~sh
python -m cosmosalpha --smoke-loop
~~~

After configuring data and the LLM, preflight and mine. These commands use Graph;
replace the configuration path to use another architecture:

~~~sh
python -m cosmosalpha run --config configs/graph_experiment.toml --run-id mine-001 --preflight-only
python -m cosmosalpha run --config configs/graph_experiment.toml --run-id mine-001
~~~

Mining reports stages, search progress, factor results, and retries to stderr;
the final JSON summary stays on stdout. Add `--log-level DEBUG` for details,
`WARNING` for warnings and errors, or `ERROR` for terminal errors only.
To save progress logs, append `2> mining.log` to the run command in Bash or PowerShell.

Preflight checks local prerequisites without calling the LLM. The full run uses
real data and LLM requests. The Graph example uses a cold start, up to 100
expansion attempts, and 4 workers. To use the built-in seed pool instead, set
`seed_mode = 'warm'` under `graph.initialization`; `seed_pool` is already
`huatai_v1`.

### 3. Train and evaluate

~~~sh
python -m cosmosalpha train-lightgbm --config configs/lightgbm_integration.example.toml --source-run-id mine-001 --run-id model-001 --preflight-only
python -m cosmosalpha train-lightgbm --config configs/lightgbm_integration.example.toml --source-run-id mine-001 --run-id model-001
~~~

LightGBM integrates factors from a Graph run for predictive and daily portfolio
evaluation. See [LightGBM configuration](docs/lightgbm.md) for preprocessing,
factor selection, and backtest settings.

The example uses mining and training data through May 2019, validation through
2020, and testing from 2021 to 2025. Keep these periods chronologically separate.

## Outputs and recovery

Outputs are stored in `runs/<run-id>/`. Mining publishes
`factor_library/factors.csv`; structured objects, calculated values, and workflow
state are kept alongside it. Model runs save the booster, predictions, signal
metrics, portfolio reports, and factor-exclusion records in their own directory.

Use a new run ID after changing configuration or data versions. Graph, Chain and
Parallel Branches can continue the same run with unchanged inputs using `--resume`.
MCTS starts a new search and requires a new run ID. Model runs also require a new
ID and refuse to overwrite an existing output directory.

## Checks and citation

~~~sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m black --check --workers 1 .
~~~

Tests use fixed inputs and mocks. Optional Qlib adapter checks are skipped when
model dependencies are absent. CI runs offline checks on Linux and Windows with
Python 3.12.

Citation metadata is in [CITATION.cff](CITATION.cff).
Source code is licensed under [MIT](LICENSE), copyright (c) 2026 Zhan-He Wang.
The paper and [framework figure](assets/flowchart.png) are excluded from this
software license. Separately obtained datasets retain their providers' terms.
