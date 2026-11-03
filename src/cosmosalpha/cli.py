"""Command-line entry point for CosmosAlpha."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import replace
from tempfile import TemporaryDirectory

from cosmosalpha import __version__
from cosmosalpha.core import CoreValidationError, load_experiment_config, to_json
from cosmosalpha.data import run_qlib_snapshot_smoke
from cosmosalpha.pipeline.llm_smoke import run_live_llm_smoke
from cosmosalpha.pipeline.mcts.live import run_mcts_live, run_mcts_live_multi
from cosmosalpha.pipeline.mcts.smoke import run_mcts_smoke
from cosmosalpha.pipeline.preflight import (
    PreflightIssue,
    PreflightReport,
    run_preflight,
)
from cosmosalpha.pipeline.runtime import run_formal_runtime
from cosmosalpha.pipeline.smoke import (
    run_fixed_replay_multicycle_smoke_loop,
    run_fixed_replay_smoke_loop,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="cosmosalpha",
        description="Multi-agent research system for expression-based factors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--smoke-loop",
        action="store_true",
        help="run the fixed-data Replay end-to-end smoke loop",
    )
    parser.add_argument(
        "--smoke-multi-cycle",
        action="store_true",
        help="run the fixed-data two-cycle Replay smoke loop with history injection",
    )
    parser.add_argument(
        "--smoke-root",
        help="optional storage root for --smoke-loop; defaults to a temporary directory",
    )
    parser.add_argument(
        "--smoke-no-resume",
        action="store_true",
        help="disable latest-completed recovery for --smoke-loop",
    )
    parser.add_argument(
        "--qlib-snapshot-smoke",
        action="store_true",
        help="load one explicit small Qlib snapshot without LLM or evaluation",
    )
    parser.add_argument(
        "--qlib-smoke-config",
        help="required TOML configuration for --qlib-snapshot-smoke",
    )
    parser.add_argument(
        "--qlib-smoke-root",
        help="optional artifact root for --qlib-snapshot-smoke",
    )
    parser.add_argument(
        "--live-llm-smoke",
        action="store_true",
        help="run one explicitly configured live LLM hypothesis call with audit storage",
    )
    parser.add_argument(
        "--live-llm-config",
        help="required full TOML configuration for --live-llm-smoke",
    )
    parser.add_argument(
        "--live-llm-root",
        help="optional local storage root for --live-llm-smoke",
    )
    parser.add_argument(
        "--mcts-smoke",
        action="store_true",
        help="run the fixed-data Replay MCTS smoke loop",
    )
    parser.add_argument(
        "--mcts-smoke-root",
        help="optional storage root for --mcts-smoke",
    )
    parser.add_argument(
        "--mcts-live",
        action="store_true",
        help="run MCTS search with a real LLM (requires config)",
    )
    parser.add_argument(
        "--mcts-live-config",
        help="required full TOML configuration for --mcts-live",
    )
    parser.add_argument(
        "--mcts-live-root",
        help="optional local storage root for --mcts-live",
    )
    parser.add_argument(
        "--mcts-multi",
        action="store_true",
        help="run multi-tree MCTS search (论文多树模式, requires --mcts-live-config)",
    )
    parser.add_argument(
        "--mcts-max-rounds",
        type=int,
        help="cap total MCTS rounds for --mcts-live/--mcts-multi "
        "(default: 30 single-tree / 80 multi-tree)",
    )
    parser.add_argument(
        "--mcts-max-evaluated",
        type=int,
        help="cap the number of evaluated factors for --mcts-multi; "
        "search stops when either --mcts-max-rounds or this is reached first",
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser(
        "run",
        help="run a configured recoverable CosmosAlpha chain",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="strict experiment TOML configuration",
    )
    run_parser.add_argument(
        "--run-id",
        required=True,
        help="stable run identifier",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume verified successful steps from workflow metadata",
    )
    run_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate configuration without assembling or running the chain",
    )
    lightgbm_parser = commands.add_parser(
        "train-lightgbm",
        help="train Qlib LightGBM over usable factors from one Graph run",
    )
    lightgbm_parser.add_argument(
        "--config",
        required=True,
        help="standalone LightGBM integration TOML configuration",
    )
    lightgbm_parser.add_argument(
        "--run-id",
        required=True,
        help="new model run identifier",
    )
    lightgbm_parser.add_argument(
        "--source-run-id",
        help="override factor_source.run_id for matrix experiments",
    )
    lightgbm_parser.add_argument(
        "--correlation-filter",
        choices=("enabled", "disabled"),
        help="override correlation_filter.enabled without duplicating TOML configs",
    )
    lightgbm_parser.add_argument(
        "--num-threads",
        type=int,
        help="override model.num_threads for the current run",
    )
    lightgbm_parser.add_argument(
        "--factor-workers",
        type=int,
        help="override parallel.factor_workers for the current run",
    )
    lightgbm_parser.add_argument(
        "--neutralization",
        choices=("enabled", "disabled"),
        help="override feature neutralization and its protocol for the current run",
    )
    lightgbm_parser.add_argument(
        "--include-rejected-factors",
        action="store_true",
        help="include rejected rows from factor_library/factors.csv",
    )
    lightgbm_parser.add_argument(
        "--zero-cost",
        action="store_true",
        help="set all exchange costs and market impact to zero",
    )
    lightgbm_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate configuration and source metadata without loading data",
    )
    monthly_parser = commands.add_parser(
        "train-lightgbm-monthly",
        help="train and backtest monthly Qlib LightGBM signals",
    )
    monthly_parser.add_argument(
        "--config",
        required=True,
        help="standalone monthly LightGBM integration TOML configuration",
    )
    monthly_parser.add_argument(
        "--run-id",
        required=True,
        help="new monthly model run identifier",
    )
    monthly_parser.add_argument(
        "--source-run-id",
        help="override factor_source.run_id for matrix experiments",
    )
    monthly_parser.add_argument(
        "--num-threads",
        type=int,
        help="override model.num_threads for the current run",
    )
    monthly_parser.add_argument(
        "--factor-workers",
        type=int,
        help="override parallel.factor_workers for the current run",
    )
    monthly_parser.add_argument(
        "--include-rejected-factors",
        action="store_true",
        help="include rejected Graph rows from factor_library/factors.csv",
    )
    monthly_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate monthly configuration and source metadata only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CosmosAlpha command-line interface.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "train-lightgbm":
        return _train_lightgbm_command(args)
    if args.command == "train-lightgbm-monthly":
        return _train_lightgbm_monthly_command(args)
    if args.smoke_multi_cycle:
        if args.smoke_root is None:
            with TemporaryDirectory() as tmpdir:
                result = run_fixed_replay_multicycle_smoke_loop(
                    tmpdir,
                    resume=not args.smoke_no_resume,
                )
                print(result.summary_json())
        else:
            result = run_fixed_replay_multicycle_smoke_loop(
                args.smoke_root,
                resume=not args.smoke_no_resume,
            )
            print(result.summary_json())
        return 0
    if args.live_llm_smoke:
        if args.live_llm_config is None:
            parser.error("--live-llm-smoke requires --live-llm-config")
        if args.live_llm_root is None:
            with TemporaryDirectory() as tmpdir:
                print(run_live_llm_smoke(args.live_llm_config, tmpdir).summary_json())
        else:
            print(
                run_live_llm_smoke(
                    args.live_llm_config,
                    args.live_llm_root,
                ).summary_json()
            )
        return 0
    if args.mcts_smoke:
        if args.mcts_smoke_root is None:
            with TemporaryDirectory() as tmpdir:
                print(run_mcts_smoke(tmpdir).summary_json())
        else:
            print(run_mcts_smoke(args.mcts_smoke_root).summary_json())
        return 0
    if args.mcts_multi:
        if args.mcts_live_config is None:
            parser.error("--mcts-multi requires --mcts-live-config")
        if args.mcts_live_root is None:
            with TemporaryDirectory() as tmpdir:
                print(
                    run_mcts_live_multi(
                        args.mcts_live_config,
                        tmpdir,
                        max_total_rounds=args.mcts_max_rounds,
                        max_evaluated_factors=args.mcts_max_evaluated,
                    ).summary_json()
                )
        else:
            print(
                run_mcts_live_multi(
                    args.mcts_live_config,
                    args.mcts_live_root,
                    max_total_rounds=args.mcts_max_rounds,
                    max_evaluated_factors=args.mcts_max_evaluated,
                ).summary_json()
            )
        return 0
    if args.mcts_live:
        if args.mcts_live_config is None:
            parser.error("--mcts-live requires --mcts-live-config")
        if args.mcts_live_root is None:
            with TemporaryDirectory() as tmpdir:
                print(
                    run_mcts_live(
                        args.mcts_live_config,
                        tmpdir,
                        max_iters=args.mcts_max_rounds,
                    ).summary_json()
                )
        else:
            print(
                run_mcts_live(
                    args.mcts_live_config,
                    args.mcts_live_root,
                    max_iters=args.mcts_max_rounds,
                ).summary_json()
            )
        return 0
    if args.qlib_snapshot_smoke:
        if args.qlib_smoke_config is None:
            parser.error("--qlib-snapshot-smoke requires --qlib-smoke-config")
        if args.qlib_smoke_root is None:
            with TemporaryDirectory() as tmpdir:
                print(run_qlib_snapshot_smoke(args.qlib_smoke_config, tmpdir))
        else:
            print(
                run_qlib_snapshot_smoke(
                    args.qlib_smoke_config,
                    args.qlib_smoke_root,
                )
            )
        return 0
    if args.smoke_loop:
        if args.smoke_root is None:
            with TemporaryDirectory() as tmpdir:
                result = run_fixed_replay_smoke_loop(
                    tmpdir,
                    resume=not args.smoke_no_resume,
                )
                print(result.summary_json())
        else:
            result = run_fixed_replay_smoke_loop(
                args.smoke_root,
                resume=not args.smoke_no_resume,
            )
            print(result.summary_json())
        return 0
    parser.print_help()
    return 0


def _train_lightgbm_command(args: object) -> int:
    """Preflight then run the standalone LightGBM integration."""
    _set_lightgbm_native_thread_environment()
    from cosmosalpha.modeling import (
        MODEL_FEATURE_NEUTRALIZED_PROTOCOL,
        MODEL_FEATURE_PREPROCESSING_PROTOCOL,
        LightGBMIntegrationError,
        load_lightgbm_integration_config,
        preflight_lightgbm_integration,
        run_lightgbm_integration,
    )

    try:
        config = load_lightgbm_integration_config(args.config)
        if args.source_run_id:
            config = replace(
                config,
                factor_source=replace(
                    config.factor_source,
                    run_id=args.source_run_id,
                ),
            )
        if args.correlation_filter:
            config = replace(
                config,
                correlation_filter=replace(
                    config.correlation_filter,
                    enabled=args.correlation_filter == "enabled",
                ),
            )
        if args.num_threads is not None:
            config = replace(
                config,
                model=replace(config.model, num_threads=args.num_threads),
            )
        if args.factor_workers is not None:
            config = replace(
                config,
                parallel=replace(
                    config.parallel,
                    factor_workers=args.factor_workers,
                ),
            )
        if args.neutralization:
            enabled = args.neutralization == "enabled"
            config = replace(
                config,
                feature_preprocessing=replace(
                    config.feature_preprocessing,
                    protocol=(
                        MODEL_FEATURE_NEUTRALIZED_PROTOCOL
                        if enabled
                        else MODEL_FEATURE_PREPROCESSING_PROTOCOL
                    ),
                    neutralization_enabled=enabled,
                ),
            )
        if args.include_rejected_factors:
            config = replace(
                config,
                factor_source=replace(
                    config.factor_source,
                    include_rejected_factors=True,
                ),
            )
        if args.zero_cost:
            config = replace(
                config,
                backtest=replace(
                    config.backtest,
                    open_cost=0.0,
                    close_cost=0.0,
                    min_cost=0,
                    impact_cost=0.0,
                ),
            )
    except (OSError, ValueError) as error:
        print(
            to_json(
                {
                    "run_id": args.run_id,
                    "ready": False,
                    "candidate_count": 0,
                    "issues": ({"code": "CONFIG_LOAD_FAILED", "message": str(error)},),
                }
            )
        )
        return 2
    report = preflight_lightgbm_integration(config, args.run_id)
    if not report.ready or args.preflight_only:
        print(report.summary_json())
        return 0 if report.ready else 2
    try:
        result = run_lightgbm_integration(config, args.run_id)
    except LightGBMIntegrationError as error:
        print(
            to_json(
                {
                    "error_code": "LIGHTGBM_INTEGRATION_FAILED",
                    "message": str(error),
                    "run_id": args.run_id,
                    "status": "failed",
                }
            )
        )
        return 1
    print(result.summary_json())
    return 0


def _set_lightgbm_native_thread_environment() -> None:
    """Keep numerical libraries single-threaded outside LightGBM itself."""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _train_lightgbm_monthly_command(args: object) -> int:
    """Preflight then run the standalone monthly LightGBM integration."""
    _set_lightgbm_native_thread_environment()
    from cosmosalpha.modeling import (
        LightGBMIntegrationError,
        load_lightgbm_monthly_integration_config,
        preflight_lightgbm_monthly_integration,
        run_lightgbm_monthly_integration,
    )

    try:
        config = load_lightgbm_monthly_integration_config(args.config)
        if args.source_run_id:
            config = replace(
                config,
                factor_source=replace(
                    config.factor_source,
                    run_id=args.source_run_id,
                ),
            )
        if args.num_threads is not None:
            config = replace(
                config,
                model=replace(config.model, num_threads=args.num_threads),
            )
        if args.factor_workers is not None:
            config = replace(
                config,
                parallel=replace(
                    config.parallel,
                    factor_workers=args.factor_workers,
                ),
            )
        if args.include_rejected_factors:
            config = replace(
                config,
                factor_source=replace(
                    config.factor_source,
                    include_rejected_factors=True,
                ),
            )
    except (OSError, ValueError) as error:
        print(
            to_json(
                {
                    "run_id": args.run_id,
                    "ready": False,
                    "candidate_count": 0,
                    "issues": ({"code": "CONFIG_LOAD_FAILED", "message": str(error)},),
                }
            )
        )
        return 2
    report = preflight_lightgbm_monthly_integration(config, args.run_id)
    if not report.ready or args.preflight_only:
        print(report.summary_json())
        return 0 if report.ready else 2
    try:
        result = run_lightgbm_monthly_integration(config, args.run_id)
    except LightGBMIntegrationError as error:
        print(
            to_json(
                {
                    "error_code": "LIGHTGBM_MONTHLY_INTEGRATION_FAILED",
                    "message": str(error),
                    "run_id": args.run_id,
                    "status": "failed",
                }
            )
        )
        return 1
    print(result.summary_json())
    return 0


def _run_command(args: object) -> int:
    """Preflight then execute a formal runtime without CLI-level assembly."""
    try:
        config = load_experiment_config(args.config)
    except (OSError, CoreValidationError, ValueError) as error:
        report = PreflightReport(
            run_id=args.run_id,
            config_snapshot_id="config-unavailable",
            ready=False,
            checks=(),
            issues=(PreflightIssue("CONFIG_LOAD_FAILED", str(error)),),
        )
        print(report.summary_json())
        return 2
    report = run_preflight(config, run_id=args.run_id, resume=args.resume)
    if not report.ready or args.preflight_only:
        print(report.summary_json())
        return 0 if report.ready else 2
    try:
        result = run_formal_runtime(
            config,
            run_id=args.run_id,
            resume=args.resume,
        )
    except Exception as error:
        print(
            to_json(
                {
                    "error_code": "RUNTIME_FAILED",
                    "message": str(error) or error.__class__.__name__,
                    "run_id": args.run_id,
                    "status": "failed",
                }
            )
        )
        return 1
    print(result.summary_json())
    return 0
