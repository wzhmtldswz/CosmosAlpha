"""Fixed-data replay smoke loop for local CLI and regression tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cosmosalpha.agents import FactorDesignConstraints, ResearchContext
from cosmosalpha.core import ArtifactRef, ObjectRef, to_json
from cosmosalpha.core.config import PromptsConfig, WorkflowConfig
from cosmosalpha.data import (
    DATA_SNAPSHOT_SCHEMA_VERSION,
    EXPOSURE_DATA_BATCH_SCHEMA_VERSION,
    FIELD_SCHEMA_VERSION,
    MARKET_DATA_BATCH_SCHEMA_VERSION,
    UNIVERSE_MASK_BATCH_SCHEMA_VERSION,
    DataSnapshot,
    ExposureDataBatch,
    MarketDataBatch,
    MarketDataRequest,
    UniverseMaskBatch,
    compute_data_snapshot_id,
    empty_quality_report,
)
from cosmosalpha.evaluation import (
    AcceptancePolicy,
    FactorEvaluator,
    GplearnNeutralizedPreprocessor,
)
from cosmosalpha.llm import (
    DEFAULT_PROMPT_ROOT,
    FeedbackPromptViewBuilder,
    PromptTemplateRegistry,
    PromptViewBuilder,
    ReplayLLMClient,
)
from cosmosalpha.pipeline.agents import (
    ExpressionRepairAgent,
    FactorDesignAgent,
    FactorExpressionAgent,
    FeedbackAgent,
    HypothesisAgent,
)
from cosmosalpha.pipeline.context import ContextBuilder
from cosmosalpha.pipeline.loop import CosmosAlphaLoop, CosmosAlphaLoopResult
from cosmosalpha.pipeline.multicycle import MultiCycleController, MultiCycleResult
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    SqliteFactorLibrary,
    SqliteWorkflowMetadataRepository,
)

SMOKE_RUN_ID = "smoke-run"
SMOKE_CYCLE_ID = "cycle-001"


@dataclass(frozen=True, slots=True)
class SmokeLoopRun:
    """Summary of a fixed-data replay smoke loop run."""

    result: CosmosAlphaLoopResult
    storage_root: str

    def summary_json(self) -> str:
        """Return a stable JSON summary for CLI output."""
        record = self.result.experiment_record
        return to_json(
            {
                "accepted_count": (
                    0 if record is None else len(record.acceptance_decision_refs)
                ),
                "candidate_count": len(self.result.candidate_results),
                "experiment_id": None if record is None else record.experiment_id,
                "resumed": self.result.resumed,
                "run_id": self.result.run_id,
                "status": None if record is None else record.status.value,
                "storage_root": self.storage_root,
            }
        )


@dataclass(frozen=True, slots=True)
class MultiCycleSmokeRun:
    """Summary of the fixed-data two-cycle Replay smoke run."""

    result: MultiCycleResult
    storage_root: str

    def summary_json(self) -> str:
        """Return a stable JSON summary for CLI output."""
        records = tuple(
            result.experiment_record for result in self.result.cycle_results
        )
        return to_json(
            {
                "cycle_count": len(self.result.cycle_results),
                "cycle_ids": tuple(
                    result.cycle_id for result in self.result.cycle_results
                ),
                "history_item_counts": tuple(
                    len(result.selected_lesson_refs)
                    for result in self.result.context_results
                ),
                "run_id": self.result.run_id,
                "statuses": tuple(
                    None if record is None else record.status.value
                    for record in records
                ),
                "storage_root": self.storage_root,
            }
        )


def run_fixed_replay_smoke_loop(
    storage_root: str | Path,
    *,
    resume: bool = True,
) -> SmokeLoopRun:
    """Run the minimal fixed-data Replay end-to-end CosmosAlpha loop."""
    root = Path(storage_root)
    artifacts = LocalArtifactRepository(root)
    objects = LocalExperimentRepository(root)
    metadata = SqliteWorkflowMetadataRepository.initialize(
        root / "workflow.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    factor_library = SqliteFactorLibrary.initialize(
        root / "factor-library.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    snapshot = _build_smoke_snapshot(artifacts)
    registry = _v2_registry()
    prompt_views = PromptViewBuilder(objects)
    loop = CosmosAlphaLoop(
        hypothesis_agent=HypothesisAgent(
            ReplayLLMClient({"hypothesis": _hypothesis_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        factor_design_agent=FactorDesignAgent(
            ReplayLLMClient({"factor_design": _factor_design_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        factor_expression_agent=FactorExpressionAgent(
            ReplayLLMClient({"factor_expression": _factor_expression_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            ReplayLLMClient({"expression_repair": _factor_expression_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        feedback_agent=FeedbackAgent(
            ReplayLLMClient({"feedback": _feedback_response()}),
            prompt_view_builder=FeedbackPromptViewBuilder(objects),
            registry=registry,
        ),
        artifact_repository=artifacts,
        experiment_repository=objects,
        workflow_metadata_repository=metadata,
        preprocessor=GplearnNeutralizedPreprocessor(
            artifacts,
            min_cross_section_samples=4,
        ),
        evaluator=FactorEvaluator(
            artifacts,
            min_cross_section_samples=4,
            reference_factor_values_provider=factor_library,
        ),
        acceptance_policy=AcceptancePolicy(
            threshold=-1.0,
            min_effective_sample_count=1,
            min_effective_date_ratio=0.0,
        ),
        factor_library=factor_library,
        generation_constraints=FactorDesignConstraints(max_designs=1),
        max_validation_repair_attempts=2,
        clock=lambda: datetime(2026, 7, 8, 0, 0, tzinfo=UTC),
    )
    result = loop.run(
        run_id=SMOKE_RUN_ID,
        cycle_id=SMOKE_CYCLE_ID,
        context=_research_context(),
        data_snapshot=snapshot,
        config_snapshot_ref=_config_snapshot_ref(artifacts),
        code_version="smoke",
        resume=resume,
    )
    return SmokeLoopRun(result=result, storage_root=str(root))


def run_fixed_replay_multicycle_smoke_loop(
    storage_root: str | Path,
    *,
    resume: bool = True,
) -> MultiCycleSmokeRun:
    """Run two fixed-data Replay cycles with bounded historical context."""
    root = Path(storage_root)
    artifacts = LocalArtifactRepository(root)
    objects = LocalExperimentRepository(root)
    metadata = SqliteWorkflowMetadataRepository.initialize(
        root / "workflow.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    factor_library = SqliteFactorLibrary.initialize(
        root / "factor-library.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    registry = _v2_registry()
    prompt_views = PromptViewBuilder(objects)
    loop = CosmosAlphaLoop(
        hypothesis_agent=HypothesisAgent(
            ReplayLLMClient({"hypothesis": _hypothesis_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        factor_design_agent=FactorDesignAgent(
            ReplayLLMClient({"factor_design": _factor_design_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        factor_expression_agent=FactorExpressionAgent(
            ReplayLLMClient({"factor_expression": _factor_expression_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            ReplayLLMClient({"expression_repair": _factor_expression_response()}),
            registry=registry,
            prompt_view_builder=prompt_views,
        ),
        feedback_agent=FeedbackAgent(
            ReplayLLMClient({"feedback": _feedback_response()}),
            prompt_view_builder=FeedbackPromptViewBuilder(objects),
            registry=registry,
        ),
        artifact_repository=artifacts,
        experiment_repository=objects,
        workflow_metadata_repository=metadata,
        preprocessor=GplearnNeutralizedPreprocessor(
            artifacts,
            min_cross_section_samples=4,
        ),
        evaluator=FactorEvaluator(
            artifacts,
            min_cross_section_samples=4,
            reference_factor_values_provider=factor_library,
        ),
        acceptance_policy=AcceptancePolicy(
            threshold=-1.0,
            min_effective_sample_count=1,
            min_effective_date_ratio=0.0,
        ),
        factor_library=factor_library,
        generation_constraints=FactorDesignConstraints(max_designs=1),
        max_validation_repair_attempts=2,
        clock=lambda: datetime(2026, 7, 8, 0, 0, tzinfo=UTC),
    )
    result = MultiCycleController(
        loop,
        ContextBuilder(objects),
        workflow_config=WorkflowConfig(
            max_cycles=2,
            max_retries=0,
        ),
        prompts_config=PromptsConfig(
            manifest="builtin:v2",
            max_input_tokens=16000,
            max_history_items=1,
            max_proposals_per_call=2,
            max_format_retries=0,
        ),
    ).run(
        run_id=SMOKE_RUN_ID,
        cycle_id_prefix="cycle",
        base_context=_research_context(),
        data_snapshot=_build_smoke_snapshot(artifacts),
        config_snapshot_ref=_config_snapshot_ref(artifacts),
        code_version="smoke",
        resume=resume,
    )
    return MultiCycleSmokeRun(result=result, storage_root=str(root))


def _research_context(
    available_fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
    research_goal: str = "Run a deterministic local smoke test for daily alpha factors.",
    constraints: tuple[str, ...] = ("Use one simple expression factor.",),
) -> ResearchContext:
    return ResearchContext(
        research_goal=research_goal,
        market="cn",
        universe="all_a_share",
        frequency="1d",
        available_fields=available_fields,
        operator_catalog_ref=ObjectRef(
            "operator_catalog",
            "operator-set-base",
            "operator-catalog/1.0",
        ),
        evaluation_protocol_ref=ObjectRef(
            "evaluation_protocol",
            "single-factor-ic",
            "evaluation-protocol/1.0",
        ),
        history_summaries=(),
        constraints=constraints,
        research_direction=None,
        research_direction_ref=None,
        schema_version="research-context/1.2",
    )


def _build_smoke_snapshot(artifacts: LocalArtifactRepository) -> DataSnapshot:
    dates = ("2020-01-01", "2020-01-02", "2020-01-03")
    instruments = tuple(f"s{i:02d}" for i in range(8))
    request = MarketDataRequest(
        market="cn",
        universe="smoke_universe",
        start_date=dates[0],
        end_date=dates[-1],
        warmup_start_date=dates[0],
        frequency="1d",
        fields=("open", "high", "low", "close", "volume"),
        dataset_version="fixed-replay-smoke-v1",
        price_adjustment="post_adjusted",
        schema_version="market-data/1.0",
    )
    market_records = _market_records(dates, instruments)
    exposure_records = _exposure_records(market_records)
    mask_records = _mask_records(market_records)
    market_ref = _put_records(artifacts, market_records, "market_data")
    exposure_ref = _put_records(artifacts, exposure_records, "exposure_data")
    mask_ref = _put_records(artifacts, mask_records, "mask_data")
    market_batch = MarketDataBatch(
        data_ref=market_ref,
        fields=tuple(sorted(request.fields)),
        sample_count=len(market_records),
        date_count=len(dates),
        instrument_count=len(instruments),
        source_start_date=dates[0],
        source_end_date=dates[-1],
        schema_version=MARKET_DATA_BATCH_SCHEMA_VERSION,
    )
    exposure_batch = ExposureDataBatch(
        data_ref=exposure_ref,
        fields=("industry_id", "log_mkt_cap", "ret20", "turn20", "vol20"),
        sample_count=len(exposure_records),
        date_count=len(dates),
        instrument_count=len(instruments),
        schema_version=EXPOSURE_DATA_BATCH_SCHEMA_VERSION,
    )
    mask_batch = UniverseMaskBatch(
        mask_data_ref=mask_ref,
        reason_data_ref=None,
        fields=(
            "universe_mask",
            "eligible_mask",
            "entry_tradable_mask",
            "exit_tradable_mask",
        ),
        sample_count=len(mask_records),
        excluded_sample_count=0,
        unknown_sample_count=0,
        schema_version=UNIVERSE_MASK_BATCH_SCHEMA_VERSION,
    )
    snapshot_id = compute_data_snapshot_id(
        request=request,
        market_data=market_batch,
        exposure_data=exposure_batch,
        mask_data=mask_batch,
        provider_alias="fixed_replay_smoke",
        provider_uri_fingerprint="fixed-replay-smoke",
        trading_calendar_version="fixed-smoke-calendar/1.0",
        field_schema_version=FIELD_SCHEMA_VERSION,
        source_date_range=(dates[0], dates[-1]),
    )
    return DataSnapshot(
        snapshot_id=snapshot_id,
        request=request,
        market_data=market_batch,
        exposure_data=exposure_batch,
        mask_data=mask_batch,
        quality_report=empty_quality_report(),
        provider_alias="fixed_replay_smoke",
        provider_uri_fingerprint="fixed-replay-smoke",
        dataset_version=request.dataset_version,
        trading_calendar_version="fixed-smoke-calendar/1.0",
        field_schema_version=FIELD_SCHEMA_VERSION,
        market="cn",
        timezone="Asia/Shanghai",
        frequency="1d",
        price_adjustment="post_adjusted",
        source_date_range=(dates[0], dates[-1]),
        artifact_refs=(market_ref, exposure_ref, mask_ref),
        schema_version=DATA_SNAPSHOT_SCHEMA_VERSION,
    )


def _market_records(
    dates: Sequence[str],
    instruments: Sequence[str],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(dates):
        for inst_index, instrument in enumerate(instruments):
            base = 10.0 + inst_index * 0.7 + date_index * 0.35
            close = base + ((inst_index + date_index) % 3) * 0.09
            rows.append(
                {
                    "datetime": date_value,
                    "instrument": instrument,
                    "close": close,
                    "high": close + 0.25,
                    "low": close - 0.25,
                    "open": base - 0.05,
                    "volume": 1000.0
                    + inst_index * 37.0
                    + date_index * 53.0
                    + ((inst_index + 2 * date_index) % 5) * 11.0,
                }
            )
    return tuple(rows)


def _exposure_records(
    market_records: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(market_records):
        close = float(row["close"])
        inst_index = index % 8
        date_index = index // 8
        rows.append(
            {
                "datetime": row["datetime"],
                "instrument": row["instrument"],
                "industry_id": inst_index % 2 + 1,
                "log_mkt_cap": math.log(close * 1_000_000.0),
                "ret20": (inst_index - 3.5) * 0.01 + date_index * 0.002,
                "turn20": 0.02 + ((inst_index * 3 + date_index) % 7) * 0.003,
                "vol20": 0.03 + ((inst_index + date_index * 2) % 5) * 0.004,
            }
        )
    return tuple(rows)


def _mask_records(
    market_records: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "datetime": row["datetime"],
            "instrument": row["instrument"],
            "entry_tradable_mask": True,
            "eligible_mask": True,
            "exit_tradable_mask": True,
            "universe_mask": True,
        }
        for row in market_records
    )


def _put_records(
    artifacts: LocalArtifactRepository,
    records: tuple[dict[str, object], ...],
    kind: str,
) -> ArtifactRef:
    return artifacts.put_bytes(
        to_json({"records": records}).encode("utf-8"),
        kind=kind,
        media_type="application/json",
    )


def _config_snapshot_ref(artifacts: LocalArtifactRepository) -> ArtifactRef:
    return artifacts.put_bytes(
        to_json(
            {
                "purpose": "fixed replay smoke loop",
                "schema_version": "config-snapshot/1.0",
            }
        ).encode("utf-8"),
        kind="config_snapshot",
        media_type="application/json",
    )


def _hypothesis_response() -> str:
    return to_json(
        {
            "assumptions": ["The smoke dataset is deterministic."],
            "expected_direction": "positive",
            "mechanism": "Large close values proxy short-term demand pressure.",
            "prediction_horizon": "1d",
            "rationale": "The expression is intentionally simple for smoke testing.",
            "required_fields": ["close", "volume"],
            "statement": "Volume-to-price rank predicts next-day returns.",
            "testable_prediction": "rank(div(volume, close)) should produce finite smoke metrics.",
        }
    )


def _factor_design_response() -> str:
    return to_json(
        {
            "designs": [
                {
                    "description": "Cross-sectional rank of volume divided by close.",
                    "expected_direction": "positive",
                    "failure_conditions": ["Price-level effects dominate activity."],
                    "formulation": "cross-sectional rank(volume / close)",
                    "mechanism_mapping": "Activity per price unit proxies demand pressure.",
                    "name": "ranked_volume_price_smoke",
                    "signal_definition": "Higher values indicate stronger pressure.",
                    "variables": [
                        {"name": "volume", "role": "activity numerator"},
                        {"name": "close", "role": "price denominator"},
                    ],
                    "window_choices": [],
                }
            ]
        }
    )


def _factor_expression_response() -> str:
    return to_json({"proposed_expression": "rank(div(volume, close))"})


def _v2_registry() -> PromptTemplateRegistry:
    return PromptTemplateRegistry.from_directory(
        DEFAULT_PROMPT_ROOT,
    )


def _feedback_response() -> str:
    return to_json(
        {
            "failure_analysis": [],
            "hypothesis_verdict": "supported",
            "next_actions": ["Use this only as a smoke test, not factor evidence."],
            "next_hypothesis_hint": "Run a larger replay or real data evaluation next.",
            "rejected_findings": [],
            "summary": "The fixed replay smoke loop completed.",
            "supported_findings": ["The chain produced persisted protocol objects."],
        }
    )
