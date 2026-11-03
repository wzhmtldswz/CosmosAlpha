"""Explicit real-LLM MCTS live entry point.

Supports both Mock data (smoke) and real Qlib data, depending on config.
Calls a real OpenAI-compatible LLM for each expansion.
Saves full audit evidence (prompts, call records, raw responses) alongside MCTS results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cosmosalpha.agents import FactorDesignConstraints
from cosmosalpha.core import (
    ObjectRef,
    create_config_snapshot,
    load_experiment_config,
    to_json,
)
from cosmosalpha.data import (
    DataSnapshot,
    MarketDataRequest,
    QlibFeatureSource,
    QlibFieldMapping,
    QlibMarketDataProvider,
)
from cosmosalpha.data.models import MARKET_DATA_FIELDS
from cosmosalpha.evaluation import (
    AcceptancePolicy,
    FactorEvaluator,
    GplearnNeutralizedPreprocessor,
)
from cosmosalpha.factors.expression import (
    build_base_field_catalog,
    build_base_operator_catalog,
)
from cosmosalpha.llm import (
    AuditedLLMExecutor,
    LLMCallRecorder,
    LLMRetryPolicy,
    OpenAICompatibleLLMClient,
    PromptTemplateRegistry,
    PromptViewBuilder,
)
from cosmosalpha.pipeline.agents import (
    ExpressionRepairAgent,
    FactorDesignAgent,
    FactorExpressionAgent,
    HypothesisAgent,
)
from cosmosalpha.pipeline.loop import CosmosAlphaLoop
from cosmosalpha.pipeline.mcts.controller import MCTSController
from cosmosalpha.pipeline.mcts.stub_agents import MCTSFeedbackStub
from cosmosalpha.pipeline.smoke import (
    _build_smoke_snapshot,
    _config_snapshot_ref,
    _research_context,
)
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    SqliteFactorLibrary,
    SqliteWorkflowMetadataRepository,
)


@dataclass
class MCTSLiveRun:
    """Summary of one real-LLM MCTS run."""

    storage_root: str
    tree_size: int
    alpha_zoo_size: int
    iterations: int
    best_rank_ic: float | None
    hypothesis_refs: tuple[ObjectRef, ...]
    prompt_refs: tuple[ObjectRef, ...]
    call_record_refs: tuple[ObjectRef, ...]

    def summary_json(self) -> str:
        """Return the independent MCTS run summary."""
        return to_json(
            {
                "storage_root": self.storage_root,
                "tree_size": self.tree_size,
                "alpha_zoo_size": self.alpha_zoo_size,
                "iterations": self.iterations,
                "best_rank_ic": self.best_rank_ic,
                "hypothesis_refs": self.hypothesis_refs,
                "prompt_refs": self.prompt_refs,
                "call_record_refs": self.call_record_refs,
            }
        )


def run_mcts_live(
    config_path: str | Path,
    storage_root: str | Path,
    max_iters: int | None = None,
) -> MCTSLiveRun:
    """Run a real-LLM MCTS search using an explicit experiment configuration.

    Args:
        config_path: Full experiment TOML with ``llm.replay_only = false``
            and a real base_url / model_alias.
        storage_root: Local object and artifact repository root.
        max_iters: Optional cap on the number of MCTS iterations; falls back
            to the built-in default when omitted.
    """
    config = load_experiment_config(config_path)
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
    snapshot_data = create_config_snapshot(config)
    objects.put_object("config_snapshot", snapshot_data.snapshot_id, snapshot_data)

    client = OpenAICompatibleLLMClient.from_config(config.llm)
    registry = PromptTemplateRegistry.from_manifest(
        config.prompts.manifest,
    )
    prompt_view_builder = PromptViewBuilder(objects)
    retry_policy = LLMRetryPolicy.from_config(config.llm, config.prompts)

    # 链式真 agents（单次挖掘复用链式结构，进化机制为 MCTS）
    factor_executor = AuditedLLMExecutor(
        client=client,
        recorder=LLMCallRecorder(artifacts),
        object_repository=objects,
        retry_policy=retry_policy,
    )
    # feedback 使用确定性 MCTSFeedbackStub（对齐 Graph，避免 LLM 空响应）

    snapshot = _load_snapshot(config, artifacts)
    loop = CosmosAlphaLoop(
        hypothesis_agent=HypothesisAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        factor_design_agent=FactorDesignAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
            prompt_id="mcts_factor_design",
        ),
        factor_expression_agent=FactorExpressionAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        feedback_agent=MCTSFeedbackStub(),
        artifact_repository=artifacts,
        experiment_repository=objects,
        workflow_metadata_repository=metadata,
        preprocessor=GplearnNeutralizedPreprocessor(
            artifacts,
            min_cross_section_samples=config.evaluation.min_cross_section_samples,
        ),
        evaluator=FactorEvaluator(
            artifacts,
            min_cross_section_samples=config.evaluation.min_cross_section_samples,
            min_daily_effective_stock_ratio=(
                config.evaluation.min_daily_effective_stock_ratio
            ),
            reference_factor_values_provider=factor_library,
        ),
        acceptance_policy=AcceptancePolicy(
            threshold=-1.0,
            min_effective_sample_count=config.evaluation.min_effective_sample_count,
            min_effective_date_ratio=config.evaluation.min_effective_date_ratio,
        ),
        factor_library=factor_library,
        generation_constraints=FactorDesignConstraints(
            max_designs=config.factor_generation.max_proposals
        ),
        max_validation_repair_attempts=(
            config.factor_generation.max_validation_repair_attempts
        ),
        clock=lambda: datetime(2026, 7, 8, 0, 0, tzinfo=UTC),
        # acceptance 阈值 -1.0 只为保留树节点；因子库用真实门槛过滤弱因子
        publish_min_rank_ic=config.evaluation.acceptance_threshold,
        # 与 FZoo _FZOO_MAX_CORR=0.75 对齐：值相关 ≥0.75 的重复因子不入交付库
        publish_max_corr=0.75,
        # MCTS 细化设计允许更深的表达式（Graph/chain 路径保持 8）
        expression_max_depth=12,
    )

    base_context = _research_context(
        available_fields=MARKET_DATA_FIELDS,
        research_goal="MCTS search for daily alpha factors on CSI300.",
        constraints=("Use the provided seed expression as the factor.",),
    )
    config_snapshot_ref = _config_snapshot_ref(artifacts)
    objects.put_object("research_context", "live-mcts-context", base_context)

    mcts = MCTSController(
        loop=loop,
        experiment_repository=objects,
        artifact_repository=artifacts,
        llm_client=client,
        field_catalog=build_base_field_catalog(),
        operator_catalog=build_base_operator_catalog(),
        exploration_weight=1.3,
        alpha_threshold=config.evaluation.acceptance_threshold,
        model_alias=config.llm.model_alias,
        correlation_min_cross_section_samples=(
            config.evaluation.min_cross_section_samples
        ),
        correlation_min_daily_effective_stock_ratio=(
            config.evaluation.min_daily_effective_stock_ratio
        ),
        correlation_min_pair_window_ratio=config.evaluation.min_effective_date_ratio,
    )
    max_iters = max_iters or 30
    result = mcts.run(
        seed_expression="rank(close - delay(close, 5))",
        data_snapshot=snapshot,
        base_context=base_context,
        config_snapshot_ref=config_snapshot_ref,
        code_version="mcts-live",
        max_iterations=max_iters,
        initial_budget=5,
        run_id="mcts-live-run",
    )

    # Collect audit evidence (design and expression agents make audited LLM calls)
    prompt_refs = (
        tuple(factor_executor.prompt_refs.values())
        if factor_executor.prompt_refs
        else ()
    )
    all_prompt_refs = prompt_refs
    call_record_refs = tuple(factor_executor.call_record_refs)
    return MCTSLiveRun(
        storage_root=str(root),
        tree_size=len(result.tree.all_nodes),
        alpha_zoo_size=len(result.tree.alpha_zoo),
        iterations=result.iterations,
        best_rank_ic=result.best_rank_ic,
        hypothesis_refs=(),
        prompt_refs=all_prompt_refs,
        call_record_refs=call_record_refs,
    )


def run_mcts_live_multi(
    config_path: str | Path,
    storage_root: str | Path,
    max_total_rounds: int | None = None,
    max_evaluated_factors: int | None = None,
) -> MCTSLiveRun:
    """Run multiple chain-rooted MCTS trees with a shared factor zoo.

    与 run_mcts_live 共享相同的配置和基础设施（种子机制已取消，
    根节点 = 一次完整链式 loop，见 RootGenerator 接口）。

    Args:
        config_path: Full experiment TOML with ``llm.replay_only = false``.
        storage_root: Local object and artifact repository root.
        max_total_rounds: Optional cap on the total number of rounds across
            all trees; falls back to the built-in default when omitted.
        max_evaluated_factors: Optional cap on the total number of evaluated
            factors; search stops when either cap is reached first.
    """
    config = load_experiment_config(Path(config_path))
    config_snapshot = create_config_snapshot(config)

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
    objects.put_object(
        "config_snapshot",
        config_snapshot.snapshot_id,
        config_snapshot,
    )

    client = OpenAICompatibleLLMClient.from_config(config.llm)
    registry = PromptTemplateRegistry.from_manifest(
        config.prompts.manifest,
    )
    prompt_view_builder = PromptViewBuilder(objects)
    retry_policy = LLMRetryPolicy.from_config(config.llm, config.prompts)

    # 链式真 agents（单次挖掘复用链式结构，进化机制为 MCTS）
    factor_executor = AuditedLLMExecutor(
        client=client,
        recorder=LLMCallRecorder(artifacts),
        object_repository=objects,
        retry_policy=retry_policy,
    )
    # feedback 使用确定性 MCTSFeedbackStub（对齐 Graph，避免 LLM 空响应）

    snapshot = _load_snapshot(config, artifacts)
    loop = CosmosAlphaLoop(
        hypothesis_agent=HypothesisAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        factor_design_agent=FactorDesignAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
            prompt_id="mcts_factor_design",
        ),
        factor_expression_agent=FactorExpressionAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            client,
            prompt_view_builder=prompt_view_builder,
            registry=registry,
            model_alias=config.llm.model_alias,
            max_input_tokens=config.prompts.max_input_tokens,
            max_output_tokens=config.llm.max_output_tokens,
            executor=factor_executor,
        ),
        feedback_agent=MCTSFeedbackStub(),
        artifact_repository=artifacts,
        experiment_repository=objects,
        workflow_metadata_repository=metadata,
        preprocessor=GplearnNeutralizedPreprocessor(
            artifacts,
            min_cross_section_samples=config.evaluation.min_cross_section_samples,
        ),
        evaluator=FactorEvaluator(
            artifacts,
            min_cross_section_samples=config.evaluation.min_cross_section_samples,
            min_daily_effective_stock_ratio=(
                config.evaluation.min_daily_effective_stock_ratio
            ),
            reference_factor_values_provider=factor_library,
        ),
        acceptance_policy=AcceptancePolicy(
            threshold=-1.0,
            min_effective_sample_count=config.evaluation.min_effective_sample_count,
            min_effective_date_ratio=config.evaluation.min_effective_date_ratio,
        ),
        factor_library=factor_library,
        generation_constraints=FactorDesignConstraints(
            max_designs=config.factor_generation.max_proposals
        ),
        max_validation_repair_attempts=(
            config.factor_generation.max_validation_repair_attempts
        ),
        clock=lambda: datetime(2026, 7, 8, 0, 0, tzinfo=UTC),
        # acceptance 阈值 -1.0 只为保留树节点；因子库用真实门槛过滤弱因子
        publish_min_rank_ic=config.evaluation.acceptance_threshold,
        # 与 FZoo _FZOO_MAX_CORR=0.75 对齐：值相关 ≥0.75 的重复因子不入交付库
        publish_max_corr=0.75,
        # MCTS 多树模式同样允许更深的表达式
        expression_max_depth=12,
    )

    base_context = _research_context(
        available_fields=MARKET_DATA_FIELDS,
        research_goal="MCTS multi-tree search for daily alpha factors on CSI300.",
        constraints=(),
    )
    config_snapshot_ref = _config_snapshot_ref(artifacts)
    objects.put_object("research_context", "live-mcts-context", base_context)

    mcts = MCTSController(
        loop=loop,
        experiment_repository=objects,
        artifact_repository=artifacts,
        llm_client=client,
        field_catalog=build_base_field_catalog(),
        operator_catalog=build_base_operator_catalog(),
        exploration_weight=1.6,  # v15o 均衡档：UCT 探索权重 1.3→1.6（多树模式，多样化为先）
        alpha_threshold=config.evaluation.acceptance_threshold,
        model_alias=config.llm.model_alias,
        correlation_min_cross_section_samples=(
            config.evaluation.min_cross_section_samples
        ),
        correlation_min_daily_effective_stock_ratio=(
            config.evaluation.min_daily_effective_stock_ratio
        ),
        correlation_min_pair_window_ratio=config.evaluation.min_effective_date_ratio,
    )
    max_total = max_total_rounds or 80
    result = mcts.run_multi_tree(
        data_snapshot=snapshot,
        base_context=base_context,
        config_snapshot_ref=config_snapshot_ref,
        code_version="mcts-multi",
        max_total_rounds=max_total,
        max_evaluated_factors=max_evaluated_factors,
        initial_budget=5,
        budget_increment=4,
        breakthrough_relax=0.9,
        run_id="mcts-multi-run",
    )

    prompt_refs = (
        tuple(ref for ref in factor_executor.prompt_refs.values())
        if factor_executor.prompt_refs
        else ()
    )
    call_record_refs = tuple(factor_executor.call_record_refs)
    return MCTSLiveRun(
        storage_root=str(root),
        tree_size=len(result.tree.all_nodes),
        alpha_zoo_size=len(result.tree.alpha_zoo),
        iterations=result.iterations,
        best_rank_ic=result.best_rank_ic,
        hypothesis_refs=(),
        prompt_refs=prompt_refs,
        call_record_refs=call_record_refs,
    )


def _load_snapshot(config, artifacts) -> DataSnapshot:
    """Load data snapshot: Qlib if enabled, Mock otherwise."""
    if config.qlib.enabled:
        request = MarketDataRequest(
            market=config.data.market,
            universe=config.universe.name,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            warmup_start_date=config.data.warmup_start_date,
            frequency=config.data.frequency,
            fields=MARKET_DATA_FIELDS,
            dataset_version=config.data.dataset_version,
            price_adjustment=config.data.price_adjustment,
            schema_version="market-data/1.0",
        )
        exposure_source = _qlib_source(config.qlib.exposure)
        mask_source = _qlib_source(config.qlib.mask)
        provider = QlibMarketDataProvider(
            artifacts,
            provider_uri=config.qlib.market.provider_uri,
            trading_calendar_version=config.data.trading_calendar_version,
            exposure_source=exposure_source,
            mask_source=mask_source,
            instruments=(
                tuple(config.qlib.instruments) if config.qlib.instruments else None
            ),
        )
        return provider.load_snapshot(request)
    return _build_smoke_snapshot(artifacts)


def _qlib_source(cfg) -> QlibFeatureSource:
    """Convert config Qlib source section to QlibFeatureSource."""
    return QlibFeatureSource(
        provider_uri=cfg.provider_uri,
        fields=tuple(QlibFieldMapping(f.field_name, f.expression) for f in cfg.fields),
    )
