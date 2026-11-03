"""Fixed-data Replay MCTS smoke loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cosmosalpha.agents import FactorDesignConstraints
from cosmosalpha.core import to_json
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
    DEFAULT_PROMPT_ROOT,
    LLMRequest,
    LLMResponse,
    PromptTemplateRegistry,
    PromptViewBuilder,
)
from cosmosalpha.pipeline.agents import (
    ExpressionRepairAgent,
    FactorDesignAgent,
    FactorExpressionAgent,
)
from cosmosalpha.pipeline.loop import CosmosAlphaLoop
from cosmosalpha.pipeline.mcts.controller import MCTSController
from cosmosalpha.pipeline.mcts.stub_agents import (
    MCTSFeedbackStub,
    MCTSHypothesisStub,
)
from cosmosalpha.pipeline.smoke import (
    _build_smoke_snapshot,
    _config_snapshot_ref,
    _factor_design_response,
    _research_context,
)
from cosmosalpha.storage import (
    LocalArtifactRepository,
    LocalExperimentRepository,
    SqliteFactorLibrary,
    SqliteWorkflowMetadataRepository,
)

MCTS_SMOKE_RUN_ID = "mcts-smoke-run"


class _SequencedReplayClient:
    """Replay fixed responses while retaining every rendered request."""

    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        sequences: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Create a strict replay client with optional per-prompt sequences."""
        self._responses = dict(responses)
        self._sequences = (
            {prompt_id: tuple(values) for prompt_id, values in sequences.items()}
            if sequences is not None
            else {}
        )
        self._counts: dict[str, int] = {}
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next fixed response for a rendered prompt."""
        self.requests.append(request)
        prompt_id = request.prompt.prompt_id
        sequence = self._sequences.get(prompt_id)
        if sequence is None:
            content = self._responses.get(prompt_id)
        else:
            index = self._counts.get(prompt_id, 0)
            if index >= len(sequence):
                raise AssertionError(f"MCTS smoke replay exhausted: {prompt_id}")
            self._counts[prompt_id] = index + 1
            content = sequence[index]
        if content is None:
            raise AssertionError(f"MCTS smoke replay missing: {prompt_id}")
        return LLMResponse(
            content=content,
            model_alias=request.model_alias,
            prompt_render_id=request.prompt.prompt_render_id,
            finish_reason="stop",
            input_tokens=request.prompt.estimated_input_tokens,
            output_tokens=max(1, (len(content) + 3) // 4),
        )


@dataclass
class MCTSSmokeRun:
    """Summary of one MCTS smoke test run."""

    tree_size: int
    alpha_zoo_size: int
    iterations: int
    best_rank_ic: float | None
    storage_root: str
    prompt_ids: tuple[str, ...]
    factor_design_prompt_texts: tuple[str, ...]

    def summary_json(self) -> str:
        """Return the fixed-data MCTS validation summary."""
        return to_json(
            {
                "tree_size": self.tree_size,
                "alpha_zoo_size": self.alpha_zoo_size,
                "iterations": self.iterations,
                "best_rank_ic": self.best_rank_ic,
                "prompt_ids": self.prompt_ids,
                "storage_root": self.storage_root,
            }
        )


def run_mcts_smoke(
    storage_root: str | Path,
    *,
    resume: bool = True,
) -> MCTSSmokeRun:
    """Run MCTS smoke test using fixed sample data and Replay LLM."""
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
    replay = _SequencedReplayClient(
        {
            "mcts_factor_design": _factor_design_response(),
            "factor_expression": to_json({"proposed_expression": "unknown(close)"}),
        },
        sequences={
            "expression_repair": (
                to_json(
                    {
                        "proposed_expression": (
                            "rank(div(delta(close, 5), ts_stddev(close, 10)))"
                        )
                    }
                ),
                to_json({"proposed_expression": "rank(ts_zscore(volume, 20))"}),
                to_json({"proposed_expression": "rank(div(volume, close))"}),
            )
        },
    )
    registry = PromptTemplateRegistry.from_directory(
        DEFAULT_PROMPT_ROOT,
    )
    prompt_views = PromptViewBuilder(objects)

    loop = CosmosAlphaLoop(
        hypothesis_agent=MCTSHypothesisStub(),
        factor_design_agent=FactorDesignAgent(
            replay,
            prompt_view_builder=prompt_views,
            registry=registry,
            prompt_id="mcts_factor_design",
        ),
        factor_expression_agent=FactorExpressionAgent(
            replay,
            prompt_view_builder=prompt_views,
            registry=registry,
        ),
        expression_repair_agent=ExpressionRepairAgent(
            replay,
            prompt_view_builder=prompt_views,
            registry=registry,
        ),
        feedback_agent=MCTSFeedbackStub(),
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
        ),
        factor_library=factor_library,
        generation_constraints=FactorDesignConstraints(max_designs=1),
        max_validation_repair_attempts=2,
        clock=lambda: datetime(2026, 7, 8, 0, 0, tzinfo=UTC),
        expression_max_depth=12,
    )

    base_context = _research_context()
    mcts = MCTSController(
        loop=loop,
        experiment_repository=objects,
        artifact_repository=artifacts,
        llm_client=None,
        field_catalog=build_base_field_catalog(),
        operator_catalog=build_base_operator_catalog(),
        exploration_weight=1.4,
        alpha_threshold=-1.0,
        correlation_min_cross_section_samples=4,
        correlation_min_daily_effective_stock_ratio=0.8,
        correlation_min_pair_window_ratio=0.8,
    )
    result = mcts.run(
        seed_expression="rank(close - delay(close, 5))",
        data_snapshot=snapshot,
        base_context=base_context,
        config_snapshot_ref=_config_snapshot_ref(artifacts),
        code_version="mcts-smoke",
        max_iterations=2,
        initial_budget=2,
        budget_increment=0,
        run_id=MCTS_SMOKE_RUN_ID,
    )
    factor_design_prompts = tuple(
        request.prompt
        for request in replay.requests
        if request.prompt.prompt_id == "mcts_factor_design"
    )
    return MCTSSmokeRun(
        tree_size=len(result.tree.all_nodes),
        alpha_zoo_size=len(result.tree.alpha_zoo),
        iterations=result.iterations,
        best_rank_ic=result.best_rank_ic,
        storage_root=str(root),
        prompt_ids=tuple(request.prompt.prompt_id for request in replay.requests),
        factor_design_prompt_texts=tuple(
            "\n".join(message.content for message in prompt.messages)
            for prompt in factor_design_prompts
        ),
    )
