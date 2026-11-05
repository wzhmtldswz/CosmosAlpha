"""Deterministic controller for serial, lineage-aware research cycles."""

from __future__ import annotations

from dataclasses import dataclass

from cosmosalpha.agents import ResearchContext
from cosmosalpha.core import ArtifactRef, CoreValidationError
from cosmosalpha.core.config import PromptsConfig, WorkflowConfig
from cosmosalpha.core.logging import LOGGER, log_context
from cosmosalpha.data import DataSnapshot
from cosmosalpha.pipeline.context import ContextBuilder, ContextBuildResult
from cosmosalpha.pipeline.loop import CosmosAlphaLoop, CosmosAlphaLoopResult


@dataclass(frozen=True, slots=True)
class MultiCycleResult:
    """Results and injected contexts for one bounded serial research run."""

    run_id: str
    cycle_results: tuple[CosmosAlphaLoopResult, ...]
    context_results: tuple[ContextBuildResult, ...]

    def __post_init__(self) -> None:
        """Validate controller output cardinality and identity."""
        if type(self.run_id) is not str or not self.run_id:
            raise CoreValidationError("run_id must be a non-empty string")
        if not isinstance(self.cycle_results, tuple) or not self.cycle_results:
            raise CoreValidationError("cycle_results must be a non-empty tuple")
        if not isinstance(self.context_results, tuple):
            raise CoreValidationError("context_results must be a tuple")
        if len(self.cycle_results) != len(self.context_results):
            raise CoreValidationError("cycle and context result counts must match")


class MultiCycleController:
    """Run configured cycles while preserving parent lineage and bounded history."""

    def __init__(
        self,
        loop: CosmosAlphaLoop,
        context_builder: ContextBuilder,
        *,
        workflow_config: WorkflowConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Create a controller from stable orchestration and configuration objects."""
        if not isinstance(loop, CosmosAlphaLoop):
            raise CoreValidationError("loop must be CosmosAlphaLoop")
        if not isinstance(context_builder, ContextBuilder):
            raise CoreValidationError("context_builder must be ContextBuilder")
        if not isinstance(workflow_config, WorkflowConfig):
            raise CoreValidationError("workflow_config must be WorkflowConfig")
        if not isinstance(prompts_config, PromptsConfig):
            raise CoreValidationError("prompts_config must be PromptsConfig")
        self._loop = loop
        self._context_builder = context_builder
        self._workflow_config = workflow_config
        self._prompts_config = prompts_config

    def run(
        self,
        *,
        run_id: str,
        cycle_id_prefix: str,
        base_context: ResearchContext,
        data_snapshot: DataSnapshot,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        parent_ids: tuple[str, ...] = (),
        run_context: ResearchContext | None = None,
        complete_run: bool = True,
        resume: bool = True,
    ) -> MultiCycleResult:
        """Run ``workflow.max_cycles`` serial research cycles.

        Each completed cycle becomes the direct parent of the next. The current cycle
        is excluded from its own history query so resuming reconstructs the same input
        context and can reuse SQLite step attempts safely.
        """
        _require_string(run_id, "run_id")
        _require_string(cycle_id_prefix, "cycle_id_prefix")
        if not isinstance(base_context, ResearchContext):
            raise CoreValidationError("base_context must be ResearchContext")
        if not isinstance(data_snapshot, DataSnapshot):
            raise CoreValidationError("data_snapshot must be DataSnapshot")
        if not isinstance(config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        if run_context is not None and not isinstance(run_context, ResearchContext):
            raise CoreValidationError("run_context must be ResearchContext or None")
        if type(complete_run) is not bool:
            raise CoreValidationError("complete_run must be a boolean")
        _require_string(code_version, "code_version")
        _require_parent_ids(parent_ids)
        if type(resume) is not bool:
            raise CoreValidationError("resume must be a boolean")

        current_parent_ids = parent_ids
        cycle_results: list[CosmosAlphaLoopResult] = []
        context_results: list[ContextBuildResult] = []
        for index in range(self._workflow_config.max_cycles):
            cycle_id = f"{cycle_id_prefix}-{index + 1:03d}"
            with log_context(cycle=cycle_id):
                LOGGER.info(
                    "Cycle started index=%d/%d",
                    index + 1,
                    self._workflow_config.max_cycles,
                )
            context_result = self._context_builder.build(
                base_context,
                data_version=data_snapshot.snapshot_id,
                parent_ids=current_parent_ids,
                max_history_items=self._prompts_config.max_history_items,
                max_input_tokens=self._prompts_config.max_input_tokens,
                current_run_id=run_id,
                exclude_run_cycle=(run_id, cycle_id),
            )
            cycle_result = self._loop.run(
                run_id=run_id,
                cycle_id=cycle_id,
                context=context_result.context,
                data_snapshot=data_snapshot,
                config_snapshot_ref=config_snapshot_ref,
                code_version=code_version,
                parent_ids=current_parent_ids,
                existing_factor_refs=context_result.existing_factor_refs,
                history_lesson_refs=context_result.selected_lesson_refs,
                run_context=base_context if run_context is None else run_context,
                complete_run=(
                    complete_run and index == self._workflow_config.max_cycles - 1
                ),
                resume=resume,
            )
            if cycle_result.experiment_record is None:
                raise CoreValidationError(
                    "completed cycle must return an ExperimentRecord"
                )
            cycle_results.append(cycle_result)
            context_results.append(context_result)
            current_parent_ids = (cycle_result.experiment_record.experiment_id,)

        return MultiCycleResult(
            run_id=run_id,
            cycle_results=tuple(cycle_results),
            context_results=tuple(context_results),
        )


def _require_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _require_parent_ids(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        type(parent_id) is str and parent_id for parent_id in value
    ):
        raise CoreValidationError("parent_ids must be a tuple of non-empty strings")
