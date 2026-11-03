"""Explicit, low-cost live LLM smoke entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cosmosalpha.agents import ResearchContext
from cosmosalpha.core import (
    ObjectRef,
    create_config_snapshot,
    load_experiment_config,
    to_json,
)
from cosmosalpha.factors import (
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
from cosmosalpha.pipeline.agents import HypothesisAgent
from cosmosalpha.storage import LocalArtifactRepository, LocalExperimentRepository


@dataclass(frozen=True, slots=True)
class LiveLLMSmokeRun:
    """Summary of one explicit live hypothesis-generation call."""

    storage_root: str
    hypothesis_ref: ObjectRef
    prompt_ref: ObjectRef
    call_record_refs: tuple[ObjectRef, ...]

    def summary_json(self) -> str:
        """Return a stable JSON summary without prompt or response contents."""
        return to_json(
            {
                "call_record_refs": self.call_record_refs,
                "hypothesis_ref": self.hypothesis_ref,
                "prompt_ref": self.prompt_ref,
                "storage_root": self.storage_root,
            }
        )


def run_live_llm_smoke(
    config_path: str | Path,
    storage_root: str | Path,
) -> LiveLLMSmokeRun:
    """Run one persisted live hypothesis prompt using an explicit configuration.

    Args:
        config_path: Full experiment TOML with ``llm.replay_only = false``.
        storage_root: Local object and artifact repository root for this invocation.

    Returns:
        References to the generated hypothesis and LLM audit evidence.
    """
    config = load_experiment_config(config_path)
    root = Path(storage_root)
    artifacts = LocalArtifactRepository(root)
    objects = LocalExperimentRepository(root)
    snapshot = create_config_snapshot(config)
    objects.put_object("config_snapshot", snapshot.snapshot_id, snapshot)

    context = _live_smoke_context()
    objects.put_object("research_context", "live-llm-smoke", context)
    field_ref = objects.put_object(
        "field_catalog",
        "live-llm-smoke-fields",
        build_base_field_catalog(),
    )
    objects.put_object(
        "operator_catalog",
        context.operator_catalog_ref.object_id,
        build_base_operator_catalog(),
    )
    client = OpenAICompatibleLLMClient.from_config(config.llm)
    executor = AuditedLLMExecutor(
        client=client,
        recorder=LLMCallRecorder(artifacts),
        object_repository=objects,
        retry_policy=LLMRetryPolicy.from_config(config.llm, config.prompts),
    )
    registry = PromptTemplateRegistry.from_manifest(
        config.prompts.manifest,
    )
    hypothesis = HypothesisAgent(
        client,
        registry=registry,
        model_alias=config.llm.model_alias,
        max_input_tokens=config.prompts.max_input_tokens,
        max_output_tokens=config.llm.max_output_tokens,
        executor=executor,
        prompt_view_builder=PromptViewBuilder(objects),
    ).propose(context, field_catalog_ref=field_ref)
    hypothesis_ref = objects.put_object(
        "research_hypothesis",
        hypothesis.hypothesis_id,
        hypothesis,
    )
    prompt_refs = tuple(executor.prompt_refs.values())
    if len(prompt_refs) != 1:
        raise RuntimeError("live LLM smoke must persist exactly one prompt")
    return LiveLLMSmokeRun(
        storage_root=str(root),
        hypothesis_ref=hypothesis_ref,
        prompt_ref=prompt_refs[0],
        call_record_refs=tuple(executor.call_record_refs),
    )


def _live_smoke_context() -> ResearchContext:
    """Build the intentionally small, no-market-data research context."""
    return ResearchContext(
        research_goal="Propose one testable daily price-volume factor hypothesis.",
        market="cn_ashare",
        universe="all_a_share",
        frequency="1d",
        available_fields=("open", "high", "low", "close", "volume"),
        operator_catalog_ref=ObjectRef(
            "operator_catalog",
            "live-llm-smoke-catalog",
            "operator-catalog/1.0",
        ),
        evaluation_protocol_ref=ObjectRef(
            "evaluation_protocol",
            "live-llm-smoke-evaluation",
            "evaluation-protocol/1.0",
        ),
        history_summaries=(),
        constraints=(
            "Use only the supplied daily price and volume fields.",
            "Do not claim validation or acceptance.",
        ),
        research_direction=None,
        research_direction_ref=None,
        schema_version="research-context/1.2",
    )
