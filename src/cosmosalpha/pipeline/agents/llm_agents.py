"""Prompt-driven Agent adapters for hypothesis, factor, and feedback steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from cosmosalpha.agents import (
    ExpressionRepairRequest,
    FactorDesign,
    FactorDesignBatch,
    FactorDesignDraftBatch,
    FactorDesignGenerationRequest,
    FactorExpressionRequest,
    FactorProposal,
    FeedbackRequest,
    GenerationSummary,
    HypothesisVerdict,
    ResearchContext,
    ResearchFeedback,
    ResearchHypothesis,
)
from cosmosalpha.core import ObjectRef, stable_sha256, to_json
from cosmosalpha.llm import (
    DEFAULT_PROMPT_ROOT,
    AgentOutputParser,
    AuditedLLMExecutor,
    FeedbackPromptViewBuilder,
    LLMClient,
    LLMOutputError,
    LLMRequest,
    PromptRenderer,
    PromptTemplateRegistry,
    PromptViewBuilder,
)

HYPOTHESIS_AGENT_VERSION = "hypothesis-agent/1.0"
FACTOR_DESIGN_AGENT_VERSION = "factor-design-agent/1.0"
FACTOR_EXPRESSION_AGENT_VERSION = "factor-expression-agent/1.0"
EXPRESSION_REPAIR_AGENT_VERSION = "expression-repair-agent/1.0"
FEEDBACK_AGENT_VERSION = "feedback-agent/2.0"
DEFAULT_MODEL_ALIAS = "replay"
DEFAULT_MAX_INPUT_TOKENS = 16000


@dataclass(frozen=True, slots=True)
class _AgentRuntime:
    client: LLMClient
    registry: PromptTemplateRegistry
    renderer: PromptRenderer
    parser: AgentOutputParser
    model_alias: str
    max_input_tokens: int
    max_output_tokens: int | None
    executor: AuditedLLMExecutor | None

    @classmethod
    def create(
        cls,
        client: LLMClient,
        *,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
    ) -> _AgentRuntime:
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be positive when configured")
        prompt_registry = registry or PromptTemplateRegistry.from_directory()
        return cls(
            client=client,
            registry=prompt_registry,
            renderer=PromptRenderer(prompt_registry),
            parser=parser or AgentOutputParser(),
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )

    def complete_and_parse[ParsedT](
        self,
        prompt_id: str,
        variables: dict[str, object],
        parse: Callable[[str], ParsedT],
    ) -> ParsedT:
        """Render, complete, and parse one Agent draft."""
        rendered = self.renderer.render(
            prompt_id,
            variables,
            max_input_tokens=self.max_input_tokens,
        )
        template = self.registry.get(prompt_id)
        request = LLMRequest(
            prompt=rendered,
            model_alias=self.model_alias,
            max_output_tokens=(
                template.max_output_tokens
                if self.max_output_tokens is None
                else min(template.max_output_tokens, self.max_output_tokens)
            ),
            response_format="json_object",
        )
        if self.executor is not None:
            return self.executor.complete_and_parse(request, parse)
        return parse(self.client.complete(request).content)


class HypothesisAgent:
    """Generate a persisted research hypothesis from a strict LLM draft."""

    def __init__(
        self,
        client: LLMClient,
        *,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
        prompt_view_builder: PromptViewBuilder,
    ) -> None:
        """Create a hypothesis agent."""
        self._runtime = _AgentRuntime.create(
            client,
            registry=registry,
            parser=parser,
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )
        self._prompt_view_builder = prompt_view_builder

    def propose(
        self,
        context: ResearchContext,
        *,
        field_catalog_ref: ObjectRef,
        history_lesson_refs: tuple[ObjectRef, ...] = (),
    ) -> ResearchHypothesis:
        """Generate one hypothesis for a research context."""
        view = self._prompt_view_builder.build_hypothesis(
            field_catalog_ref,
            context.operator_catalog_ref,
            frequency=context.frequency,
            backend="qlib",
            lesson_refs=history_lesson_refs,
        )
        variables = {
            "context_json": to_json(replace(context, history_summaries=())),
            "prompt_view_json": to_json(view),
        }
        draft = self._runtime.complete_and_parse(
            "hypothesis",
            variables,
            self._runtime.parser.parse_hypothesis,
        )
        available_fields = {field.name for field in view.catalog.fields}
        allowed_fields = available_fields.intersection(context.available_fields)
        if not set(draft.required_fields).issubset(allowed_fields):
            raise LLMOutputError(
                "LLM_OUTPUT_SCHEMA_MISMATCH: hypothesis uses unavailable fields"
            )
        hypothesis_id = _stable_prefixed_id(
            "hypothesis",
            {
                "context": context,
                "draft": draft,
                "history_lesson_refs": history_lesson_refs,
            },
        )
        return ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            statement=draft.statement,
            rationale=draft.rationale,
            mechanism=draft.mechanism,
            testable_prediction=draft.testable_prediction,
            expected_direction=draft.expected_direction,
            required_fields=draft.required_fields,
            prediction_horizon=draft.prediction_horizon,
            assumptions=draft.assumptions,
            source_refs=(
                (field_catalog_ref,)
                + (
                    context.operator_catalog_ref,
                    context.evaluation_protocol_ref,
                )
            )
            + (
                ()
                if context.research_direction_ref is None
                else (context.research_direction_ref,)
            )
            + history_lesson_refs,
            schema_version="research-hypothesis/1.0",
        )


class FactorDesignAgent:
    """Generate one or two structured designs without writing DSL."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_view_builder: PromptViewBuilder,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
        prompt_id: str = "factor_design",
    ) -> None:
        """Create a v2 factor design agent."""
        self._runtime = _AgentRuntime.create(
            client,
            registry=registry
            or PromptTemplateRegistry.from_directory(
                DEFAULT_PROMPT_ROOT,
            ),
            parser=parser,
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )
        self._prompt_view_builder = prompt_view_builder
        self._prompt_id = prompt_id

    def generate(self, request: FactorDesignGenerationRequest) -> FactorDesignBatch:
        """Generate and validate structured designs against runtime Catalogs."""
        view = self._prompt_view_builder.build_factor_design(
            request.field_catalog_ref,
            request.operator_catalog_ref,
            frequency=request.frequency,
            backend="qlib",
        )

        def parse_batch(content: str) -> FactorDesignDraftBatch:
            batch = self._runtime.parser.parse_factor_design_batch(content)
            if len(batch.designs) > request.generation_constraints.max_designs:
                raise LLMOutputError(
                    "LLM_OUTPUT_SCHEMA_MISMATCH: too many factor designs"
                )
            available_fields = {field.name for field in view.catalog.fields}
            supported_windows = {
                window
                for operator in view.catalog.operators
                for window in operator.supported_windows
            }
            for design in batch.designs:
                invalid_fields = tuple(
                    sorted(
                        {
                            variable.name
                            for variable in design.variables
                            if variable.name not in available_fields
                        }
                    )
                )
                if invalid_fields:
                    raise LLMOutputError(
                        "LLM_OUTPUT_SCHEMA_MISMATCH: design uses unavailable field(s): "
                        + ", ".join(invalid_fields)
                        + "; variables must contain only exact raw Catalog field names; "
                        "intermediate quantities belong in formulation or signal_definition"
                    )
                if any(
                    choice.window not in supported_windows
                    for choice in design.window_choices
                ):
                    raise LLMOutputError(
                        "LLM_OUTPUT_SCHEMA_MISMATCH: design uses unsupported window"
                    )
            return batch

        draft_batch = self._runtime.complete_and_parse(
            self._prompt_id,
            {
                "request_json": to_json(request),
                "prompt_view_json": to_json(view),
            },
            parse_batch,
        )
        designs = tuple(
            FactorDesign.from_draft(
                draft,
                hypothesis_ref=request.hypothesis_ref,
                frequency=request.frequency,
                field_catalog_ref=request.field_catalog_ref,
                operator_catalog_ref=request.operator_catalog_ref,
                index=index,
            )
            for index, draft in enumerate(draft_batch.designs)
        )
        return FactorDesignBatch(
            hypothesis_id=request.hypothesis.hypothesis_id,
            designs=designs,
            generation_summary=GenerationSummary(
                requested_count=request.generation_constraints.max_designs,
                parsed_count=len(designs),
                dropped_count=0,
                warnings=(),
            ),
            schema_version="factor-design-batch/1.0",
        )


class FactorExpressionAgent:
    """Translate exactly one persisted FactorDesign into CosmosAlpha DSL."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_view_builder: PromptViewBuilder,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
    ) -> None:
        """Create a v2 factor expression agent."""
        self._runtime = _AgentRuntime.create(
            client,
            registry=registry
            or PromptTemplateRegistry.from_directory(
                DEFAULT_PROMPT_ROOT,
            ),
            parser=parser,
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )
        self._prompt_view_builder = prompt_view_builder

    def generate(self, request: FactorExpressionRequest) -> FactorProposal:
        """Generate one expression and inherit all non-expression fields."""
        view = self._prompt_view_builder.build_factor_expression(
            request.field_catalog_ref,
            request.operator_catalog_ref,
            frequency=request.frequency,
            backend="qlib",
        )
        draft = self._runtime.complete_and_parse(
            "factor_expression",
            {
                "request_json": to_json(request),
                "prompt_view_json": to_json(view),
            },
            self._runtime.parser.parse_factor_expression,
        )
        return FactorProposal.from_design(
            request.factor_design,
            draft.proposed_expression,
            generation_attempt=1,
        )


class ExpressionRepairAgent:
    """Repair one invalid expression without changing its FactorDesign."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_view_builder: PromptViewBuilder,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
    ) -> None:
        """Create a validation-aware expression repair agent."""
        self._runtime = _AgentRuntime.create(
            client,
            registry=registry
            or PromptTemplateRegistry.from_directory(
                DEFAULT_PROMPT_ROOT,
            ),
            parser=parser,
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )
        self._prompt_view_builder = prompt_view_builder

    def repair(self, request: ExpressionRepairRequest) -> FactorProposal:
        """Generate a new immutable proposal from deterministic validation errors."""
        view = self._prompt_view_builder.build_expression_repair(
            request.field_catalog_ref,
            request.operator_catalog_ref,
            frequency=request.factor_design.frequency,
            backend="qlib",
        )
        draft = self._runtime.complete_and_parse(
            "expression_repair",
            {
                "request_json": to_json(request),
                "prompt_view_json": to_json(view),
            },
            self._runtime.parser.parse_factor_expression,
        )
        return FactorProposal.from_design(
            request.factor_design,
            draft.proposed_expression,
            generation_attempt=request.generation_attempt,
        )


class FeedbackAgent:
    """Generate v2 research feedback from resolved deterministic evidence."""

    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_view_builder: FeedbackPromptViewBuilder,
        registry: PromptTemplateRegistry | None = None,
        parser: AgentOutputParser | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens: int | None = None,
        executor: AuditedLLMExecutor | None = None,
    ) -> None:
        """Create a feedback agent."""
        self._runtime = _AgentRuntime.create(
            client,
            registry=registry
            or PromptTemplateRegistry.from_directory(
                DEFAULT_PROMPT_ROOT,
            ),
            parser=parser,
            model_alias=model_alias,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            executor=executor,
        )
        self._prompt_view_builder = prompt_view_builder

    def generate(self, request: FeedbackRequest, *, cycle_id: str) -> ResearchFeedback:
        """Generate feedback for one workflow cycle."""
        view = self._prompt_view_builder.build(request)
        draft = self._runtime.complete_and_parse(
            "feedback",
            {"feedback_view_json": to_json(view)},
            self._runtime.parser.parse_feedback,
        )
        verdict = (
            HypothesisVerdict.INCONCLUSIVE
            if not view.evaluations
            else draft.hypothesis_verdict
        )
        feedback_id = _stable_prefixed_id(
            "feedback",
            {
                "cycle_id": cycle_id,
                "draft": draft,
                "evidence_checksum": view.content_checksum,
                "verdict": verdict,
            },
        )
        return ResearchFeedback(
            feedback_id=feedback_id,
            cycle_id=cycle_id,
            hypothesis_verdict=verdict,
            summary=draft.summary,
            supported_findings=draft.supported_findings,
            rejected_findings=draft.rejected_findings,
            failure_analysis=draft.failure_analysis,
            next_actions=draft.next_actions,
            next_hypothesis_hint=draft.next_hypothesis_hint,
            evidence_refs=view.source_refs,
            schema_version="research-feedback/2.0",
        )


def _stable_prefixed_id(prefix: str, value: object) -> str:
    return f"{prefix}-{stable_sha256(value)[:24]}"
