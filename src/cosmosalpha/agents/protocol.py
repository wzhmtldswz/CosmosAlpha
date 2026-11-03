"""Immutable agent protocol objects shared across workflow boundaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from cosmosalpha.core import (
    ArtifactRef,
    CoreValidationError,
    FailureInfo,
    ObjectRef,
    WarningInfo,
    stable_sha256,
)

_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _require_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_schema_version(value: object) -> str:
    version = _require_string(value, "schema_version")
    if _SCHEMA_VERSION_PATTERN.fullmatch(version) is None:
        raise CoreValidationError(
            "schema_version must match '<schema-name>/<major>.<minor>'"
        )
    return version


def _require_code(value: object, field_name: str) -> str:
    code = _require_string(value, field_name)
    if _CODE_PATTERN.fullmatch(code) is None:
        raise CoreValidationError(
            f"{field_name} must contain only uppercase letters, digits, and underscores"
        )
    return code


def _require_tuple_of(
    value: object,
    item_type: type[object],
    field_name: str,
) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    if not all(isinstance(item, item_type) for item in value):
        raise CoreValidationError(
            f"every item in {field_name} must be {item_type.__name__}"
        )


def _require_string_tuple(value: object, field_name: str) -> None:
    _require_tuple_of(value, str, field_name)
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


def _require_code_tuple(value: object, field_name: str) -> None:
    _require_tuple_of(value, str, field_name)
    for index, item in enumerate(value):
        _require_code(item, f"{field_name}[{index}]")


def _require_non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise CoreValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise CoreValidationError(f"{field_name} must be a boolean")
    return value


def _require_finite_float(value: object, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise CoreValidationError(f"{field_name} must be a finite float")
    return value


def _require_coverage(value: object, field_name: str) -> float:
    coverage = _require_finite_float(value, field_name)
    if coverage < 0.0 or coverage > 1.0:
        raise CoreValidationError(f"{field_name} must be between 0.0 and 1.0")
    return coverage


def _require_date_range(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise CoreValidationError(f"{field_name} must be a two-item date tuple")
    start, end = value
    if type(start) is not date or type(end) is not date:
        raise CoreValidationError(f"{field_name} must contain date values")
    if start > end:
        raise CoreValidationError(f"{field_name} start must not be after end")


def _require_utc_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise CoreValidationError(f"{field_name} must be a datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None or offset != timedelta(0):
        raise CoreValidationError(f"{field_name} must use UTC timezone information")
    return value


def _require_optional_utc_datetime(
    value: object,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _require_utc_datetime(value, field_name)


def _require_object_ref_tuple(value: object, field_name: str) -> None:
    _require_tuple_of(value, ObjectRef, field_name)


def _require_artifact_ref_tuple(value: object, field_name: str) -> None:
    _require_tuple_of(value, ArtifactRef, field_name)


def _require_catalog_ref(value: object, object_type: str) -> ObjectRef:
    if not isinstance(value, ObjectRef) or value.object_type != object_type:
        raise CoreValidationError(f"{object_type}_ref must reference {object_type}")
    return value


def _require_warning_tuple(value: object, field_name: str) -> None:
    _require_tuple_of(value, WarningInfo, field_name)


class ExperimentStatus(StrEnum):
    """Stable lifecycle states for an experiment record."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    COMPUTED = "computed"
    PREPROCESSED = "preprocessed"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class BranchStatus(StrEnum):
    """Terminal states for one independently executed research branch."""

    COMPLETED = "completed"
    FAILED = "failed"


class HypothesisVerdict(StrEnum):
    """Evidence-based interpretation of one research hypothesis."""

    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ResearchDirection:
    """One stable, configured research direction."""

    direction_id: str
    description: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate direction identity and content."""
        _require_string(self.direction_id, "direction_id")
        _require_string(self.description, "description")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class HypothesisDraft:
    """LLM-owned content for a falsifiable research hypothesis."""

    statement: str
    rationale: str
    mechanism: str
    testable_prediction: str
    expected_direction: str
    required_fields: tuple[str, ...]
    prediction_horizon: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate model-owned draft content without protocol-owned fields."""
        _require_string(self.statement, "statement")
        _require_string(self.rationale, "rationale")
        _require_string(self.mechanism, "mechanism")
        _require_string(self.testable_prediction, "testable_prediction")
        _require_string(self.expected_direction, "expected_direction")
        _require_string_tuple(self.required_fields, "required_fields")
        _require_string(self.prediction_horizon, "prediction_horizon")
        _require_string_tuple(self.assumptions, "assumptions")


@dataclass(frozen=True, slots=True)
class FactorVariableDraft:
    """One market field and its role in a structured factor design."""

    name: str
    role: str

    def __post_init__(self) -> None:
        """Validate variable semantics."""
        _require_string(self.name, "name")
        _require_string(self.role, "role")


@dataclass(frozen=True, slots=True)
class WindowChoiceDraft:
    """One supported lookback and its design rationale."""

    window: int
    rationale: str

    def __post_init__(self) -> None:
        """Validate window choice content."""
        if type(self.window) is not int or self.window <= 0:
            raise CoreValidationError("window must be a positive integer")
        _require_string(self.rationale, "rationale")


@dataclass(frozen=True, slots=True)
class FactorDesignDraft:
    """LLM-owned structured semantics for one factor candidate."""

    name: str
    description: str
    mechanism_mapping: str
    signal_definition: str
    variables: tuple[FactorVariableDraft, ...]
    formulation: str
    window_choices: tuple[WindowChoiceDraft, ...]
    expected_direction: str
    failure_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate structured design content."""
        _require_string(self.name, "name")
        _require_string(self.description, "description")
        _require_string(self.mechanism_mapping, "mechanism_mapping")
        _require_string(self.signal_definition, "signal_definition")
        _require_tuple_of(self.variables, FactorVariableDraft, "variables")
        if not self.variables:
            raise CoreValidationError("variables must not be empty")
        variable_names = tuple(variable.name for variable in self.variables)
        if len(set(variable_names)) != len(variable_names):
            raise CoreValidationError("variable names must be unique")
        _require_string(self.formulation, "formulation")
        _require_tuple_of(
            self.window_choices,
            WindowChoiceDraft,
            "window_choices",
        )
        windows = tuple(choice.window for choice in self.window_choices)
        if len(set(windows)) != len(windows):
            raise CoreValidationError("window choices must be unique")
        _require_string(self.expected_direction, "expected_direction")
        _require_string_tuple(self.failure_conditions, "failure_conditions")


@dataclass(frozen=True, slots=True)
class FactorDesignDraftBatch:
    """LLM-owned batch of one or two structured factor designs."""

    designs: tuple[FactorDesignDraft, ...]

    def __post_init__(self) -> None:
        """Validate draft design collection."""
        _require_tuple_of(self.designs, FactorDesignDraft, "designs")
        if len(self.designs) not in (1, 2):
            raise CoreValidationError("designs must contain one or two items")
        names = tuple(design.name for design in self.designs)
        if len(set(names)) != len(names):
            raise CoreValidationError("design names must be unique")


@dataclass(frozen=True, slots=True)
class FactorExpressionDraft:
    """LLM-owned CosmosAlpha DSL translation for one FactorDesign."""

    proposed_expression: str

    def __post_init__(self) -> None:
        """Validate expression draft content."""
        _require_string(self.proposed_expression, "proposed_expression")


@dataclass(frozen=True, slots=True)
class FeedbackDraft:
    """LLM-owned feedback that explains deterministic evidence."""

    summary: str
    hypothesis_verdict: HypothesisVerdict
    supported_findings: tuple[str, ...]
    rejected_findings: tuple[str, ...]
    failure_analysis: tuple[str, ...]
    next_actions: tuple[str, ...]
    next_hypothesis_hint: str | None

    def __post_init__(self) -> None:
        """Validate feedback draft fields."""
        _require_string(self.summary, "summary")
        if not isinstance(self.hypothesis_verdict, HypothesisVerdict):
            raise CoreValidationError("hypothesis_verdict must be HypothesisVerdict")
        _require_string_tuple(self.supported_findings, "supported_findings")
        _require_string_tuple(self.rejected_findings, "rejected_findings")
        _require_string_tuple(self.failure_analysis, "failure_analysis")
        _require_string_tuple(self.next_actions, "next_actions")
        _require_optional_string(self.next_hypothesis_hint, "next_hypothesis_hint")


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """Controlled context passed to the hypothesis agent."""

    research_goal: str
    market: str
    universe: str
    frequency: str
    available_fields: tuple[str, ...]
    operator_catalog_ref: ObjectRef
    evaluation_protocol_ref: ObjectRef
    history_summaries: tuple[str, ...]
    constraints: tuple[str, ...]
    research_direction: ResearchDirection | None
    research_direction_ref: ObjectRef | None
    schema_version: str

    def __post_init__(self) -> None:
        """Validate research context metadata."""
        _require_string(self.research_goal, "research_goal")
        _require_string(self.market, "market")
        _require_string(self.universe, "universe")
        _require_string(self.frequency, "frequency")
        _require_string_tuple(self.available_fields, "available_fields")
        if not isinstance(self.operator_catalog_ref, ObjectRef):
            raise CoreValidationError("operator_catalog_ref must be ObjectRef")
        if not isinstance(self.evaluation_protocol_ref, ObjectRef):
            raise CoreValidationError("evaluation_protocol_ref must be ObjectRef")
        _require_string_tuple(self.history_summaries, "history_summaries")
        _require_string_tuple(self.constraints, "constraints")
        if (self.research_direction is None) != (self.research_direction_ref is None):
            raise CoreValidationError(
                "research_direction and research_direction_ref must both be set or absent"
            )
        if self.research_direction is not None:
            if not isinstance(self.research_direction, ResearchDirection):
                raise CoreValidationError(
                    "research_direction must be ResearchDirection or None"
                )
            if not isinstance(self.research_direction_ref, ObjectRef):
                raise CoreValidationError(
                    "research_direction_ref must be ObjectRef or None"
                )
            if (
                self.research_direction_ref.object_type != "research_direction"
                or self.research_direction_ref.object_id
                != self.research_direction.direction_id
                or self.research_direction_ref.schema_version
                != self.research_direction.schema_version
            ):
                raise CoreValidationError(
                    "research_direction_ref must reference research_direction"
                )
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    """Persisted terminal result returned across a process boundary by reference."""

    run_id: str
    branch_id: str
    branch_index: int
    direction_ref: ObjectRef
    status: BranchStatus
    cycle_ids: tuple[str, ...]
    experiment_refs: tuple[ObjectRef, ...]
    terminal_experiment_ref: ObjectRef | None
    failure: FailureInfo | None
    prompt_render_refs: tuple[ObjectRef, ...]
    llm_call_record_refs: tuple[ObjectRef, ...]
    llm_raw_response_refs: tuple[ArtifactRef, ...]
    schema_version: str

    def __post_init__(self) -> None:
        """Validate branch result identity, status, and audit references."""
        _require_string(self.run_id, "run_id")
        _require_string(self.branch_id, "branch_id")
        if type(self.branch_index) is not int or self.branch_index < 1:
            raise CoreValidationError("branch_index must be a positive integer")
        if not isinstance(self.direction_ref, ObjectRef):
            raise CoreValidationError("direction_ref must be ObjectRef")
        if self.direction_ref.object_type != "research_direction":
            raise CoreValidationError(
                "direction_ref must reference a research_direction"
            )
        if not isinstance(self.status, BranchStatus):
            raise CoreValidationError("status must be BranchStatus")
        _require_string_tuple(self.cycle_ids, "cycle_ids")
        _require_object_ref_tuple(self.experiment_refs, "experiment_refs")
        if self.terminal_experiment_ref is not None and not isinstance(
            self.terminal_experiment_ref, ObjectRef
        ):
            raise CoreValidationError(
                "terminal_experiment_ref must be ObjectRef or None"
            )
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        if self.status is BranchStatus.COMPLETED:
            if (
                self.failure is not None
                or not self.experiment_refs
                or self.terminal_experiment_ref != self.experiment_refs[-1]
            ):
                raise CoreValidationError(
                    "completed branch requires its final experiment and no failure"
                )
        elif self.failure is None or self.terminal_experiment_ref is not None:
            raise CoreValidationError(
                "failed branch requires failure and no terminal experiment"
            )
        _require_object_ref_tuple(self.prompt_render_refs, "prompt_render_refs")
        _require_object_ref_tuple(self.llm_call_record_refs, "llm_call_record_refs")
        _require_artifact_ref_tuple(self.llm_raw_response_refs, "llm_raw_response_refs")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ResearchHypothesis:
    """Versioned research hypothesis produced by an agent adapter."""

    hypothesis_id: str
    statement: str
    rationale: str
    mechanism: str
    testable_prediction: str
    expected_direction: str
    required_fields: tuple[str, ...]
    prediction_horizon: str
    assumptions: tuple[str, ...]
    source_refs: tuple[ObjectRef, ...]
    schema_version: str

    @classmethod
    def from_fixed_designs(
        cls,
        drafts: tuple[FactorDesignDraft, ...],
        *,
        hypothesis_id: str,
        frequency: str,
        source: str,
        source_refs: tuple[ObjectRef, ...],
    ) -> ResearchHypothesis:
        """Record the provenance of supplied expressions without an LLM claim."""
        if not drafts:
            raise CoreValidationError("fixed designs must not be empty")
        _require_tuple_of(drafts, FactorDesignDraft, "drafts")
        _require_string(source, "source")
        return cls(
            hypothesis_id=hypothesis_id,
            statement=f"Evaluate fixed factor expressions supplied by {source}.",
            rationale=f"The factor definitions were supplied by {source}.",
            mechanism="Supplied expressions are evaluated under the configured protocol.",
            testable_prediction="The supplied factors can be evaluated deterministically.",
            expected_direction="unknown",
            required_fields=tuple(
                sorted(
                    {variable.name for draft in drafts for variable in draft.variables}
                )
            ),
            prediction_horizon=frequency,
            assumptions=(
                "Provenance describes supplied definitions, not generated research.",
            ),
            source_refs=source_refs,
            schema_version="research-hypothesis/1.0",
        )

    def __post_init__(self) -> None:
        """Validate persisted hypothesis fields."""
        _require_string(self.hypothesis_id, "hypothesis_id")
        _require_string(self.statement, "statement")
        _require_string(self.rationale, "rationale")
        _require_string(self.mechanism, "mechanism")
        _require_string(self.testable_prediction, "testable_prediction")
        _require_string(self.expected_direction, "expected_direction")
        _require_string_tuple(self.required_fields, "required_fields")
        _require_string(self.prediction_horizon, "prediction_horizon")
        _require_string_tuple(self.assumptions, "assumptions")
        _require_object_ref_tuple(self.source_refs, "source_refs")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class FactorDesignConstraints:
    """Deterministic limits for one FactorDesign LLM call."""

    max_designs: int

    def __post_init__(self) -> None:
        """Validate design generation limits."""
        if type(self.max_designs) is not int or self.max_designs not in (1, 2):
            raise CoreValidationError("max_designs must be 1 or 2")


@dataclass(frozen=True, slots=True)
class FactorDesignGenerationRequest:
    """Input contract for turning one hypothesis into structured designs."""

    hypothesis_ref: ObjectRef
    hypothesis: ResearchHypothesis
    field_catalog_ref: ObjectRef
    operator_catalog_ref: ObjectRef
    frequency: str
    universe: str
    existing_factor_refs: tuple[ObjectRef, ...]
    generation_constraints: FactorDesignConstraints
    schema_version: str
    # ── 可选父因子结构化上下文（MCTS expand 填充，对齐 Graph GRAPH_GUIDANCE
    #    的做法：让 design 无需从假设散文推断父表达式与优化目标）──
    parent_expression: str | None = None
    focus_dimension: str | None = None
    parent_scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate design generation references and limits."""
        if not isinstance(self.hypothesis_ref, ObjectRef):
            raise CoreValidationError("hypothesis_ref must be ObjectRef")
        if not isinstance(self.hypothesis, ResearchHypothesis):
            raise CoreValidationError("hypothesis must be ResearchHypothesis")
        if (
            self.hypothesis_ref.object_type != "research_hypothesis"
            or self.hypothesis_ref.object_id != self.hypothesis.hypothesis_id
            or self.hypothesis_ref.schema_version != self.hypothesis.schema_version
        ):
            raise CoreValidationError("hypothesis_ref must reference hypothesis")
        _require_catalog_ref(self.field_catalog_ref, "field_catalog")
        _require_catalog_ref(self.operator_catalog_ref, "operator_catalog")
        _require_string(self.frequency, "frequency")
        _require_string(self.universe, "universe")
        _require_object_ref_tuple(self.existing_factor_refs, "existing_factor_refs")
        if not isinstance(self.generation_constraints, FactorDesignConstraints):
            raise CoreValidationError(
                "generation_constraints must be FactorDesignConstraints"
            )
        if self.schema_version != "factor-design-generation-request/1.1":
            raise CoreValidationError(
                "invalid factor design generation request schema_version"
            )
        if self.parent_expression is not None and (
            type(self.parent_expression) is not str
            or not self.parent_expression
            or self.parent_expression != self.parent_expression.strip()
        ):
            raise CoreValidationError(
                "parent_expression must be non-empty text when provided"
            )
        if self.focus_dimension is not None and (
            type(self.focus_dimension) is not str
            or not self.focus_dimension
            or self.focus_dimension != self.focus_dimension.strip()
        ):
            raise CoreValidationError(
                "focus_dimension must be non-empty text when provided"
            )
        if not isinstance(self.parent_scores, tuple):
            raise CoreValidationError("parent_scores must be a tuple")
        for entry in self.parent_scores:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or type(entry[0]) is not str
                or type(entry[1]) is not float
            ):
                raise CoreValidationError(
                    "parent_scores entries must be (name, float) pairs"
                )


@dataclass(frozen=True, slots=True)
class FactorDesign:
    """Persisted structured factor semantics associated with one hypothesis."""

    design_id: str
    hypothesis_id: str
    hypothesis_ref: ObjectRef
    name: str
    description: str
    mechanism_mapping: str
    signal_definition: str
    variables: tuple[FactorVariableDraft, ...]
    formulation: str
    window_choices: tuple[WindowChoiceDraft, ...]
    expected_direction: str
    failure_conditions: tuple[str, ...]
    frequency: str
    field_catalog_ref: ObjectRef
    operator_catalog_ref: ObjectRef
    schema_version: str

    @classmethod
    def from_draft(
        cls,
        draft: FactorDesignDraft,
        *,
        hypothesis_ref: ObjectRef,
        frequency: str,
        field_catalog_ref: ObjectRef,
        operator_catalog_ref: ObjectRef,
        index: int = 0,
    ) -> FactorDesign:
        """Attach deterministic identity and provenance to a design draft."""
        if not isinstance(draft, FactorDesignDraft):
            raise CoreValidationError("draft must be FactorDesignDraft")
        if not isinstance(hypothesis_ref, ObjectRef):
            raise CoreValidationError("hypothesis_ref must be ObjectRef")
        _require_non_negative_int(index, "index")
        return cls(
            design_id="design-"
            + stable_sha256(
                {
                    "draft": draft,
                    "hypothesis_id": hypothesis_ref.object_id,
                    "index": index,
                }
            )[:24],
            hypothesis_id=hypothesis_ref.object_id,
            hypothesis_ref=hypothesis_ref,
            name=draft.name,
            description=draft.description,
            mechanism_mapping=draft.mechanism_mapping,
            signal_definition=draft.signal_definition,
            variables=draft.variables,
            formulation=draft.formulation,
            window_choices=draft.window_choices,
            expected_direction=draft.expected_direction,
            failure_conditions=draft.failure_conditions,
            frequency=frequency,
            field_catalog_ref=field_catalog_ref,
            operator_catalog_ref=operator_catalog_ref,
            schema_version="factor-design/1.0",
        )

    def __post_init__(self) -> None:
        """Validate persisted design identity and semantics."""
        _require_string(self.design_id, "design_id")
        _require_string(self.hypothesis_id, "hypothesis_id")
        if (
            not isinstance(self.hypothesis_ref, ObjectRef)
            or self.hypothesis_ref.object_type != "research_hypothesis"
            or self.hypothesis_ref.object_id != self.hypothesis_id
        ):
            raise CoreValidationError(
                "hypothesis_ref must reference the design hypothesis"
            )
        FactorDesignDraft(
            name=self.name,
            description=self.description,
            mechanism_mapping=self.mechanism_mapping,
            signal_definition=self.signal_definition,
            variables=self.variables,
            formulation=self.formulation,
            window_choices=self.window_choices,
            expected_direction=self.expected_direction,
            failure_conditions=self.failure_conditions,
        )
        _require_string(self.frequency, "frequency")
        _require_catalog_ref(self.field_catalog_ref, "field_catalog")
        _require_catalog_ref(self.operator_catalog_ref, "operator_catalog")
        if self.schema_version != "factor-design/1.0":
            raise CoreValidationError("invalid factor design schema_version")


@dataclass(frozen=True, slots=True)
class FactorDesignBatch:
    """Persisted batch index for structured factor designs."""

    hypothesis_id: str
    designs: tuple[FactorDesign, ...]
    generation_summary: GenerationSummary
    schema_version: str

    def __post_init__(self) -> None:
        """Validate design batch consistency."""
        _require_string(self.hypothesis_id, "hypothesis_id")
        _require_tuple_of(self.designs, FactorDesign, "designs")
        if not isinstance(self.generation_summary, GenerationSummary):
            raise CoreValidationError("generation_summary must be GenerationSummary")
        if len(self.designs) != self.generation_summary.parsed_count:
            raise CoreValidationError("design count must match parsed_count")
        if not self.designs:
            raise CoreValidationError("designs must not be empty")
        if any(design.hypothesis_id != self.hypothesis_id for design in self.designs):
            raise CoreValidationError("all designs must share the batch hypothesis_id")
        names = tuple(design.name for design in self.designs)
        if len(set(names)) != len(names):
            raise CoreValidationError("design names must be unique")
        if self.schema_version != "factor-design-batch/1.0":
            raise CoreValidationError("invalid factor design batch schema_version")


@dataclass(frozen=True, slots=True)
class ExpressionConstraints:
    """Deterministic expression limits shown to the expression Agent."""

    max_characters: int
    max_depth: int
    max_operator_nodes: int
    max_fields: int

    def __post_init__(self) -> None:
        """Validate positive expression limits."""
        for field_name in (
            "max_characters",
            "max_depth",
            "max_operator_nodes",
            "max_fields",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise CoreValidationError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FactorExpressionRequest:
    """Input contract for translating one FactorDesign into CosmosAlpha DSL."""

    factor_design_ref: ObjectRef
    factor_design: FactorDesign
    field_catalog_ref: ObjectRef
    operator_catalog_ref: ObjectRef
    existing_factor_refs: tuple[ObjectRef, ...]
    expression_constraints: ExpressionConstraints
    frequency: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate design identity, Catalog provenance, and constraints."""
        if not isinstance(self.factor_design_ref, ObjectRef):
            raise CoreValidationError("factor_design_ref must be ObjectRef")
        if not isinstance(self.factor_design, FactorDesign):
            raise CoreValidationError("factor_design must be FactorDesign")
        if (
            self.factor_design_ref.object_type != "factor_design"
            or self.factor_design_ref.object_id != self.factor_design.design_id
            or self.factor_design_ref.schema_version
            != self.factor_design.schema_version
        ):
            raise CoreValidationError("factor_design_ref must reference factor_design")
        _require_catalog_ref(self.field_catalog_ref, "field_catalog")
        _require_catalog_ref(self.operator_catalog_ref, "operator_catalog")
        if (
            self.field_catalog_ref != self.factor_design.field_catalog_ref
            or self.operator_catalog_ref != self.factor_design.operator_catalog_ref
        ):
            raise CoreValidationError(
                "expression request Catalogs must match factor design"
            )
        _require_object_ref_tuple(self.existing_factor_refs, "existing_factor_refs")
        if not isinstance(self.expression_constraints, ExpressionConstraints):
            raise CoreValidationError(
                "expression_constraints must be ExpressionConstraints"
            )
        _require_string(self.frequency, "frequency")
        if self.frequency != self.factor_design.frequency:
            raise CoreValidationError(
                "expression request frequency must match factor design"
            )
        if self.schema_version != "factor-expression-request/1.0":
            raise CoreValidationError(
                "invalid factor expression request schema_version"
            )


@dataclass(frozen=True, slots=True)
class FactorProposal:
    """Persisted v2 expression proposal for one FactorDesign."""

    proposal_id: str
    factor_design_ref: ObjectRef
    hypothesis_id: str
    name: str
    proposed_expression: str
    description: str
    frequency: str
    expected_direction: str
    generation_attempt: int
    schema_version: str

    @classmethod
    def from_design(
        cls,
        design: FactorDesign,
        proposed_expression: str,
        *,
        generation_attempt: int = 1,
    ) -> FactorProposal:
        """Create an expression attempt inheriting its design's provenance."""
        if not isinstance(design, FactorDesign):
            raise CoreValidationError("design must be FactorDesign")
        return cls(
            proposal_id="proposal-"
            + stable_sha256(
                {
                    "design_id": design.design_id,
                    "generation_attempt": generation_attempt,
                    "proposed_expression": proposed_expression,
                }
            )[:24],
            factor_design_ref=ObjectRef(
                "factor_design", design.design_id, design.schema_version
            ),
            hypothesis_id=design.hypothesis_id,
            name=design.name,
            proposed_expression=proposed_expression,
            description=design.description,
            frequency=design.frequency,
            expected_direction=design.expected_direction,
            generation_attempt=generation_attempt,
            schema_version="factor-proposal/2.0",
        )

    def __post_init__(self) -> None:
        """Validate v2 proposal identity and inherited design fields."""
        _require_string(self.proposal_id, "proposal_id")
        if (
            not isinstance(self.factor_design_ref, ObjectRef)
            or self.factor_design_ref.object_type != "factor_design"
        ):
            raise CoreValidationError("factor_design_ref must reference a FactorDesign")
        _require_string(self.hypothesis_id, "hypothesis_id")
        _require_string(self.name, "name")
        _require_string(self.proposed_expression, "proposed_expression")
        _require_string(self.description, "description")
        _require_string(self.frequency, "frequency")
        _require_string(self.expected_direction, "expected_direction")
        if type(self.generation_attempt) is not int or self.generation_attempt < 1:
            raise CoreValidationError("generation_attempt must be a positive integer")
        if self.schema_version != "factor-proposal/2.0":
            raise CoreValidationError("invalid v2 factor proposal schema_version")


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Parsing summary for LLM factor proposal drafts."""

    requested_count: int
    parsed_count: int
    dropped_count: int
    warnings: tuple[WarningInfo, ...]

    def __post_init__(self) -> None:
        """Validate generation summary counts and warnings."""
        requested = _require_non_negative_int(self.requested_count, "requested_count")
        parsed = _require_non_negative_int(self.parsed_count, "parsed_count")
        dropped = _require_non_negative_int(self.dropped_count, "dropped_count")
        if requested not in (1, 2):
            raise CoreValidationError("requested_count must be 1 or 2")
        if parsed + dropped > requested:
            raise CoreValidationError(
                "parsed_count plus dropped_count must not exceed requested_count"
            )
        _require_warning_tuple(self.warnings, "warnings")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Machine-readable expression validation issue."""

    code: str
    severity: str
    location: str
    message: str

    def __post_init__(self) -> None:
        """Validate validation issue metadata."""
        _require_code(self.code, "code")
        _require_string(self.severity, "severity")
        _require_string(self.location, "location")
        _require_string(self.message, "message")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Deterministic expression validation report."""

    proposal_id: str
    factor_id: str | None
    is_valid: bool
    normalized_expression: str | None
    required_fields: tuple[str, ...]
    required_lookback: int
    issues: tuple[ValidationIssue, ...]
    validator_version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate expression report state."""
        _require_string(self.proposal_id, "proposal_id")
        _require_optional_string(self.factor_id, "factor_id")
        is_valid = _require_bool(self.is_valid, "is_valid")
        _require_optional_string(self.normalized_expression, "normalized_expression")
        _require_string_tuple(self.required_fields, "required_fields")
        _require_non_negative_int(self.required_lookback, "required_lookback")
        _require_tuple_of(self.issues, ValidationIssue, "issues")
        _require_string(self.validator_version, "validator_version")
        _require_schema_version(self.schema_version)
        if is_valid and self.factor_id is None:
            raise CoreValidationError("valid report must include factor_id")
        if is_valid and self.normalized_expression is None:
            raise CoreValidationError("valid report must include normalized_expression")
        if not is_valid and self.factor_id is not None:
            raise CoreValidationError("invalid report must not include factor_id")


@dataclass(frozen=True, slots=True)
class ExpressionRepairRequest:
    """Input contract for repairing one deterministically invalid expression."""

    factor_design_ref: ObjectRef
    factor_design: FactorDesign
    factor_proposal_ref: ObjectRef
    factor_proposal: FactorProposal
    validation_report_ref: ObjectRef
    validation_report: ValidationReport
    field_catalog_ref: ObjectRef
    operator_catalog_ref: ObjectRef
    generation_attempt: int
    max_validation_repair_attempts: int
    schema_version: str

    def __post_init__(self) -> None:
        """Validate repair provenance, attempt budget, and evidence."""
        if not isinstance(self.factor_design, FactorDesign):
            raise CoreValidationError("factor_design must be FactorDesign")
        if (
            not isinstance(self.factor_design_ref, ObjectRef)
            or self.factor_design_ref.object_type != "factor_design"
            or self.factor_design_ref.object_id != self.factor_design.design_id
            or self.factor_design_ref.schema_version
            != self.factor_design.schema_version
        ):
            raise CoreValidationError("factor_design_ref must reference factor_design")
        if not isinstance(self.factor_proposal, FactorProposal):
            raise CoreValidationError("factor_proposal must be FactorProposal")
        if (
            not isinstance(self.factor_proposal_ref, ObjectRef)
            or self.factor_proposal_ref.object_type != "factor_proposal"
            or self.factor_proposal_ref.object_id != self.factor_proposal.proposal_id
            or self.factor_proposal_ref.schema_version
            != self.factor_proposal.schema_version
        ):
            raise CoreValidationError(
                "factor_proposal_ref must reference factor_proposal"
            )
        if self.factor_proposal.factor_design_ref != self.factor_design_ref:
            raise CoreValidationError(
                "factor proposal must reference the repair FactorDesign"
            )
        if not isinstance(self.validation_report, ValidationReport):
            raise CoreValidationError("validation_report must be ValidationReport")
        if (
            not isinstance(self.validation_report_ref, ObjectRef)
            or self.validation_report_ref.object_type != "validation_report"
            or self.validation_report_ref.schema_version
            != self.validation_report.schema_version
        ):
            raise CoreValidationError(
                "validation_report_ref must reference validation_report"
            )
        if (
            self.validation_report.proposal_id != self.factor_proposal.proposal_id
            or self.validation_report.is_valid
            or not self.validation_report.issues
        ):
            raise CoreValidationError(
                "repair requires the proposal's invalid ValidationReport"
            )
        _require_catalog_ref(self.field_catalog_ref, "field_catalog")
        _require_catalog_ref(self.operator_catalog_ref, "operator_catalog")
        if (
            self.field_catalog_ref != self.factor_design.field_catalog_ref
            or self.operator_catalog_ref != self.factor_design.operator_catalog_ref
        ):
            raise CoreValidationError("repair Catalogs must match factor design")
        if type(
            self.max_validation_repair_attempts
        ) is not int or self.max_validation_repair_attempts not in (1, 2):
            raise CoreValidationError("max_validation_repair_attempts must be 1 or 2")
        if (
            type(self.generation_attempt) is not int
            or self.generation_attempt != self.factor_proposal.generation_attempt + 1
            or self.generation_attempt > self.max_validation_repair_attempts + 1
        ):
            raise CoreValidationError(
                "generation_attempt must be the next attempt within repair budget"
            )
        if self.schema_version != "expression-repair-request/1.0":
            raise CoreValidationError(
                "invalid expression repair request schema_version"
            )


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """Authoritative validated factor specification."""

    factor_id: str
    proposal_id: str
    hypothesis_id: str
    name: str
    expression: str
    description: str
    frequency: str
    required_fields: tuple[str, ...]
    expected_direction: str
    operator_set_version: str
    field_schema_version: str
    expression_schema_version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate factor specification fields."""
        _require_string(self.factor_id, "factor_id")
        _require_string(self.proposal_id, "proposal_id")
        _require_string(self.hypothesis_id, "hypothesis_id")
        _require_string(self.name, "name")
        _require_string(self.expression, "expression")
        _require_string(self.description, "description")
        _require_string(self.frequency, "frequency")
        _require_string_tuple(self.required_fields, "required_fields")
        _require_string(self.expected_direction, "expected_direction")
        _require_string(self.operator_set_version, "operator_set_version")
        _require_schema_version(self.field_schema_version)
        _require_schema_version(self.expression_schema_version)
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class FactorValidationResult:
    """Validation result tying a proposal to its report and optional spec."""

    proposal_id: str
    report: ValidationReport
    factor_spec: FactorSpec | None
    schema_version: str

    def __post_init__(self) -> None:
        """Validate result/report/spec consistency."""
        _require_string(self.proposal_id, "proposal_id")
        if not isinstance(self.report, ValidationReport):
            raise CoreValidationError("report must be ValidationReport")
        if self.report.proposal_id != self.proposal_id:
            raise CoreValidationError("report proposal_id must match result")
        if self.factor_spec is not None and not isinstance(
            self.factor_spec, FactorSpec
        ):
            raise CoreValidationError("factor_spec must be FactorSpec or None")
        if self.report.is_valid:
            if self.factor_spec is None:
                raise CoreValidationError("valid result must include factor_spec")
            if self.factor_spec.proposal_id != self.proposal_id:
                raise CoreValidationError("factor_spec proposal_id must match result")
            if self.factor_spec.factor_id != self.report.factor_id:
                raise CoreValidationError("factor_spec factor_id must match report")
        elif self.factor_spec is not None:
            raise CoreValidationError("invalid result must not include factor_spec")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class FactorCalculationResult:
    """Metadata for deterministic factor value calculation."""

    factor_id: str
    validation_report_ref: ObjectRef
    data_snapshot_id: str
    mask_data_ref: ArtifactRef
    factor_values_ref: ArtifactRef
    date_range: tuple[date, date]
    sample_count: int
    missing_count: int
    eligible_mask_applied: bool
    calculation_version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate calculation result metadata."""
        _require_string(self.factor_id, "factor_id")
        if not isinstance(self.validation_report_ref, ObjectRef):
            raise CoreValidationError("validation_report_ref must be ObjectRef")
        _require_string(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.mask_data_ref, ArtifactRef):
            raise CoreValidationError("mask_data_ref must be ArtifactRef")
        if not isinstance(self.factor_values_ref, ArtifactRef):
            raise CoreValidationError("factor_values_ref must be ArtifactRef")
        _require_date_range(self.date_range, "date_range")
        _require_non_negative_int(self.sample_count, "sample_count")
        _require_non_negative_int(self.missing_count, "missing_count")
        if self.missing_count > self.sample_count:
            raise CoreValidationError("missing_count must not exceed sample_count")
        if (
            _require_bool(self.eligible_mask_applied, "eligible_mask_applied")
            is not True
        ):
            raise CoreValidationError("eligible_mask_applied must be true")
        _require_string(self.calculation_version, "calculation_version")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class FactorPreprocessingResult:
    """Metadata for deterministic factor preprocessing artifacts."""

    factor_id: str
    data_snapshot_id: str
    input_factor_values_ref: ArtifactRef
    eligible_factor_values_ref: ArtifactRef
    winsorized_factor_values_ref: ArtifactRef
    neutralized_factor_values_ref: ArtifactRef
    evaluated_factor_values_ref: ArtifactRef
    forward_return_ref: ArtifactRef
    exposure_data_ref: ArtifactRef
    mask_data_ref: ArtifactRef
    preprocessing_protocol_id: str
    preprocessing_protocol_version: str
    neutralization_applied: bool
    excluded_sample_count: int
    effective_sample_count: int
    warnings: tuple[WarningInfo, ...]
    schema_version: str

    def __post_init__(self) -> None:
        """Validate preprocessing result metadata."""
        _require_string(self.factor_id, "factor_id")
        _require_string(self.data_snapshot_id, "data_snapshot_id")
        for field_name in (
            "input_factor_values_ref",
            "eligible_factor_values_ref",
            "winsorized_factor_values_ref",
            "neutralized_factor_values_ref",
            "evaluated_factor_values_ref",
            "forward_return_ref",
            "exposure_data_ref",
            "mask_data_ref",
        ):
            if not isinstance(getattr(self, field_name), ArtifactRef):
                raise CoreValidationError(f"{field_name} must be ArtifactRef")
        _require_string(self.preprocessing_protocol_id, "preprocessing_protocol_id")
        _require_string(
            self.preprocessing_protocol_version,
            "preprocessing_protocol_version",
        )
        if (
            _require_bool(self.neutralization_applied, "neutralization_applied")
            is not True
        ):
            raise CoreValidationError("neutralization_applied must be true")
        _require_non_negative_int(self.excluded_sample_count, "excluded_sample_count")
        _require_non_negative_int(self.effective_sample_count, "effective_sample_count")
        _require_warning_tuple(self.warnings, "warnings")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Single deterministic metric value for a factor evaluation."""

    name: str
    value: float
    direction: str
    sample_count: int
    calculation_version: str

    def __post_init__(self) -> None:
        """Validate metric metadata."""
        _require_string(self.name, "name")
        _require_finite_float(self.value, "value")
        _require_string(self.direction, "direction")
        _require_non_negative_int(self.sample_count, "sample_count")
        _require_string(self.calculation_version, "calculation_version")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Deterministic factor evaluation result."""

    factor_id: str
    data_snapshot_id: str
    raw_factor_values_ref: ArtifactRef
    evaluated_factor_values_ref: ArtifactRef
    factor_preprocessing_result_ref: ObjectRef
    forward_return_ref: ArtifactRef
    evaluation_protocol_id: str
    preprocessing_protocol_id: str
    neutralization_applied: bool
    metrics: tuple[MetricValue, ...]
    effective_date_count: int
    effective_sample_count: int
    coverage: float
    warnings: tuple[WarningInfo, ...]
    evaluator_version: str
    schema_version: str
    total_date_count: int | None = None

    def __post_init__(self) -> None:
        """Validate evaluation result metadata."""
        _require_string(self.factor_id, "factor_id")
        _require_string(self.data_snapshot_id, "data_snapshot_id")
        if not isinstance(self.raw_factor_values_ref, ArtifactRef):
            raise CoreValidationError("raw_factor_values_ref must be ArtifactRef")
        if not isinstance(self.evaluated_factor_values_ref, ArtifactRef):
            raise CoreValidationError("evaluated_factor_values_ref must be ArtifactRef")
        if not isinstance(self.factor_preprocessing_result_ref, ObjectRef):
            raise CoreValidationError(
                "factor_preprocessing_result_ref must be ObjectRef"
            )
        if not isinstance(self.forward_return_ref, ArtifactRef):
            raise CoreValidationError("forward_return_ref must be ArtifactRef")
        _require_string(self.evaluation_protocol_id, "evaluation_protocol_id")
        _require_string(self.preprocessing_protocol_id, "preprocessing_protocol_id")
        if (
            _require_bool(self.neutralization_applied, "neutralization_applied")
            is not True
        ):
            raise CoreValidationError("neutralization_applied must be true")
        _require_tuple_of(self.metrics, MetricValue, "metrics")
        _require_non_negative_int(self.effective_date_count, "effective_date_count")
        _require_non_negative_int(self.effective_sample_count, "effective_sample_count")
        if self.total_date_count is not None:
            _require_non_negative_int(self.total_date_count, "total_date_count")
            if self.effective_date_count > self.total_date_count:
                raise CoreValidationError(
                    "effective_date_count must not exceed total_date_count"
                )
        _require_coverage(self.coverage, "coverage")
        _require_warning_tuple(self.warnings, "warnings")
        _require_string(self.evaluator_version, "evaluator_version")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    """Deterministic decision produced by an acceptance policy."""

    factor_id: str
    evaluation_result_ref: ObjectRef
    accepted: bool
    reason_codes: tuple[str, ...]
    policy_id: str
    policy_version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate acceptance decision metadata."""
        _require_string(self.factor_id, "factor_id")
        if not isinstance(self.evaluation_result_ref, ObjectRef):
            raise CoreValidationError("evaluation_result_ref must be ObjectRef")
        _require_bool(self.accepted, "accepted")
        _require_code_tuple(self.reason_codes, "reason_codes")
        _require_string(self.policy_id, "policy_id")
        _require_string(self.policy_version, "policy_version")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class AcceptanceMetricThreshold:
    """One actual metric bound applied by an AcceptancePolicy."""

    metric_name: str
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        """Validate threshold bounds."""
        _require_string(self.metric_name, "metric_name")
        for field_name in ("minimum", "maximum"):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_float(value, field_name)
        if self.minimum is None and self.maximum is None:
            raise CoreValidationError("metric threshold requires a bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise CoreValidationError("minimum must not exceed maximum")


@dataclass(frozen=True, slots=True)
class AcceptancePolicySnapshot:
    """Replayable configuration actually used for acceptance decisions."""

    policy_id: str
    policy_version: str
    acceptance_metric: str
    metric_thresholds: tuple[AcceptanceMetricThreshold, ...]
    min_effective_sample_count: int
    min_coverage: float
    schema_version: str
    min_effective_date_ratio: float = 0.0
    use_absolute_value: bool = False

    def __post_init__(self) -> None:
        """Validate the policy evidence snapshot."""
        _require_string(self.policy_id, "policy_id")
        _require_string(self.policy_version, "policy_version")
        _require_string(self.acceptance_metric, "acceptance_metric")
        _require_tuple_of(
            self.metric_thresholds,
            AcceptanceMetricThreshold,
            "metric_thresholds",
        )
        if not self.metric_thresholds:
            raise CoreValidationError("metric_thresholds must not be empty")
        names = tuple(threshold.metric_name for threshold in self.metric_thresholds)
        if len(set(names)) != len(names):
            raise CoreValidationError("metric thresholds must be unique")
        if (
            type(self.min_effective_sample_count) is not int
            or self.min_effective_sample_count < 1
        ):
            raise CoreValidationError("min_effective_sample_count must be positive")
        _require_coverage(self.min_coverage, "min_coverage")
        _require_coverage(
            self.min_effective_date_ratio,
            "min_effective_date_ratio",
        )
        _require_bool(self.use_absolute_value, "use_absolute_value")
        if self.schema_version not in {
            "acceptance-policy-snapshot/1.0",
            "acceptance-policy-snapshot/1.1",
        }:
            raise CoreValidationError(
                "invalid acceptance policy snapshot schema_version"
            )


@dataclass(frozen=True, slots=True)
class FeedbackRequest:
    """Replayable evidence contract passed to the v2 feedback agent."""

    hypothesis_ref: ObjectRef
    factor_design_refs: tuple[ObjectRef, ...]
    factor_proposal_refs: tuple[ObjectRef, ...]
    factor_refs: tuple[ObjectRef, ...]
    validation_report_refs: tuple[ObjectRef, ...]
    calculation_result_refs: tuple[ObjectRef, ...]
    factor_preprocessing_result_refs: tuple[ObjectRef, ...]
    evaluation_result_refs: tuple[ObjectRef, ...]
    acceptance_decision_refs: tuple[ObjectRef, ...]
    comparison_baseline_refs: tuple[ObjectRef, ...]
    failures: tuple[FailureInfo, ...]
    acceptance_policy: AcceptancePolicySnapshot
    schema_version: str

    def __post_init__(self) -> None:
        """Validate feedback evidence references and deterministic context."""
        if not isinstance(self.hypothesis_ref, ObjectRef):
            raise CoreValidationError("hypothesis_ref must be ObjectRef")
        for field_name in (
            "factor_design_refs",
            "factor_proposal_refs",
            "factor_refs",
            "validation_report_refs",
            "calculation_result_refs",
            "factor_preprocessing_result_refs",
            "evaluation_result_refs",
            "acceptance_decision_refs",
            "comparison_baseline_refs",
        ):
            _require_object_ref_tuple(getattr(self, field_name), field_name)
        _require_tuple_of(self.failures, FailureInfo, "failures")
        if not isinstance(self.acceptance_policy, AcceptancePolicySnapshot):
            raise CoreValidationError(
                "acceptance_policy must be AcceptancePolicySnapshot"
            )
        if self.schema_version != "feedback-request/2.0":
            raise CoreValidationError("invalid feedback request schema_version")


@dataclass(frozen=True, slots=True)
class ResearchFeedback:
    """Persisted research feedback for the next cycle."""

    feedback_id: str
    cycle_id: str
    hypothesis_verdict: HypothesisVerdict
    summary: str
    supported_findings: tuple[str, ...]
    rejected_findings: tuple[str, ...]
    failure_analysis: tuple[str, ...]
    next_actions: tuple[str, ...]
    next_hypothesis_hint: str | None
    evidence_refs: tuple[ObjectRef, ...]
    schema_version: str

    def __post_init__(self) -> None:
        """Validate persisted feedback fields."""
        _require_string(self.feedback_id, "feedback_id")
        _require_string(self.cycle_id, "cycle_id")
        if not isinstance(self.hypothesis_verdict, HypothesisVerdict):
            raise CoreValidationError("hypothesis_verdict must be HypothesisVerdict")
        _require_string(self.summary, "summary")
        _require_string_tuple(self.supported_findings, "supported_findings")
        _require_string_tuple(self.rejected_findings, "rejected_findings")
        _require_string_tuple(self.failure_analysis, "failure_analysis")
        _require_string_tuple(self.next_actions, "next_actions")
        _require_optional_string(self.next_hypothesis_hint, "next_hypothesis_hint")
        _require_object_ref_tuple(self.evidence_refs, "evidence_refs")
        if self.schema_version != "research-feedback/2.0":
            raise CoreValidationError("invalid research feedback schema_version")


@dataclass(frozen=True, slots=True)
class ExperimentLessonDesign:
    """Structured design semantics retained from one experiment."""

    design_id: str
    name: str
    mechanism_mapping: str
    signal_definition: str
    formulation: str
    expected_direction: str
    failure_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate lesson design semantics."""
        for field_name in (
            "design_id",
            "name",
            "mechanism_mapping",
            "signal_definition",
            "formulation",
            "expected_direction",
        ):
            _require_string(getattr(self, field_name), field_name)
        _require_string_tuple(self.failure_conditions, "failure_conditions")


@dataclass(frozen=True, slots=True)
class ExperimentLessonFactorOutcome:
    """Deterministic stage outcome for one factor design."""

    design_id: str
    design_name: str
    proposal_id: str | None
    factor_id: str | None
    canonical_expression: str | None
    validation_passed: bool | None
    calculation_completed: bool
    preprocessing_completed: bool
    evaluation_completed: bool
    accepted: bool | None
    acceptance_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate monotonic stage completion and acceptance facts."""
        _require_string(self.design_id, "design_id")
        _require_string(self.design_name, "design_name")
        _require_optional_string(self.proposal_id, "proposal_id")
        _require_optional_string(self.factor_id, "factor_id")
        _require_optional_string(self.canonical_expression, "canonical_expression")
        if self.validation_passed is not None:
            _require_bool(self.validation_passed, "validation_passed")
        for field_name in (
            "calculation_completed",
            "preprocessing_completed",
            "evaluation_completed",
        ):
            _require_bool(getattr(self, field_name), field_name)
        if self.accepted is not None:
            _require_bool(self.accepted, "accepted")
        _require_code_tuple(
            self.acceptance_reason_codes,
            "acceptance_reason_codes",
        )
        if self.validation_passed is True and (
            self.factor_id is None or self.canonical_expression is None
        ):
            raise CoreValidationError(
                "validated lesson outcome requires factor and canonical expression"
            )
        if self.validation_passed is not True and self.factor_id is not None:
            raise CoreValidationError(
                "unvalidated lesson outcome must not contain a factor_id"
            )
        if self.preprocessing_completed and not self.calculation_completed:
            raise CoreValidationError(
                "preprocessing completion requires calculation completion"
            )
        if self.evaluation_completed and not self.preprocessing_completed:
            raise CoreValidationError(
                "evaluation completion requires preprocessing completion"
            )
        if self.accepted is not None and not self.evaluation_completed:
            raise CoreValidationError("acceptance requires evaluation completion")
        if self.accepted is None and self.acceptance_reason_codes:
            raise CoreValidationError(
                "acceptance reason codes require an acceptance decision"
            )


@dataclass(frozen=True, slots=True)
class ExperimentLessonMetric:
    """One deterministic scalar metric retained for future hypotheses."""

    factor_id: str
    name: str
    value: float
    direction: str
    sample_count: int
    calculation_version: str

    def __post_init__(self) -> None:
        """Validate lesson metric provenance and value."""
        _require_string(self.factor_id, "factor_id")
        _require_string(self.name, "name")
        _require_finite_float(self.value, "value")
        _require_string(self.direction, "direction")
        _require_non_negative_int(self.sample_count, "sample_count")
        _require_string(self.calculation_version, "calculation_version")


@dataclass(frozen=True, slots=True)
class ExperimentLesson:
    """Immutable structured lesson deterministically derived from experiment facts."""

    lesson_id: str
    experiment_id: str
    run_id: str
    cycle_id: str
    research_direction_id: str | None
    hypothesis_statement: str
    hypothesis_mechanism: str
    design_summaries: tuple[ExperimentLessonDesign, ...]
    factor_outcomes: tuple[ExperimentLessonFactorOutcome, ...]
    key_metrics: tuple[ExperimentLessonMetric, ...]
    stable_failure_codes: tuple[str, ...]
    hypothesis_verdict: HypothesisVerdict
    effective_lessons: tuple[str, ...]
    next_actions: tuple[str, ...]
    hypothesis_ref: ObjectRef
    feedback_ref: ObjectRef
    source_refs: tuple[ObjectRef, ...]
    schema_version: str

    def __post_init__(self) -> None:
        """Validate lesson identity, structured facts, and complete provenance."""
        _require_string(self.lesson_id, "lesson_id")
        _require_string(self.experiment_id, "experiment_id")
        if self.lesson_id != self.experiment_id:
            raise CoreValidationError("lesson_id must equal its experiment_id")
        _require_string(self.run_id, "run_id")
        _require_string(self.cycle_id, "cycle_id")
        _require_optional_string(self.research_direction_id, "research_direction_id")
        _require_string(self.hypothesis_statement, "hypothesis_statement")
        _require_string(self.hypothesis_mechanism, "hypothesis_mechanism")
        _require_tuple_of(
            self.design_summaries,
            ExperimentLessonDesign,
            "design_summaries",
        )
        _require_tuple_of(
            self.factor_outcomes,
            ExperimentLessonFactorOutcome,
            "factor_outcomes",
        )
        _require_tuple_of(
            self.key_metrics,
            ExperimentLessonMetric,
            "key_metrics",
        )
        _require_code_tuple(self.stable_failure_codes, "stable_failure_codes")
        if not isinstance(self.hypothesis_verdict, HypothesisVerdict):
            raise CoreValidationError("hypothesis_verdict must be HypothesisVerdict")
        _require_string_tuple(self.effective_lessons, "effective_lessons")
        _require_string_tuple(self.next_actions, "next_actions")
        if not isinstance(self.hypothesis_ref, ObjectRef):
            raise CoreValidationError("hypothesis_ref must be ObjectRef")
        if not isinstance(self.feedback_ref, ObjectRef):
            raise CoreValidationError("feedback_ref must be ObjectRef")
        _require_object_ref_tuple(self.source_refs, "source_refs")
        if self.hypothesis_ref not in self.source_refs:
            raise CoreValidationError("source_refs must contain hypothesis_ref")
        if self.feedback_ref not in self.source_refs:
            raise CoreValidationError("source_refs must contain feedback_ref")
        if self.schema_version != "experiment-lesson/1.0":
            raise CoreValidationError("invalid experiment lesson schema_version")


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Immutable index of completed workflow facts for one experiment."""

    experiment_id: str
    run_id: str
    cycle_id: str
    status: ExperimentStatus
    parent_ids: tuple[str, ...]
    hypothesis_ref: ObjectRef
    factor_proposal_refs: tuple[ObjectRef, ...]
    factor_refs: tuple[ObjectRef, ...]
    validation_report_refs: tuple[ObjectRef, ...]
    calculation_result_refs: tuple[ObjectRef, ...]
    factor_preprocessing_result_refs: tuple[ObjectRef, ...]
    evaluation_result_refs: tuple[ObjectRef, ...]
    acceptance_decision_refs: tuple[ObjectRef, ...]
    feedback_ref: ObjectRef | None
    config_snapshot_ref: ArtifactRef
    code_version: str
    data_version: str
    created_at: datetime
    completed_at: datetime | None
    schema_version: str

    def __post_init__(self) -> None:
        """Validate experiment record references and lifecycle timestamps."""
        _require_string(self.experiment_id, "experiment_id")
        _require_string(self.run_id, "run_id")
        _require_string(self.cycle_id, "cycle_id")
        if not isinstance(self.status, ExperimentStatus):
            raise CoreValidationError("status must be ExperimentStatus")
        _require_string_tuple(self.parent_ids, "parent_ids")
        if not isinstance(self.hypothesis_ref, ObjectRef):
            raise CoreValidationError("hypothesis_ref must be ObjectRef")
        _require_object_ref_tuple(self.factor_proposal_refs, "factor_proposal_refs")
        _require_object_ref_tuple(self.factor_refs, "factor_refs")
        _require_object_ref_tuple(self.validation_report_refs, "validation_report_refs")
        _require_object_ref_tuple(
            self.calculation_result_refs,
            "calculation_result_refs",
        )
        _require_object_ref_tuple(
            self.factor_preprocessing_result_refs,
            "factor_preprocessing_result_refs",
        )
        _require_object_ref_tuple(self.evaluation_result_refs, "evaluation_result_refs")
        _require_object_ref_tuple(
            self.acceptance_decision_refs,
            "acceptance_decision_refs",
        )
        if self.feedback_ref is not None and not isinstance(
            self.feedback_ref, ObjectRef
        ):
            raise CoreValidationError("feedback_ref must be ObjectRef or None")
        if not isinstance(self.config_snapshot_ref, ArtifactRef):
            raise CoreValidationError("config_snapshot_ref must be ArtifactRef")
        _require_string(self.code_version, "code_version")
        _require_string(self.data_version, "data_version")
        created_at = _require_utc_datetime(self.created_at, "created_at")
        completed_at = _require_optional_utc_datetime(
            self.completed_at,
            "completed_at",
        )
        if completed_at is not None and completed_at < created_at:
            raise CoreValidationError("completed_at must not be before created_at")
        _require_schema_version(self.schema_version)
