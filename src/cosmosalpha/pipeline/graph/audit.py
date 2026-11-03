"""Graph-specific candidate outcome audit objects and reports."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from cosmosalpha.agents import (
    AcceptanceDecision,
    ExperimentStatus,
    FactorProposal,
    FactorSpec,
)
from cosmosalpha.core import (
    CoreValidationError,
    FailureInfo,
    MissingFieldError,
    ObjectRef,
    stable_sha256,
    to_json,
)
from cosmosalpha.storage import WorkflowStepAttempt, WorkflowStepStatus

GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE = "graph_candidate_outcome"
GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION = "graph-candidate-outcome/1.0"
GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION = "graph-candidate-outcome/2.0"
GRAPH_CANDIDATE_OUTCOME_REPORT_SCHEMA_VERSION = "graph-candidate-outcome-report/2.0"


class GraphCandidateOutcomeStatus(StrEnum):
    """Terminal outcome categories for one Graph candidate audit row."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    CALCULATION_FAILURE = "calculation_failure"
    PREPROCESSING_FAILURE = "preprocessing_failure"
    EVALUATION_FAILURE = "evaluation_failure"
    OTHER_FAILURE = "other_failure"


@dataclass(frozen=True, slots=True)
class GraphCandidateOutcome:
    """Persisted, candidate-level Graph outcome independent of ``GraphResult``."""

    run_id: str
    round_id: int
    attempt_id: str
    parent_id: str
    candidate_index: int
    outcome: GraphCandidateOutcomeStatus
    candidate_status: ExperimentStatus | None
    proposal_ref: ObjectRef | None
    validation_report_ref: ObjectRef | None
    factor_ref: ObjectRef | None
    calculation_result_ref: ObjectRef | None
    preprocessing_result_ref: ObjectRef | None
    evaluation_result_ref: ObjectRef | None
    acceptance_decision_ref: ObjectRef | None
    attempt_index: int = 0
    pre_calculation_gate_reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    failure: FailureInfo | None = None
    expression: str | None = None
    schema_version: str = GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate stable identity, references, and terminal outcome data."""
        for field_name in ("run_id", "attempt_id", "parent_id"):
            _require_text(getattr(self, field_name), field_name)
        if type(self.round_id) is not int or self.round_id < 0:
            raise CoreValidationError("round_id must be non-negative")
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise CoreValidationError("candidate_index must be non-negative")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise CoreValidationError("attempt_index must be non-negative")
        if not isinstance(self.outcome, GraphCandidateOutcomeStatus):
            raise CoreValidationError("outcome must be GraphCandidateOutcomeStatus")
        if self.candidate_status is not None and not isinstance(
            self.candidate_status, ExperimentStatus
        ):
            raise CoreValidationError(
                "candidate_status must be ExperimentStatus or None"
            )
        for field_name in (
            "proposal_ref",
            "validation_report_ref",
            "factor_ref",
            "calculation_result_ref",
            "preprocessing_result_ref",
            "evaluation_result_ref",
            "acceptance_decision_ref",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ObjectRef):
                raise CoreValidationError(f"{field_name} must be ObjectRef or None")
        if self.pre_calculation_gate_reason is not None:
            _require_text(
                self.pre_calculation_gate_reason,
                "pre_calculation_gate_reason",
            )
        if not isinstance(self.reason_codes, tuple):
            raise CoreValidationError("reason_codes must be a tuple")
        for index, code in enumerate(self.reason_codes):
            _require_text(code, f"reason_codes[{index}]")
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        if self.expression is not None:
            _require_text(self.expression, "expression")
        if self.schema_version not in {
            GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION,
            GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
        }:
            raise CoreValidationError("invalid GraphCandidateOutcome schema_version")
        if (
            self.schema_version == GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION
            and self.attempt_index != 0
        ):
            raise CoreValidationError(
                "legacy candidate outcomes require attempt_index 0"
            )


@dataclass(frozen=True, slots=True)
class _LegacyGraphCandidateOutcome:
    """Exact v1 DTO used only to upgrade persisted audit objects."""

    run_id: str
    round_id: int
    attempt_id: str
    parent_id: str
    candidate_index: int
    outcome: GraphCandidateOutcomeStatus
    candidate_status: ExperimentStatus | None
    proposal_ref: ObjectRef | None
    validation_report_ref: ObjectRef | None
    factor_ref: ObjectRef | None
    calculation_result_ref: ObjectRef | None
    preprocessing_result_ref: ObjectRef | None
    evaluation_result_ref: ObjectRef | None
    acceptance_decision_ref: ObjectRef | None
    pre_calculation_gate_reason: str | None
    reason_codes: tuple[str, ...]
    failure: FailureInfo | None
    expression: str | None
    schema_version: str


def graph_candidate_outcome_id(
    run_id: str,
    attempt_id: str,
    candidate_index: int,
    attempt_index: int | None = None,
) -> str:
    """Return one immutable candidate audit attempt identity."""
    _require_text(run_id, "run_id")
    _require_text(attempt_id, "attempt_id")
    if type(candidate_index) is not int or candidate_index < 0:
        raise CoreValidationError("candidate_index must be non-negative")
    if attempt_index is not None and (
        type(attempt_index) is not int or attempt_index < 0
    ):
        raise CoreValidationError("attempt_index must be non-negative or None")
    base_id = (
        "graph-candidate-"
        + stable_sha256(
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "candidate_index": candidate_index,
            }
        )[:24]
    )
    return base_id if attempt_index is None else f"{base_id}-{attempt_index}"


def graph_candidate_outcome_ref(outcome: GraphCandidateOutcome) -> ObjectRef:
    """Return the object reference for one candidate outcome."""
    if not isinstance(outcome, GraphCandidateOutcome):
        raise CoreValidationError("outcome must be GraphCandidateOutcome")
    return ObjectRef(
        GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE,
        graph_candidate_outcome_id(
            outcome.run_id,
            outcome.attempt_id,
            outcome.candidate_index,
            (
                None
                if outcome.schema_version
                == GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION
                else outcome.attempt_index
            ),
        ),
        outcome.schema_version,
    )


def persist_graph_candidate_outcome(
    objects: object,
    outcome: GraphCandidateOutcome,
) -> ObjectRef:
    """Persist one immutable audit revision without conflicting on Resume."""
    if not isinstance(outcome, GraphCandidateOutcome):
        raise CoreValidationError("outcome must be GraphCandidateOutcome")
    put_object = getattr(objects, "put_object", None)
    if not callable(put_object):
        raise CoreValidationError("objects must provide put_object")
    matching = _candidate_outcome_history(objects, outcome)
    for existing in matching:
        if _same_outcome(existing, outcome):
            return graph_candidate_outcome_ref(existing)
    stored = replace(
        outcome,
        attempt_index=(
            0 if not matching else max(item.attempt_index for item in matching) + 1
        ),
        schema_version=GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
    )
    ref = graph_candidate_outcome_ref(stored)
    return put_object(ref.object_type, ref.object_id, stored)


def reconcile_graph_candidate_outcomes(
    objects: object,
    workflow_attempts: Iterable[WorkflowStepAttempt],
    *,
    run_id: str,
    attempt_identities: Mapping[str, tuple[int, str]],
) -> tuple[GraphCandidateOutcome, ...]:
    """Recover completed candidate facts when a later workflow step failed."""
    _require_text(run_id, "run_id")
    attempts = tuple(
        attempt
        for attempt in workflow_attempts
        if attempt.run_id == run_id and attempt.cycle_id in attempt_identities
    )
    by_cycle: dict[str, list[WorkflowStepAttempt]] = {}
    for attempt in attempts:
        by_cycle.setdefault(attempt.cycle_id, []).append(attempt)
    for cycle_id, cycle_attempts in by_cycle.items():
        round_id, parent_id = attempt_identities[cycle_id]
        validation_groups: dict[str, list[WorkflowStepAttempt]] = {}
        for attempt in cycle_attempts:
            if (
                attempt.status is not WorkflowStepStatus.SUCCEEDED
                or not attempt.step_id.startswith("candidate:")
                or not attempt.step_id.endswith(":validation")
            ):
                continue
            proposal_ref = _ref_by_type(attempt.output_object_refs, "factor_proposal")
            if proposal_ref is None:
                continue
            proposal = _load_proposal(objects, proposal_ref)
            group_id = (
                proposal.factor_design_ref.object_id
                if isinstance(proposal, FactorProposal)
                else proposal.proposal_id
            )
            validation_groups.setdefault(group_id, []).append(attempt)
        for candidate_index, group in enumerate(validation_groups.values()):
            validation = group[-1]
            factor_ref = _ref_by_type(validation.output_object_refs, "factor_spec")
            proposal_ref = _ref_by_type(
                validation.output_object_refs, "factor_proposal"
            )
            validation_ref = _ref_by_type(
                validation.output_object_refs, "validation_report"
            )
            if factor_ref is None or proposal_ref is None or validation_ref is None:
                continue
            step_prefix = f"candidate:{factor_ref.object_id}:"
            decision_attempt = _successful_step(
                cycle_attempts,
                step_prefix + "acceptance",
            )
            if decision_attempt is None:
                continue
            decision_ref = _ref_by_type(
                decision_attempt.output_object_refs,
                "acceptance_decision",
            )
            if decision_ref is None:
                continue
            decision = _get_object(objects, decision_ref, AcceptanceDecision)
            factor = _get_object(objects, factor_ref, FactorSpec)
            calculation = _successful_step(
                cycle_attempts,
                step_prefix + "calculation",
            )
            preprocessing = _successful_step(
                cycle_attempts,
                step_prefix + "preprocessing",
            )
            persist_graph_candidate_outcome(
                objects,
                GraphCandidateOutcome(
                    run_id=run_id,
                    round_id=round_id,
                    attempt_id=cycle_id,
                    parent_id=parent_id,
                    candidate_index=candidate_index,
                    outcome=(
                        GraphCandidateOutcomeStatus.ACCEPTED
                        if decision.accepted
                        else GraphCandidateOutcomeStatus.REJECTED
                    ),
                    candidate_status=(
                        ExperimentStatus.ACCEPTED
                        if decision.accepted
                        else ExperimentStatus.REJECTED
                    ),
                    proposal_ref=proposal_ref,
                    validation_report_ref=validation_ref,
                    factor_ref=factor_ref,
                    calculation_result_ref=_only_output_ref(calculation),
                    preprocessing_result_ref=_only_output_ref(preprocessing),
                    evaluation_result_ref=decision.evaluation_result_ref,
                    acceptance_decision_ref=decision_ref,
                    reason_codes=decision.reason_codes,
                    expression=factor.expression,
                ),
            )
    return load_graph_candidate_outcomes(objects, run_id=run_id)


_REPORT_COLUMNS = (
    "run_id",
    "round_id",
    "attempt_id",
    "parent_id",
    "candidate_index",
    "attempt_index",
    "outcome",
    "candidate_status",
    "proposal_id",
    "factor_id",
    "expression",
    "pre_calculation_gate_reason",
    "reason_codes",
    "failure_stage",
    "failure_error_code",
    "failure_message",
    "failure_retryable",
    "proposal_ref",
    "validation_report_ref",
    "factor_ref",
    "calculation_result_ref",
    "preprocessing_result_ref",
    "evaluation_result_ref",
    "acceptance_decision_ref",
)


def graph_candidate_outcome_rows(
    outcomes: Iterable[GraphCandidateOutcome],
) -> tuple[dict[str, object], ...]:
    """Convert outcomes into stable CSV rows without changing their objects."""
    ordered = tuple(
        sorted(
            outcomes,
            key=lambda item: (
                item.round_id,
                item.attempt_id,
                item.candidate_index,
                item.attempt_index,
            ),
        )
    )
    return tuple(
        {
            "run_id": outcome.run_id,
            "round_id": outcome.round_id,
            "attempt_id": outcome.attempt_id,
            "parent_id": outcome.parent_id,
            "candidate_index": outcome.candidate_index,
            "attempt_index": outcome.attempt_index,
            "outcome": outcome.outcome.value,
            "candidate_status": (
                ""
                if outcome.candidate_status is None
                else outcome.candidate_status.value
            ),
            "proposal_id": _ref_id(outcome.proposal_ref),
            "factor_id": _ref_id(outcome.factor_ref),
            "expression": outcome.expression or "",
            "pre_calculation_gate_reason": outcome.pre_calculation_gate_reason or "",
            "reason_codes": "|".join(outcome.reason_codes),
            "failure_stage": "" if outcome.failure is None else outcome.failure.stage,
            "failure_error_code": (
                "" if outcome.failure is None else outcome.failure.error_code
            ),
            "failure_message": (
                "" if outcome.failure is None else outcome.failure.message
            ),
            "failure_retryable": (
                "" if outcome.failure is None else outcome.failure.retryable
            ),
            "proposal_ref": _ref_json(outcome.proposal_ref),
            "validation_report_ref": _ref_json(outcome.validation_report_ref),
            "factor_ref": _ref_json(outcome.factor_ref),
            "calculation_result_ref": _ref_json(outcome.calculation_result_ref),
            "preprocessing_result_ref": _ref_json(outcome.preprocessing_result_ref),
            "evaluation_result_ref": _ref_json(outcome.evaluation_result_ref),
            "acceptance_decision_ref": _ref_json(outcome.acceptance_decision_ref),
        }
        for outcome in ordered
    )


def load_graph_candidate_outcomes(
    objects: object,
    *,
    run_id: str,
) -> tuple[GraphCandidateOutcome, ...]:
    """Load candidate outcomes for one run in deterministic object order."""
    _require_text(run_id, "run_id")
    list_object_ids = getattr(objects, "list_object_ids", None)
    get_object = getattr(objects, "get_object", None)
    if not callable(list_object_ids) or not callable(get_object):
        raise CoreValidationError("objects must provide Graph audit repository methods")
    outcomes: list[GraphCandidateOutcome] = []
    for object_id in list_object_ids(GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE):
        outcome = _load_candidate_outcome(
            objects,
            ObjectRef(
                GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE,
                object_id,
                GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
            ),
        )
        if outcome.run_id == run_id:
            outcomes.append(outcome)
    return tuple(
        sorted(
            outcomes,
            key=lambda item: (
                item.round_id,
                item.attempt_id,
                item.candidate_index,
                item.attempt_index,
            ),
        )
    )


def write_graph_candidate_outcome_report(
    run_root: Path,
    outcomes: Iterable[GraphCandidateOutcome],
    *,
    run_id: str,
) -> tuple[Path, Path]:
    """Write the Graph candidate CSV and its independent manifest."""
    _require_text(run_id, "run_id")
    selected = tuple(outcome for outcome in outcomes if outcome.run_id == run_id)
    rows = graph_candidate_outcome_rows(selected)
    report_root = run_root / "graph"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "candidate_outcomes.csv"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_REPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(output.getvalue(), encoding="utf-8")

    refs = tuple(
        graph_candidate_outcome_ref(outcome)
        for outcome in sorted(
            selected,
            key=lambda item: (
                item.round_id,
                item.attempt_id,
                item.candidate_index,
                item.attempt_index,
            ),
        )
    )
    manifest_path = report_root / "candidate_outcomes.json"
    manifest_path.write_text(
        to_json(
            {
                "run_id": run_id,
                "csv_path": "candidate_outcomes.csv",
                "outcome_refs": refs,
                "row_count": len(rows),
                "schema_version": GRAPH_CANDIDATE_OUTCOME_REPORT_SCHEMA_VERSION,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _same_outcome(
    left: GraphCandidateOutcome,
    right: GraphCandidateOutcome,
) -> bool:
    """Compare business facts while ignoring audit revision metadata."""
    return replace(
        left,
        attempt_index=0,
        schema_version=GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
    ) == replace(
        right,
        attempt_index=0,
        schema_version=GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
    )


def _candidate_outcome_history(
    objects: object,
    outcome: GraphCandidateOutcome,
) -> tuple[GraphCandidateOutcome, ...]:
    """Load only revisions belonging to one deterministic candidate identity."""
    exists = getattr(objects, "exists", None)
    get_object = getattr(objects, "get_object", None)
    if not callable(exists) or not callable(get_object):
        raise CoreValidationError("objects must provide exists and get_object")
    refs = [
        ObjectRef(
            GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE,
            graph_candidate_outcome_id(
                outcome.run_id,
                outcome.attempt_id,
                outcome.candidate_index,
            ),
            GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION,
        )
    ]
    attempt_index = 0
    while True:
        ref = ObjectRef(
            GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE,
            graph_candidate_outcome_id(
                outcome.run_id,
                outcome.attempt_id,
                outcome.candidate_index,
                attempt_index,
            ),
            GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION,
        )
        if not exists(ref):
            break
        refs.append(ref)
        attempt_index += 1
    return tuple(_load_candidate_outcome(objects, ref) for ref in refs if exists(ref))


def _load_candidate_outcome(
    objects: object,
    ref: ObjectRef,
) -> GraphCandidateOutcome:
    """Load v2 directly or explicitly upgrade one strict v1 payload."""
    get_object = getattr(objects, "get_object", None)
    if not callable(get_object):
        raise CoreValidationError("objects must provide get_object")
    try:
        return get_object(ref, GraphCandidateOutcome)
    except MissingFieldError:
        legacy = get_object(ref, _LegacyGraphCandidateOutcome)
    if legacy.schema_version != GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION:
        raise CoreValidationError("invalid legacy candidate outcome schema_version")
    return GraphCandidateOutcome(
        run_id=legacy.run_id,
        round_id=legacy.round_id,
        attempt_id=legacy.attempt_id,
        parent_id=legacy.parent_id,
        candidate_index=legacy.candidate_index,
        outcome=legacy.outcome,
        candidate_status=legacy.candidate_status,
        proposal_ref=legacy.proposal_ref,
        validation_report_ref=legacy.validation_report_ref,
        factor_ref=legacy.factor_ref,
        calculation_result_ref=legacy.calculation_result_ref,
        preprocessing_result_ref=legacy.preprocessing_result_ref,
        evaluation_result_ref=legacy.evaluation_result_ref,
        acceptance_decision_ref=legacy.acceptance_decision_ref,
        attempt_index=0,
        pre_calculation_gate_reason=legacy.pre_calculation_gate_reason,
        reason_codes=legacy.reason_codes,
        failure=legacy.failure,
        expression=legacy.expression,
        schema_version=legacy.schema_version,
    )


def _get_object[ObjectT](
    objects: object,
    ref: ObjectRef,
    target_type: type[ObjectT],
) -> ObjectT:
    get_object = getattr(objects, "get_object", None)
    if not callable(get_object):
        raise CoreValidationError("objects must provide get_object")
    return get_object(ref, target_type)


def _load_proposal(
    objects: object,
    ref: ObjectRef,
) -> FactorProposal:
    return _get_object(objects, ref, FactorProposal)


def _successful_step(
    attempts: Iterable[WorkflowStepAttempt],
    step_id: str,
) -> WorkflowStepAttempt | None:
    matches = tuple(
        attempt
        for attempt in attempts
        if attempt.step_id == step_id and attempt.status is WorkflowStepStatus.SUCCEEDED
    )
    return None if not matches else max(matches, key=lambda item: item.attempt)


def _ref_by_type(
    refs: Iterable[ObjectRef],
    object_type: str,
) -> ObjectRef | None:
    matches = tuple(ref for ref in refs if ref.object_type == object_type)
    return matches[0] if len(matches) == 1 else None


def _only_output_ref(attempt: WorkflowStepAttempt | None) -> ObjectRef | None:
    if attempt is None or len(attempt.output_object_refs) != 1:
        return None
    return attempt.output_object_refs[0]


def _ref_id(ref: ObjectRef | None) -> str:
    return "" if ref is None else ref.object_id


def _ref_json(ref: ObjectRef | None) -> str:
    return "" if ref is None else to_json(ref)


__all__ = [
    "GRAPH_CANDIDATE_OUTCOME_OBJECT_TYPE",
    "GRAPH_CANDIDATE_OUTCOME_LEGACY_SCHEMA_VERSION",
    "GRAPH_CANDIDATE_OUTCOME_REPORT_SCHEMA_VERSION",
    "GRAPH_CANDIDATE_OUTCOME_SCHEMA_VERSION",
    "GraphCandidateOutcome",
    "GraphCandidateOutcomeStatus",
    "graph_candidate_outcome_id",
    "graph_candidate_outcome_ref",
    "graph_candidate_outcome_rows",
    "load_graph_candidate_outcomes",
    "persist_graph_candidate_outcome",
    "reconcile_graph_candidate_outcomes",
    "write_graph_candidate_outcome_report",
]
