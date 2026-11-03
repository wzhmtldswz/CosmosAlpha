"""Graph-owned accepted-factor publication reconciliation and audit reports."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    ExperimentRecord,
    FactorSpec,
)
from cosmosalpha.core import (
    CoreValidationError,
    FailureInfo,
    ObjectRef,
    stable_sha256,
    to_json,
)
from cosmosalpha.pipeline.graph.audit import (
    GraphCandidateOutcome,
    GraphCandidateOutcomeStatus,
)
from cosmosalpha.storage import (
    FactorLibraryEntry,
    LocalExperimentRepository,
    SqliteFactorLibrary,
    WorkflowStepAttempt,
    WorkflowStepStatus,
)

GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE = "graph_publication_outcome"
GRAPH_PUBLICATION_OUTCOME_SCHEMA_VERSION = "graph-publication-outcome/1.0"
GRAPH_PUBLICATION_REPORT_SCHEMA_VERSION = "graph-publication-report/1.0"
FACTOR_LIBRARY_ENTRY_OBJECT_TYPE = "factor_library_entry"


class GraphPublicationOutcomeStatus(StrEnum):
    """Terminal state of one accepted decision's publication attempt."""

    PUBLISHED = "published"
    PUBLICATION_FAILED = "publication_failed"


@dataclass(frozen=True, slots=True)
class GraphPublicationOutcome:
    """Immutable audit record for one accepted-decision publication attempt."""

    run_id: str
    cycle_id: str
    experiment_id: str
    acceptance_decision_ref: ObjectRef
    factor_id: str
    factor_ref: ObjectRef | None
    evaluation_result_ref: ObjectRef | None
    attempt_index: int
    status: GraphPublicationOutcomeStatus
    entry_ref: ObjectRef | None = None
    reused: bool = False
    failure: FailureInfo | None = None
    schema_version: str = GRAPH_PUBLICATION_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate publication identity, references, and terminal state."""
        for field_name in ("run_id", "cycle_id", "experiment_id", "factor_id"):
            _require_text(getattr(self, field_name), field_name)
        if not isinstance(self.acceptance_decision_ref, ObjectRef):
            raise CoreValidationError("acceptance_decision_ref must be ObjectRef")
        for field_name in ("factor_ref", "evaluation_result_ref", "entry_ref"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ObjectRef):
                raise CoreValidationError(f"{field_name} must be ObjectRef or None")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise CoreValidationError("attempt_index must be non-negative")
        if not isinstance(self.status, GraphPublicationOutcomeStatus):
            raise CoreValidationError("status must be GraphPublicationOutcomeStatus")
        if type(self.reused) is not bool:
            raise CoreValidationError("reused must be a boolean")
        if self.failure is not None and not isinstance(self.failure, FailureInfo):
            raise CoreValidationError("failure must be FailureInfo or None")
        if self.status is GraphPublicationOutcomeStatus.PUBLISHED:
            if self.entry_ref is None or self.failure is not None:
                raise CoreValidationError(
                    "published outcome requires entry_ref and no failure"
                )
        elif self.status is GraphPublicationOutcomeStatus.PUBLICATION_FAILED:
            if self.failure is None:
                raise CoreValidationError("failed outcome requires failure")
        if self.schema_version != GRAPH_PUBLICATION_OUTCOME_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphPublicationOutcome schema_version")


def graph_publication_outcome_id(
    run_id: str,
    acceptance_decision_ref: ObjectRef,
    attempt_index: int,
) -> str:
    """Return an immutable ID for one decision publication attempt."""
    _require_text(run_id, "run_id")
    if not isinstance(acceptance_decision_ref, ObjectRef):
        raise CoreValidationError("acceptance_decision_ref must be ObjectRef")
    if type(attempt_index) is not int or attempt_index < 0:
        raise CoreValidationError("attempt_index must be non-negative")
    return (
        "graph-publication-"
        + stable_sha256(
            {
                "run_id": run_id,
                "acceptance_decision_ref": acceptance_decision_ref,
                "attempt_index": attempt_index,
            }
        )[:24]
    )


def graph_publication_outcome_ref(
    outcome: GraphPublicationOutcome,
) -> ObjectRef:
    """Return the object reference for one publication outcome."""
    if not isinstance(outcome, GraphPublicationOutcome):
        raise CoreValidationError("outcome must be GraphPublicationOutcome")
    return ObjectRef(
        GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE,
        graph_publication_outcome_id(
            outcome.run_id,
            outcome.acceptance_decision_ref,
            outcome.attempt_index,
        ),
        outcome.schema_version,
    )


def load_graph_publication_outcomes(
    objects: LocalExperimentRepository,
    *,
    run_id: str,
) -> tuple[GraphPublicationOutcome, ...]:
    """Load all publication attempts for one run in deterministic order."""
    _require_text(run_id, "run_id")
    if not isinstance(objects, LocalExperimentRepository):
        raise CoreValidationError("objects must be LocalExperimentRepository")
    outcomes = [
        objects.get_object(
            ObjectRef(
                GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE,
                object_id,
                GRAPH_PUBLICATION_OUTCOME_SCHEMA_VERSION,
            ),
            GraphPublicationOutcome,
        )
        for object_id in objects.list_object_ids(GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE)
    ]
    return tuple(
        sorted(
            (outcome for outcome in outcomes if outcome.run_id == run_id),
            key=_outcome_sort_key,
        )
    )


def reconcile_graph_publications(
    objects: LocalExperimentRepository,
    factor_library: SqliteFactorLibrary,
    *,
    run_id: str,
    clock: Callable[[], datetime],
    candidate_outcomes: Iterable[GraphCandidateOutcome] = (),
    workflow_attempts: Iterable[WorkflowStepAttempt] = (),
) -> tuple[GraphPublicationOutcome, ...]:
    """Reconcile accepted decisions even when later workflow steps failed.

    Existing Loop publication remains in place for round-to-round visibility.  This
    pass repairs a missing library entry or DTO mirror, records structured failure,
    and is safe to repeat after a process crash.
    """
    _require_text(run_id, "run_id")
    if not isinstance(objects, LocalExperimentRepository):
        raise CoreValidationError("objects must be LocalExperimentRepository")
    if not isinstance(factor_library, SqliteFactorLibrary):
        raise CoreValidationError("factor_library must be SqliteFactorLibrary")
    previous = load_graph_publication_outcomes(objects, run_id=run_id)
    latest_by_decision: dict[ObjectRef, GraphPublicationOutcome] = {}
    for outcome in previous:
        current = latest_by_decision.get(outcome.acceptance_decision_ref)
        if current is None or outcome.attempt_index > current.attempt_index:
            latest_by_decision[outcome.acceptance_decision_ref] = outcome

    candidates = _accepted_candidates(
        objects,
        run_id=run_id,
        candidate_outcomes=candidate_outcomes,
        workflow_attempts=workflow_attempts,
    )
    reconciled: list[GraphPublicationOutcome] = []
    seen_decisions: set[ObjectRef] = set()
    for (
        record,
        cycle_id,
        experiment_id,
        decision_ref,
        decision,
        factor_ref,
        factor,
    ) in candidates:
        if decision_ref in seen_decisions:
            continue
        seen_decisions.add(decision_ref)
        latest = latest_by_decision.get(decision_ref)
        next_attempt = 0 if latest is None else latest.attempt_index + 1
        entry: FactorLibraryEntry | None = None
        try:
            evaluation = objects.get_object(
                decision.evaluation_result_ref,
                EvaluationResult,
            )
            existing_entry = factor_library.get_entry(factor.factor_id)
            entry = existing_entry
            published = factor_library.publish_accepted(
                factor_spec=factor,
                factor_ref=factor_ref,
                evaluation_result=evaluation,
                evaluation_result_ref=decision.evaluation_result_ref,
                decision=decision,
                decision_ref=decision_ref,
                source_experiment_id=experiment_id,
                accepted_at=clock(),
            )
            entry = published
            entry_ref = objects.put_object(
                FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
                published.factor_id,
                published,
            )
            if (
                latest is not None
                and latest.status is GraphPublicationOutcomeStatus.PUBLISHED
            ):
                reconciled.append(latest)
                continue
            outcome = GraphPublicationOutcome(
                run_id=run_id,
                cycle_id=cycle_id,
                experiment_id=experiment_id,
                acceptance_decision_ref=decision_ref,
                factor_id=factor.factor_id,
                factor_ref=factor_ref,
                evaluation_result_ref=decision.evaluation_result_ref,
                attempt_index=next_attempt,
                status=GraphPublicationOutcomeStatus.PUBLISHED,
                entry_ref=entry_ref,
                reused=existing_entry is not None,
            )
            _persist_publication_outcome(objects, outcome)
            reconciled.append(outcome)
        except Exception as error:  # noqa: BLE001 - audit must retain root cause
            outcome = GraphPublicationOutcome(
                run_id=run_id,
                cycle_id=cycle_id,
                experiment_id=experiment_id,
                acceptance_decision_ref=decision_ref,
                factor_id=factor.factor_id,
                factor_ref=factor_ref,
                evaluation_result_ref=decision.evaluation_result_ref,
                attempt_index=next_attempt,
                status=GraphPublicationOutcomeStatus.PUBLICATION_FAILED,
                entry_ref=(
                    None
                    if entry is None
                    else ObjectRef(
                        FACTOR_LIBRARY_ENTRY_OBJECT_TYPE,
                        entry.factor_id,
                        entry.schema_version,
                    )
                ),
                failure=_publication_failure(
                    error,
                    record=record,
                    decision_ref=decision_ref,
                    factor_ref=factor_ref,
                ),
            )
            _persist_publication_outcome(objects, outcome)
            reconciled.append(outcome)
    return tuple(sorted(reconciled, key=_outcome_sort_key))


def write_graph_publication_report(
    run_root: Path,
    outcomes: Iterable[GraphPublicationOutcome],
    *,
    run_id: str,
) -> tuple[Path, Path]:
    """Write publication history and a manifest outside the GraphResult protocol."""
    _require_text(run_id, "run_id")
    selected = tuple(outcome for outcome in outcomes if outcome.run_id == run_id)
    ordered = tuple(sorted(selected, key=_outcome_sort_key))
    report_root = run_root / "graph"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "publication_outcomes.csv"
    columns = (
        "run_id",
        "cycle_id",
        "experiment_id",
        "acceptance_decision_id",
        "factor_id",
        "attempt_index",
        "status",
        "reused",
        "entry_id",
        "failure_stage",
        "failure_error_code",
        "failure_message",
        "failure_retryable",
        "factor_ref",
        "evaluation_result_ref",
        "acceptance_decision_ref",
        "entry_ref",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(_publication_rows(ordered))
    csv_path.write_text(output.getvalue(), encoding="utf-8")
    manifest_path = report_root / "publication_outcomes.json"
    manifest_path.write_text(
        to_json(
            {
                "run_id": run_id,
                "csv_path": "publication_outcomes.csv",
                "outcome_refs": tuple(
                    graph_publication_outcome_ref(outcome) for outcome in ordered
                ),
                "row_count": len(ordered),
                "schema_version": GRAPH_PUBLICATION_REPORT_SCHEMA_VERSION,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path


def _accepted_candidates(
    objects: LocalExperimentRepository,
    *,
    run_id: str,
    candidate_outcomes: Iterable[GraphCandidateOutcome],
    workflow_attempts: Iterable[WorkflowStepAttempt],
) -> tuple[
    tuple[
        ExperimentRecord | None,
        str,
        str,
        ObjectRef,
        AcceptanceDecision,
        ObjectRef,
        FactorSpec,
    ],
    ...,
]:
    records = tuple(
        sorted(
            (
                record
                for record in objects.list_experiments()
                if record.run_id == run_id and record.completed_at is not None
            ),
            key=lambda record: record.cycle_id,
        )
    )
    candidates: list[
        tuple[
            ExperimentRecord | None,
            str,
            str,
            ObjectRef,
            AcceptanceDecision,
            ObjectRef,
            FactorSpec,
        ]
    ] = []
    for record in records:
        factor_pairs = tuple(
            (factor_ref, objects.get_object(factor_ref, FactorSpec))
            for factor_ref in record.factor_refs
        )
        for decision_ref in record.acceptance_decision_refs:
            decision = objects.get_object(decision_ref, AcceptanceDecision)
            if not decision.accepted:
                continue
            matches = tuple(
                (factor_ref, factor)
                for factor_ref, factor in factor_pairs
                if factor.factor_id == decision.factor_id
            )
            if not matches:
                raise CoreValidationError("GRAPH_PUBLICATION_FACTOR_REFERENCE_MISSING")
            factor_ref, factor = min(matches, key=lambda item: item[0].object_id)
            candidates.append(
                (
                    record,
                    record.cycle_id,
                    record.experiment_id,
                    decision_ref,
                    decision,
                    factor_ref,
                    factor,
                )
            )
    hypothesis_refs = _hypothesis_refs_by_cycle(workflow_attempts, run_id=run_id)
    for outcome in candidate_outcomes:
        if (
            outcome.run_id != run_id
            or outcome.outcome is not GraphCandidateOutcomeStatus.ACCEPTED
            or outcome.acceptance_decision_ref is None
            or outcome.factor_ref is None
        ):
            continue
        decision = objects.get_object(
            outcome.acceptance_decision_ref,
            AcceptanceDecision,
        )
        if not decision.accepted:
            continue
        factor = objects.get_object(outcome.factor_ref, FactorSpec)
        hypothesis_ref = hypothesis_refs.get(outcome.attempt_id)
        experiment_id = (
            _experiment_id(run_id, outcome.attempt_id, hypothesis_ref)
            if hypothesis_ref is not None
            else "graph-partial-experiment-"
            + stable_sha256(
                {
                    "run_id": run_id,
                    "cycle_id": outcome.attempt_id,
                    "decision_ref": outcome.acceptance_decision_ref,
                }
            )[:24]
        )
        candidates.append(
            (
                None,
                outcome.attempt_id,
                experiment_id,
                outcome.acceptance_decision_ref,
                decision,
                outcome.factor_ref,
                factor,
            )
        )
    return tuple(candidates)


def _publication_failure(
    error: Exception,
    *,
    record: ExperimentRecord | None,
    decision_ref: ObjectRef,
    factor_ref: ObjectRef,
) -> FailureInfo:
    message = str(error) or type(error).__name__
    code_match = re.match(r"^[A-Z][A-Z0-9_]*", message)
    error_code = code_match.group(0) if code_match is not None else "PUBLICATION_FAILED"
    input_refs = (factor_ref, decision_ref)
    artifact_refs = ()
    if record is not None:
        input_refs = (
            ObjectRef(
                "experiment_record",
                record.experiment_id,
                record.schema_version,
            ),
            *input_refs,
        )
        artifact_refs = (record.config_snapshot_ref,)
    return FailureInfo(
        stage="factor_library_publication",
        error_code=error_code,
        message=message,
        retryable=True,
        input_refs=input_refs,
        artifact_refs=artifact_refs,
    )


def _hypothesis_refs_by_cycle(
    workflow_attempts: Iterable[WorkflowStepAttempt],
    *,
    run_id: str,
) -> dict[str, ObjectRef]:
    refs: dict[str, ObjectRef] = {}
    for attempt in workflow_attempts:
        if (
            attempt.run_id == run_id
            and attempt.step_id == "hypothesis"
            and attempt.status is WorkflowStepStatus.SUCCEEDED
            and len(attempt.output_object_refs) == 1
        ):
            refs[attempt.cycle_id] = attempt.output_object_refs[0]
    return refs


def _experiment_id(
    run_id: str,
    cycle_id: str,
    hypothesis_ref: ObjectRef,
) -> str:
    return (
        "experiment-"
        + stable_sha256(
            {
                "cycle_id": cycle_id,
                "hypothesis_ref": hypothesis_ref,
                "run_id": run_id,
            }
        )[:24]
    )


def _publication_rows(
    outcomes: Iterable[GraphPublicationOutcome],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": outcome.run_id,
            "cycle_id": outcome.cycle_id,
            "experiment_id": outcome.experiment_id,
            "acceptance_decision_id": outcome.acceptance_decision_ref.object_id,
            "factor_id": outcome.factor_id,
            "attempt_index": outcome.attempt_index,
            "status": outcome.status.value,
            "reused": outcome.reused,
            "entry_id": (
                "" if outcome.entry_ref is None else outcome.entry_ref.object_id
            ),
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
            "factor_ref": _ref_json(outcome.factor_ref),
            "evaluation_result_ref": _ref_json(outcome.evaluation_result_ref),
            "acceptance_decision_ref": _ref_json(outcome.acceptance_decision_ref),
            "entry_ref": _ref_json(outcome.entry_ref),
        }
        for outcome in outcomes
    )


def _persist_publication_outcome(
    objects: LocalExperimentRepository,
    outcome: GraphPublicationOutcome,
) -> None:
    """Persist one immutable outcome without changing an existing attempt."""
    objects.put_object(
        GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE,
        graph_publication_outcome_id(
            outcome.run_id,
            outcome.acceptance_decision_ref,
            outcome.attempt_index,
        ),
        outcome,
    )


def _outcome_sort_key(
    outcome: GraphPublicationOutcome,
) -> tuple[str, str, str, int]:
    return (
        outcome.cycle_id,
        outcome.factor_id,
        outcome.acceptance_decision_ref.object_id,
        outcome.attempt_index,
    )


def _ref_json(ref: ObjectRef | None) -> str:
    return "" if ref is None else to_json(ref)


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


__all__ = [
    "FACTOR_LIBRARY_ENTRY_OBJECT_TYPE",
    "GRAPH_PUBLICATION_OUTCOME_OBJECT_TYPE",
    "GRAPH_PUBLICATION_OUTCOME_SCHEMA_VERSION",
    "GRAPH_PUBLICATION_REPORT_SCHEMA_VERSION",
    "GraphPublicationOutcome",
    "GraphPublicationOutcomeStatus",
    "graph_publication_outcome_id",
    "graph_publication_outcome_ref",
    "load_graph_publication_outcomes",
    "reconcile_graph_publications",
    "write_graph_publication_report",
]
