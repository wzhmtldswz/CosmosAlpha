"""Deterministic comparison-baseline selection for Feedback evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    ExperimentRecord,
    FactorSpec,
    MetricValue,
)
from cosmosalpha.core import CoreError, ObjectRef


class FeedbackHistoryRepository(Protocol):
    """Read-only repository methods needed for baseline selection."""

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        """Return persisted experiment records."""

    def get_object[ObjectT](
        self,
        ref: ObjectRef,
        target_type: type[ObjectT],
    ) -> ObjectT:
        """Resolve one strict object reference."""


@dataclass(frozen=True, slots=True)
class _BaselineCandidate:
    record: ExperimentRecord
    factor_ref: ObjectRef
    evaluation_ref: ObjectRef
    evaluation: EvaluationResult
    metric: MetricValue
    decision_ref: ObjectRef | None
    accepted: bool

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        experiment_ref = ObjectRef(
            "experiment_record",
            self.record.experiment_id,
            self.record.schema_version,
        )
        refs = (
            experiment_ref,
            self.factor_ref,
            self.evaluation.factor_preprocessing_result_ref,
            self.evaluation_ref,
        )
        return refs if self.decision_ref is None else (*refs, self.decision_ref)


def select_comparison_baseline_refs(
    repository: FeedbackHistoryRepository,
    *,
    parent_ids: tuple[str, ...],
    current_evaluation_refs: tuple[ObjectRef, ...],
    acceptance_metric: str,
) -> tuple[ObjectRef, ...]:
    """Select the best comparable candidate from the current branch ancestry."""
    current = tuple(
        repository.get_object(ref, EvaluationResult) for ref in current_evaluation_refs
    )
    current_metrics = tuple(
        metric
        for evaluation in current
        for metric in evaluation.metrics
        if metric.name == acceptance_metric
    )
    if not current or not current_metrics:
        return ()
    comparison_key = (
        current[0].data_snapshot_id,
        current[0].evaluation_protocol_id,
        current[0].preprocessing_protocol_id,
    )
    direction = current_metrics[0].direction
    if direction not in ("higher_is_better", "lower_is_better"):
        return ()
    if any(
        (
            evaluation.data_snapshot_id,
            evaluation.evaluation_protocol_id,
            evaluation.preprocessing_protocol_id,
        )
        != comparison_key
        for evaluation in current
    ) or any(metric.direction != direction for metric in current_metrics):
        return ()

    records = repository.list_experiments()
    ancestor_ids = _ancestor_ids(records, parent_ids)
    candidates = tuple(
        candidate
        for record in records
        if record.experiment_id in ancestor_ids
        and record.completed_at is not None
        and record.data_version == comparison_key[0]
        for candidate in _record_candidates(
            repository,
            record,
            comparison_key=comparison_key,
            acceptance_metric=acceptance_metric,
            direction=direction,
        )
    )
    if not candidates:
        return ()
    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    eligible = accepted or candidates
    return min(
        eligible,
        key=lambda candidate: _selection_key(candidate, direction),
    ).refs


def _record_candidates(
    repository: FeedbackHistoryRepository,
    record: ExperimentRecord,
    *,
    comparison_key: tuple[str, str, str],
    acceptance_metric: str,
    direction: str,
) -> tuple[_BaselineCandidate, ...]:
    factors = {
        factor.factor_id: (ref, factor)
        for ref in record.factor_refs
        if (factor := _resolve(repository, ref, FactorSpec)) is not None
    }
    decisions = tuple(
        (ref, decision)
        for ref in record.acceptance_decision_refs
        if (decision := _resolve(repository, ref, AcceptanceDecision)) is not None
    )
    candidates: list[_BaselineCandidate] = []
    for evaluation_ref in record.evaluation_result_refs:
        evaluation = _resolve(repository, evaluation_ref, EvaluationResult)
        if (
            evaluation is None
            or (
                evaluation.data_snapshot_id,
                evaluation.evaluation_protocol_id,
                evaluation.preprocessing_protocol_id,
            )
            != comparison_key
        ):
            continue
        metric = next(
            (
                value
                for value in evaluation.metrics
                if value.name == acceptance_metric and value.direction == direction
            ),
            None,
        )
        factor_entry = factors.get(evaluation.factor_id)
        if metric is None or factor_entry is None:
            continue
        decision_entry = next(
            (
                (ref, decision)
                for ref, decision in decisions
                if decision.factor_id == evaluation.factor_id
                and decision.evaluation_result_ref == evaluation_ref
            ),
            None,
        )
        candidates.append(
            _BaselineCandidate(
                record=record,
                factor_ref=factor_entry[0],
                evaluation_ref=evaluation_ref,
                evaluation=evaluation,
                metric=metric,
                decision_ref=None if decision_entry is None else decision_entry[0],
                accepted=(
                    False if decision_entry is None else decision_entry[1].accepted
                ),
            )
        )
    return tuple(candidates)


def _ancestor_ids(
    records: tuple[ExperimentRecord, ...],
    parent_ids: tuple[str, ...],
) -> frozenset[str]:
    by_id = {record.experiment_id: record for record in records}
    found: set[str] = set()
    pending = list(parent_ids)
    while pending:
        experiment_id = pending.pop(0)
        if experiment_id in found:
            continue
        record = by_id.get(experiment_id)
        if record is None:
            continue
        found.add(experiment_id)
        pending.extend(record.parent_ids)
    return frozenset(found)


def _selection_key(
    candidate: _BaselineCandidate,
    direction: str,
) -> tuple[float, float, str]:
    completed_at = candidate.record.completed_at
    if not isinstance(completed_at, datetime):
        raise ValueError("baseline candidate must be completed")
    metric_rank = (
        -candidate.metric.value
        if direction == "higher_is_better"
        else candidate.metric.value
    )
    return metric_rank, -completed_at.timestamp(), candidate.evaluation.factor_id


def _resolve[ObjectT](
    repository: FeedbackHistoryRepository,
    ref: ObjectRef,
    target_type: type[ObjectT],
) -> ObjectT | None:
    try:
        return repository.get_object(ref, target_type)
    except CoreError:
        return None
