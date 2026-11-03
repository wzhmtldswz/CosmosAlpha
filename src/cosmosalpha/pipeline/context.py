"""Deterministic experiment-history selection for bounded research contexts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from cosmosalpha.agents import (
    ExperimentLesson,
    ExperimentRecord,
    ResearchContext,
)
from cosmosalpha.core import CoreValidationError, ObjectRef, to_json
from cosmosalpha.pipeline.lesson import lesson_ref_for_experiment


class ExperimentHistoryRepository(Protocol):
    """Read-only object repository subset used to build a research context."""

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        """Return completed experiment records in stable order."""

    def get_object[ObjectT](
        self,
        ref: ObjectRef,
        target_type: type[ObjectT],
    ) -> ObjectT:
        """Resolve one persisted domain object."""


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Bounded context and factor references selected from experiment history."""

    context: ResearchContext
    existing_factor_refs: tuple[ObjectRef, ...]
    selected_lesson_refs: tuple[ObjectRef, ...]
    selected_experiment_ids: tuple[str, ...]
    omitted_experiment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate deterministic selection results."""
        if not isinstance(self.context, ResearchContext):
            raise CoreValidationError("context must be ResearchContext")
        _require_ref_tuple(self.existing_factor_refs, "existing_factor_refs")
        _require_ref_tuple(self.selected_lesson_refs, "selected_lesson_refs")
        _require_string_tuple(self.selected_experiment_ids, "selected_experiment_ids")
        _require_string_tuple(self.omitted_experiment_ids, "omitted_experiment_ids")


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    """One sortable historical experiment with its structured Lesson."""

    record: ExperimentRecord
    lesson_ref: ObjectRef
    lesson: ExperimentLesson
    lineage_distance: int | None
    direction_match: bool
    mechanism_match: bool
    failure_match: bool


class ContextBuilder:
    """Build a deterministic, token-bounded history view for one research cycle."""

    def __init__(
        self,
        repository: ExperimentHistoryRepository,
    ) -> None:
        """Create a structured history builder backed by experiment objects."""
        self._repository = repository

    def build(
        self,
        base_context: ResearchContext,
        *,
        data_version: str,
        parent_ids: tuple[str, ...],
        max_history_items: int,
        max_input_tokens: int,
        current_run_id: str | None = None,
        exclude_run_cycle: tuple[str, str] | None = None,
    ) -> ContextBuildResult:
        """Select history and existing factors for one cycle.

        Only completed experiments using ``data_version`` participate. Parent
        ancestry is ranked first, followed by direction, mechanism, failure-code
        similarity, and recency. Lessons are never sliced: a complete Lesson is
        selected or omitted.

        Args:
            base_context: Stable non-history context for this research run.
            data_version: Data snapshot identifier required for comparable history.
            parent_ids: Immediate parent experiment identifiers for this cycle.
            max_history_items: Maximum number of Lessons injected into the prompt.
            max_input_tokens: Total input budget before prompt template overhead.
            current_run_id: Current run whose non-ancestor cycles are excluded.
            exclude_run_cycle: Current run/cycle pair omitted during resume rebuilding.

        Returns:
            A context with selected summaries and deduplicated existing factor refs.
        """
        if not isinstance(base_context, ResearchContext):
            raise CoreValidationError("base_context must be ResearchContext")
        _require_string(data_version, "data_version")
        _require_string_tuple(parent_ids, "parent_ids")
        _require_positive_int(max_input_tokens, "max_input_tokens")
        if type(max_history_items) is not int or max_history_items < 0:
            raise CoreValidationError("max_history_items must be non-negative")
        if exclude_run_cycle is not None:
            _require_string_tuple(exclude_run_cycle, "exclude_run_cycle")
            if len(exclude_run_cycle) != 2:
                raise CoreValidationError(
                    "exclude_run_cycle must contain run and cycle"
                )
        if current_run_id is not None:
            _require_string(current_run_id, "current_run_id")

        compatible_records = tuple(
            record
            for record in self._repository.list_experiments()
            if record.completed_at is not None and record.data_version == data_version
        )
        all_lineage_distances = _lineage_distances(compatible_records, parent_ids)
        records = tuple(
            record
            for record in compatible_records
            if (
                exclude_run_cycle is None
                or (record.run_id, record.cycle_id) != exclude_run_cycle
            )
            and (
                current_run_id is None
                or record.run_id != current_run_id
                or record.experiment_id in all_lineage_distances
            )
        )
        lineage_distances = _lineage_distances(records, parent_ids)
        lesson_records = tuple(
            (
                record,
                lesson_ref_for_experiment(record.experiment_id),
                _optional_object(
                    self._repository,
                    lesson_ref_for_experiment(record.experiment_id),
                    ExperimentLesson,
                ),
            )
            for record in records
        )
        ancestor_lessons = tuple(
            lesson
            for record, _, lesson in lesson_records
            if lesson is not None and record.experiment_id in lineage_distances
        )
        target_direction_id = (
            None
            if base_context.research_direction is None
            else base_context.research_direction.direction_id
        )
        target_mechanisms = {
            _normalized_text(lesson.hypothesis_mechanism) for lesson in ancestor_lessons
        }
        target_failure_codes = {
            code for lesson in ancestor_lessons for code in lesson.stable_failure_codes
        }
        entries = tuple(
            _HistoryEntry(
                record=record,
                lesson_ref=lesson_ref,
                lesson=lesson,
                lineage_distance=lineage_distances.get(record.experiment_id),
                direction_match=(
                    target_direction_id is not None
                    and lesson.research_direction_id == target_direction_id
                ),
                mechanism_match=(
                    bool(target_mechanisms)
                    and _normalized_text(lesson.hypothesis_mechanism)
                    in target_mechanisms
                ),
                failure_match=bool(
                    target_failure_codes.intersection(lesson.stable_failure_codes)
                ),
            )
            for record, lesson_ref, lesson in lesson_records
            if lesson is not None
        )
        ranked = tuple(sorted(entries, key=_history_sort_key))
        budget = _history_budget_tokens(base_context, max_input_tokens)
        selected: list[_HistoryEntry] = []
        used_tokens = 0
        for entry in ranked:
            if len(selected) == max_history_items:
                break
            lesson_tokens = _estimate_tokens(to_json(entry.lesson))
            if used_tokens + lesson_tokens > budget:
                continue
            selected.append(entry)
            used_tokens += lesson_tokens

        selected_ids = tuple(entry.record.experiment_id for entry in selected)
        selected_lesson_refs = tuple(entry.lesson_ref for entry in selected)
        selected_id_set = set(selected_ids)
        omitted_ids = tuple(
            entry.record.experiment_id
            for entry in ranked
            if entry.record.experiment_id not in selected_id_set
        )
        existing_factor_refs = _sorted_unique_refs(
            ref for record in records for ref in record.factor_refs
        )
        context = replace(
            base_context,
            history_summaries=(),
        )
        return ContextBuildResult(
            context=context,
            existing_factor_refs=existing_factor_refs,
            selected_lesson_refs=selected_lesson_refs,
            selected_experiment_ids=selected_ids,
            omitted_experiment_ids=omitted_ids,
        )


def _lineage_distances(
    records: Iterable[ExperimentRecord],
    parent_ids: tuple[str, ...],
) -> dict[str, int]:
    """Return shortest parent distance for every reachable historical experiment."""
    by_id = {record.experiment_id: record for record in records}
    distances: dict[str, int] = {}
    pending = [(parent_id, 0) for parent_id in parent_ids]
    while pending:
        experiment_id, distance = pending.pop(0)
        if experiment_id in distances and distances[experiment_id] <= distance:
            continue
        record = by_id.get(experiment_id)
        if record is None:
            continue
        distances[experiment_id] = distance
        pending.extend((parent_id, distance + 1) for parent_id in record.parent_ids)
    return distances


def _history_sort_key(
    entry: _HistoryEntry,
) -> tuple[int, int, int, int, int, str]:
    """Order structured Lessons without relying on repository iteration order."""
    completed_at = entry.record.completed_at
    if completed_at is None:
        raise CoreValidationError("history entry must be completed")
    lineage_rank = (
        entry.lineage_distance if entry.lineage_distance is not None else 1_000_000
    )
    return (
        lineage_rank,
        0 if entry.direction_match else 1,
        0 if entry.mechanism_match else 1,
        0 if entry.failure_match else 1,
        -_timestamp_microseconds(completed_at),
        entry.record.experiment_id,
    )


def _optional_object[ObjectT](
    repository: ExperimentHistoryRepository,
    ref: ObjectRef,
    target_type: type[ObjectT],
) -> ObjectT | None:
    """Resolve an optional historical object without hiding its absence in summaries."""
    try:
        return repository.get_object(ref, target_type)
    except Exception:
        return None


def _history_budget_tokens(context: ResearchContext, max_input_tokens: int) -> int:
    """Reserve context fields before allocating the remaining prompt budget to history."""
    empty_history_context = replace(context, history_summaries=())
    return max(0, max_input_tokens - _estimate_tokens(to_json(empty_history_context)))


def _normalized_text(value: str) -> str:
    """Normalize exact mechanism matching without fuzzy or vector retrieval."""
    return " ".join(value.casefold().split())


def _estimate_tokens(value: str) -> int:
    """Use the same conservative character estimate as prompt rendering."""
    return (len(value) + 3) // 4


def _timestamp_microseconds(value: datetime) -> int:
    """Convert one UTC datetime to an integer without float ordering artifacts."""
    return int(value.timestamp() * 1_000_000)


def _sorted_unique_refs(refs: Iterable[ObjectRef]) -> tuple[ObjectRef, ...]:
    """Deduplicate references in a stable storage-independent order."""
    return tuple(
        sorted(
            set(refs),
            key=lambda ref: (ref.object_type, ref.object_id, ref.schema_version),
        )
    )


def _require_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or not all(
        type(item) is str and item for item in value
    ):
        raise CoreValidationError(f"{field_name} must be a tuple of non-empty strings")


def _require_ref_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, ObjectRef) for item in value
    ):
        raise CoreValidationError(f"{field_name} must be a tuple of ObjectRef")


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise CoreValidationError(f"{field_name} must be a positive integer")
