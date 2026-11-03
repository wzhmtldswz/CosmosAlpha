"""Human-readable run reports derived from authoritative stored objects."""

from __future__ import annotations

import csv
import io
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from cosmosalpha.agents import (
    AcceptanceDecision,
    EvaluationResult,
    ExperimentRecord,
    FactorSpec,
)
from cosmosalpha.core import to_json
from cosmosalpha.storage.local import LocalExperimentRepository, RunStorageLayout

_LEADING_COLUMNS = (
    "branch_cycle",
    "expression",
    "status",
    "accepted",
)
_GRAPH_LEADING_COLUMNS = (
    "run_id",
    "cycle_id",
    "expression",
    "status",
    "accepted",
)
_METRIC_COLUMNS = (
    "rank_ic_mean",
    "rank_ic_ir",
    "ic_mean",
    "ic_ir",
    "rank_ic_positive_rate",
    "rank_ic_std",
    "rank_ic_oos_mean",
    "rank_ic_oos_min",
    "factor_max_abs_daily_correlation",
    "coverage",
    "total_date_count",
    "effective_date_count",
    "effective_date_ratio",
    "effective_sample_count",
)
_GRAPH_NORMALIZED_COLUMNS = (
    "normalized_rank_ic",
    "normalized_rank_ic_std",
    "normalized_top20_mean_return",
)
_TRAILING_COLUMNS = (
    "reason_codes",
    "name",
    "description",
    "factor_id",
    "expected_direction",
    "frequency",
    "run_id",
    "cycle_id",
    "experiment_id",
    "data_snapshot_id",
    "evaluation_protocol_id",
    "raw_factor_values_artifact_id",
    "evaluated_factor_values_artifact_id",
)
_GRAPH_TRAILING_COLUMNS = tuple(
    column for column in _TRAILING_COLUMNS if column not in {"run_id", "cycle_id"}
)
_STATIC_COLUMNS = frozenset(
    (
        *_LEADING_COLUMNS,
        *_METRIC_COLUMNS,
        *_GRAPH_NORMALIZED_COLUMNS,
        *_TRAILING_COLUMNS,
    )
)


def write_run_reports(
    layout: RunStorageLayout,
    records: tuple[ExperimentRecord, ...],
    *,
    objects: LocalExperimentRepository,
    include_branch_cycle: bool = True,
    graph_normalized_metrics_by_factor_id: (
        Mapping[str, Mapping[str, float]] | None
    ) = None,
) -> tuple[tuple[str, ...], str]:
    """Write per-cycle reports and the final all-candidate CSV view."""
    cycle_csv_paths: list[str] = []
    all_rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.cycle_id):
        cycle_root = layout.cycles_root / record.cycle_id
        cycle_root.mkdir(parents=True, exist_ok=True)
        _write_text(
            cycle_root / "result.json",
            to_json(
                {
                    "experiment_record": record,
                    "factor_results_csv": "factors.csv",
                    "schema_version": "cycle-result-manifest/1.0",
                }
            )
            + "\n",
        )
        rows = _cycle_rows(
            record,
            objects,
            include_branch_cycle=include_branch_cycle,
            graph_normalized_metrics_by_factor_id=(
                graph_normalized_metrics_by_factor_id or {}
            ),
        )
        all_rows.extend(rows)
        csv_path = cycle_root / "factors.csv"
        _write_csv(csv_path, rows, include_branch_cycle=include_branch_cycle)
        cycle_csv_paths.append(str(csv_path))

    _write_csv(
        layout.factor_library_csv_path,
        _final_factor_rows(all_rows),
        include_branch_cycle=include_branch_cycle,
        include_accepted=False,
    )
    return tuple(cycle_csv_paths), str(layout.factor_library_csv_path)


def _cycle_rows(
    record: ExperimentRecord,
    objects: LocalExperimentRepository,
    *,
    include_branch_cycle: bool,
    graph_normalized_metrics_by_factor_id: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, object], ...]:
    factors = tuple(objects.get_object(ref, FactorSpec) for ref in record.factor_refs)
    evaluations = {
        value.factor_id: value
        for value in (
            objects.get_object(ref, EvaluationResult)
            for ref in record.evaluation_result_refs
        )
    }
    decisions = {
        value.factor_id: value
        for value in (
            objects.get_object(ref, AcceptanceDecision)
            for ref in record.acceptance_decision_refs
        )
    }
    return tuple(
        _factor_row(
            factor=factor,
            record=record,
            evaluation=evaluations.get(factor.factor_id),
            decision=decisions.get(factor.factor_id),
            include_branch_cycle=include_branch_cycle,
            graph_normalized_metrics=graph_normalized_metrics_by_factor_id.get(
                factor.factor_id
            ),
        )
        for factor in factors
    )


def _final_factor_rows(
    rows: Iterable[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Keep the best result per factor and order by its cycle label."""
    status_order = {"accepted": 0, "rejected": 1, "failed": 2}
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        factor_id = str(row["factor_id"])
        current = selected.get(factor_id)
        if (
            current is None
            or status_order[str(row["status"])] < status_order[str(current["status"])]
        ):
            selected[factor_id] = row
    return tuple(
        sorted(
            selected.values(),
            key=lambda row: (
                str(row["branch_cycle"] if "branch_cycle" in row else row["cycle_id"]),
                str(row["expression"]),
                str(row["factor_id"]),
            ),
        )
    )


def _factor_row(
    *,
    factor: FactorSpec,
    record: ExperimentRecord,
    evaluation: EvaluationResult | None,
    decision: AcceptanceDecision | None,
    include_branch_cycle: bool,
    graph_normalized_metrics: Mapping[str, float] | None,
) -> dict[str, object]:
    accepted = None if decision is None else decision.accepted
    effective_date_ratio = (
        ""
        if evaluation is None or evaluation.total_date_count in (None, 0)
        else evaluation.effective_date_count / evaluation.total_date_count
    )
    row: dict[str, object] = {
        "run_id": record.run_id,
        "cycle_id": record.cycle_id,
        "experiment_id": record.experiment_id,
        "factor_id": factor.factor_id,
        "name": factor.name,
        "expression": factor.expression,
        "description": factor.description,
        "frequency": factor.frequency,
        "expected_direction": factor.expected_direction,
        "status": (
            "failed"
            if decision is None
            else ("accepted" if decision.accepted else "rejected")
        ),
        "accepted": "" if accepted is None else accepted,
        "reason_codes": "" if decision is None else "|".join(decision.reason_codes),
        "data_snapshot_id": ("" if evaluation is None else evaluation.data_snapshot_id),
        "evaluation_protocol_id": (
            "" if evaluation is None else evaluation.evaluation_protocol_id
        ),
        "coverage": "" if evaluation is None else evaluation.coverage,
        "total_date_count": ("" if evaluation is None else evaluation.total_date_count),
        "effective_date_count": (
            "" if evaluation is None else evaluation.effective_date_count
        ),
        "effective_date_ratio": effective_date_ratio,
        "effective_sample_count": (
            "" if evaluation is None else evaluation.effective_sample_count
        ),
        "raw_factor_values_artifact_id": (
            "" if evaluation is None else evaluation.raw_factor_values_ref.artifact_id
        ),
        "evaluated_factor_values_artifact_id": (
            ""
            if evaluation is None
            else evaluation.evaluated_factor_values_ref.artifact_id
        ),
    }
    if include_branch_cycle:
        row["branch_cycle"] = _branch_cycle_label(record.cycle_id)
    if evaluation is not None:
        row.update({metric.name: metric.value for metric in evaluation.metrics})
    if graph_normalized_metrics is not None:
        row.update(
            {
                column: graph_normalized_metrics[column]
                for column in _GRAPH_NORMALIZED_COLUMNS
                if column in graph_normalized_metrics
            }
        )
    return row


_BRANCH_CYCLE_PATTERN = re.compile(r"^(branch-[0-9]+)-(cycle-[0-9]+)$")


def _branch_cycle_label(cycle_id: str) -> str:
    """Return a stable human-readable branch/cycle label for CSV reports."""
    match = _BRANCH_CYCLE_PATTERN.fullmatch(cycle_id)
    if match is not None:
        return f"{match.group(1)}/{match.group(2)}"
    return f"branch-001/{cycle_id}"


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    include_branch_cycle: bool,
    include_accepted: bool = True,
) -> None:
    values = tuple(rows)
    extra_metric_columns = sorted(
        {key for row in values for key in row if key not in _STATIC_COLUMNS}
    )
    output = io.StringIO(newline="")
    fieldnames = (
        *(_LEADING_COLUMNS if include_branch_cycle else _GRAPH_LEADING_COLUMNS),
        *_METRIC_COLUMNS,
        *(() if include_branch_cycle else _GRAPH_NORMALIZED_COLUMNS),
        *extra_metric_columns,
        *(_TRAILING_COLUMNS if include_branch_cycle else _GRAPH_TRAILING_COLUMNS),
    )
    if not include_accepted:
        fieldnames = tuple(column for column in fieldnames if column != "accepted")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: value
            for key, value in row.items()
            if include_accepted or key != "accepted"
        }
        for row in values
    )
    _write_text(path, output.getvalue())


def _write_text(path: Path, value: str) -> None:
    """Atomically replace one derived report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
