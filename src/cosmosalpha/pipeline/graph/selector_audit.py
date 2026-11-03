"""Graph parent-selection audit objects and derived reports."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cosmosalpha.core import CoreValidationError, ObjectRef, stable_sha256, to_json
from cosmosalpha.pipeline.graph.core import QualityParetoArchive
from cosmosalpha.pipeline.graph.selector import ParentSelectionResult

GRAPH_SELECTOR_AUDIT_OBJECT_TYPE = "graph_selector_audit"
GRAPH_SELECTOR_AUDIT_SCHEMA_VERSION = "graph-selector-audit/1.0"
GRAPH_SELECTOR_AUDIT_REPORT_SCHEMA_VERSION = "graph-selector-audit-report/1.0"

type RewardVector = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GraphSelectorAudit:
    """Immutable audit of one deterministic parent-selection round."""

    run_id: str
    round_id: int
    quality_archive_ids: tuple[str, ...]
    eligible_parent_ids: tuple[str, ...]
    flow_fronts: tuple[tuple[str, ...], ...]
    selected_parent_ids: tuple[str, ...]
    flow_node_ids: tuple[str, ...]
    raw_rewards: tuple[RewardVector, ...]
    propagated_rewards: tuple[RewardVector, ...]
    flow_iterations: int
    flow_converged: bool
    schema_version: str = GRAPH_SELECTOR_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate stable IDs, Flow alignment, and deterministic front data."""
        _require_text(self.run_id, "run_id")
        _require_non_negative_int(self.round_id, "round_id")
        _require_id_tuple(self.quality_archive_ids, "quality_archive_ids")
        _require_id_tuple(self.eligible_parent_ids, "eligible_parent_ids")
        _require_id_tuple(self.selected_parent_ids, "selected_parent_ids")
        _require_id_tuple(self.flow_node_ids, "flow_node_ids")
        if not isinstance(self.flow_fronts, tuple):
            raise CoreValidationError("flow_fronts must be a tuple")
        flattened_fronts: list[str] = []
        for index, front in enumerate(self.flow_fronts):
            _require_id_tuple(front, f"flow_fronts[{index}]")
            if tuple(sorted(front)) != front:
                raise CoreValidationError(
                    f"flow_fronts[{index}] must use deterministic sorted order"
                )
            flattened_fronts.extend(front)
        if tuple(sorted(flattened_fronts)) != tuple(
            sorted(self.eligible_parent_ids)
        ) or len(flattened_fronts) != len(set(flattened_fronts)):
            raise CoreValidationError(
                "flow_fronts must partition eligible_parent_ids exactly"
            )
        if not set(self.selected_parent_ids).issubset(self.eligible_parent_ids):
            raise CoreValidationError("selected_parent_ids must be eligible")
        if not set(self.eligible_parent_ids).issubset(self.flow_node_ids):
            raise CoreValidationError(
                "eligible_parent_ids must be present in flow_node_ids"
            )
        _require_reward_matrix(self.raw_rewards, len(self.flow_node_ids), "raw_rewards")
        _require_reward_matrix(
            self.propagated_rewards,
            len(self.flow_node_ids),
            "propagated_rewards",
        )
        _require_non_negative_int(self.flow_iterations, "flow_iterations")
        if type(self.flow_converged) is not bool:
            raise CoreValidationError("flow_converged must be a boolean")
        if self.schema_version != GRAPH_SELECTOR_AUDIT_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphSelectorAudit schema_version")

    @classmethod
    def from_selection(
        cls,
        *,
        run_id: str,
        round_id: int,
        quality_archive: QualityParetoArchive,
        selection: ParentSelectionResult,
    ) -> GraphSelectorAudit:
        """Project an existing selector result into an audit-only object."""
        _require_text(run_id, "run_id")
        _require_non_negative_int(round_id, "round_id")
        if not isinstance(quality_archive, QualityParetoArchive):
            raise CoreValidationError("quality_archive must be QualityParetoArchive")
        if not isinstance(selection, ParentSelectionResult):
            raise CoreValidationError("selection must be ParentSelectionResult")
        flow = selection.flow_result
        return cls(
            run_id=run_id,
            round_id=round_id,
            quality_archive_ids=tuple(
                point.node_id for point in quality_archive.points
            ),
            eligible_parent_ids=selection.eligible_parent_ids,
            flow_fronts=selection.pareto_fronts,
            selected_parent_ids=selection.selected_parent_ids,
            flow_node_ids=flow.node_ids,
            raw_rewards=flow.raw_rewards,
            propagated_rewards=flow.propagated_rewards,
            flow_iterations=flow.iterations_run,
            flow_converged=flow.converged,
        )


def graph_selector_audit_id(run_id: str, round_id: int) -> str:
    """Return the stable object ID for one run and selection round."""
    _require_text(run_id, "run_id")
    _require_non_negative_int(round_id, "round_id")
    return (
        "graph-selector-audit-"
        + stable_sha256({"run_id": run_id, "round_id": round_id})[:24]
    )


def graph_selector_audit_ref(audit: GraphSelectorAudit) -> ObjectRef:
    """Return the repository reference for one selector audit."""
    if not isinstance(audit, GraphSelectorAudit):
        raise CoreValidationError("audit must be GraphSelectorAudit")
    return ObjectRef(
        GRAPH_SELECTOR_AUDIT_OBJECT_TYPE,
        graph_selector_audit_id(audit.run_id, audit.round_id),
        audit.schema_version,
    )


def persist_graph_selector_audit(
    objects: object,
    audit: GraphSelectorAudit,
) -> ObjectRef:
    """Persist one audit through the repository's immutable idempotent write."""
    if not isinstance(audit, GraphSelectorAudit):
        raise CoreValidationError("audit must be GraphSelectorAudit")
    put_object = getattr(objects, "put_object", None)
    if not callable(put_object):
        raise CoreValidationError("objects must provide put_object")
    ref = graph_selector_audit_ref(audit)
    put_object(GRAPH_SELECTOR_AUDIT_OBJECT_TYPE, ref.object_id, audit)
    return ref


def load_graph_selector_audits(
    objects: object,
    *,
    run_id: str,
) -> tuple[GraphSelectorAudit, ...]:
    """Load selector audits for one run in stable round order."""
    _require_text(run_id, "run_id")
    list_object_ids = getattr(objects, "list_object_ids", None)
    get_object = getattr(objects, "get_object", None)
    if not callable(list_object_ids) or not callable(get_object):
        raise CoreValidationError(
            "objects must provide Graph selector audit repository methods"
        )
    audits = []
    for object_id in list_object_ids(GRAPH_SELECTOR_AUDIT_OBJECT_TYPE):
        audit = get_object(
            ObjectRef(
                GRAPH_SELECTOR_AUDIT_OBJECT_TYPE,
                object_id,
                GRAPH_SELECTOR_AUDIT_SCHEMA_VERSION,
            ),
            GraphSelectorAudit,
        )
        if audit.run_id == run_id:
            audits.append(audit)
    return tuple(sorted(audits, key=lambda item: item.round_id))


_REPORT_COLUMNS = (
    "run_id",
    "round_id",
    "quality_archive_ids",
    "eligible_parent_ids",
    "flow_fronts",
    "selected_parent_ids",
    "flow_node_ids",
    "raw_rewards",
    "propagated_rewards",
    "flow_iterations",
    "flow_converged",
)


def graph_selector_audit_rows(
    audits: Iterable[GraphSelectorAudit],
) -> tuple[dict[str, object], ...]:
    """Convert audits into deterministic, lossless CSV rows."""
    ordered = tuple(sorted(audits, key=lambda item: (item.round_id, item.run_id)))
    return tuple(
        {
            "run_id": audit.run_id,
            "round_id": audit.round_id,
            "quality_archive_ids": "|".join(audit.quality_archive_ids),
            "eligible_parent_ids": "|".join(audit.eligible_parent_ids),
            "flow_fronts": to_json(audit.flow_fronts),
            "selected_parent_ids": "|".join(audit.selected_parent_ids),
            "flow_node_ids": "|".join(audit.flow_node_ids),
            "raw_rewards": to_json(audit.raw_rewards),
            "propagated_rewards": to_json(audit.propagated_rewards),
            "flow_iterations": audit.flow_iterations,
            "flow_converged": audit.flow_converged,
        }
        for audit in ordered
    )


def write_graph_selector_audit_report(
    run_root: Path,
    audits: Iterable[GraphSelectorAudit],
    *,
    run_id: str,
) -> tuple[Path, Path]:
    """Write the selector CSV and independent manifest."""
    _require_text(run_id, "run_id")
    selected = tuple(audit for audit in audits if audit.run_id == run_id)
    rows = graph_selector_audit_rows(selected)
    report_root = run_root / "graph"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "selector_audits.csv"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_REPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(output.getvalue(), encoding="utf-8")

    ordered = tuple(sorted(selected, key=lambda item: (item.round_id, item.run_id)))
    manifest_path = report_root / "selector_audits.json"
    manifest_path.write_text(
        to_json(
            {
                "run_id": run_id,
                "csv_path": "selector_audits.csv",
                "audit_refs": tuple(
                    graph_selector_audit_ref(audit) for audit in ordered
                ),
                "row_count": len(rows),
                "schema_version": GRAPH_SELECTOR_AUDIT_REPORT_SCHEMA_VERSION,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise CoreValidationError(f"{field_name} must be non-negative")


def _require_id_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    if not all(type(item) is str and item and item == item.strip() for item in value):
        raise CoreValidationError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise CoreValidationError(f"{field_name} must contain unique IDs")


def _require_reward_matrix(
    value: object,
    expected_rows: int,
    field_name: str,
) -> None:
    if not isinstance(value, tuple) or len(value) != expected_rows:
        raise CoreValidationError(
            f"{field_name} must contain one reward vector per Flow node"
        )
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != 3:
            raise CoreValidationError(
                f"{field_name}[{row_index}] must contain three values"
            )
        for column_index, item in enumerate(row):
            if type(item) not in (int, float) or not math.isfinite(float(item)):
                raise CoreValidationError(
                    f"{field_name}[{row_index}][{column_index}] must be finite"
                )


__all__ = [
    "GRAPH_SELECTOR_AUDIT_OBJECT_TYPE",
    "GRAPH_SELECTOR_AUDIT_REPORT_SCHEMA_VERSION",
    "GRAPH_SELECTOR_AUDIT_SCHEMA_VERSION",
    "GraphSelectorAudit",
    "graph_selector_audit_id",
    "graph_selector_audit_ref",
    "graph_selector_audit_rows",
    "load_graph_selector_audits",
    "persist_graph_selector_audit",
    "write_graph_selector_audit_report",
]
