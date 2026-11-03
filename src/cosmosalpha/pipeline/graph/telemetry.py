"""Optional Graph performance telemetry kept outside search state."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cosmosalpha.core import CoreValidationError, ObjectRef, stable_sha256, to_json
from cosmosalpha.evaluation import CorrelationTiming
from cosmosalpha.pipeline.graph.flow import GraphFlowResult
from cosmosalpha.storage import LocalExperimentRepository

GRAPH_TELEMETRY_OBJECT_TYPE = "graph_telemetry"
GRAPH_TELEMETRY_SCHEMA_VERSION = "graph-telemetry/1.0"
GRAPH_TELEMETRY_REPORT_SCHEMA_VERSION = "graph-telemetry-report/1.0"


@dataclass(frozen=True, slots=True)
class GraphTelemetryRecord:
    """One round's derived timing observations, excluded from Graph identity."""

    run_id: str
    round_id: int
    correlation_total_seconds: float
    exposure_decode_seconds: float
    pairwise_correlation_seconds: float
    w_construction_seconds: float
    flow_total_seconds: float
    flow_iterations: tuple[int, ...]
    flow_converged: tuple[bool, ...]
    schema_version: str = GRAPH_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate telemetry values without imposing algorithmic semantics."""
        _require_text(self.run_id, "run_id")
        if type(self.round_id) is not int or self.round_id < 0:
            raise CoreValidationError("round_id must be non-negative")
        for field_name in (
            "correlation_total_seconds",
            "exposure_decode_seconds",
            "pairwise_correlation_seconds",
            "w_construction_seconds",
            "flow_total_seconds",
        ):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise CoreValidationError(f"{field_name} must be finite")
            if float(value) < 0.0:
                raise CoreValidationError(f"{field_name} must be non-negative")
        if not isinstance(self.flow_iterations, tuple):
            raise CoreValidationError("flow_iterations must be a tuple")
        if not all(type(value) is int and value >= 0 for value in self.flow_iterations):
            raise CoreValidationError("flow_iterations must contain non-negative ints")
        if not isinstance(self.flow_converged, tuple):
            raise CoreValidationError("flow_converged must be a tuple")
        if not all(type(value) is bool for value in self.flow_converged):
            raise CoreValidationError("flow_converged must contain booleans")
        if len(self.flow_iterations) != len(self.flow_converged):
            raise CoreValidationError(
                "flow_iterations and flow_converged must have equal lengths"
            )
        if self.schema_version != GRAPH_TELEMETRY_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphTelemetryRecord schema_version")


def graph_telemetry_id(run_id: str, round_id: int) -> str:
    """Return the stable object identity for one run/round telemetry record."""
    _require_text(run_id, "run_id")
    if type(round_id) is not int or round_id < 0:
        raise CoreValidationError("round_id must be non-negative")
    return (
        "graph-telemetry-"
        + stable_sha256({"run_id": run_id, "round_id": round_id})[:24]
    )


def graph_telemetry_ref(record: GraphTelemetryRecord) -> ObjectRef:
    """Return one telemetry object's repository reference."""
    if not isinstance(record, GraphTelemetryRecord):
        raise CoreValidationError("record must be GraphTelemetryRecord")
    return ObjectRef(
        GRAPH_TELEMETRY_OBJECT_TYPE,
        graph_telemetry_id(record.run_id, record.round_id),
        record.schema_version,
    )


class GraphTelemetryRecorder:
    """Accumulate optional timings and persist them independently of Graph state."""

    def __init__(self, objects: LocalExperimentRepository, *, run_id: str) -> None:
        """Create a recorder for one run without touching the object repository."""
        if not isinstance(objects, LocalExperimentRepository):
            raise CoreValidationError("objects must be LocalExperimentRepository")
        _require_text(run_id, "run_id")
        self._objects = objects
        self._run_id = run_id
        self._rounds: dict[int, dict[str, object]] = {}

    def record_correlation(self, round_id: int, timing: CorrelationTiming) -> None:
        """Add one correlation engine timing sample to a round."""
        if not isinstance(timing, CorrelationTiming):
            raise CoreValidationError("timing must be CorrelationTiming")
        values = self._round(round_id)
        values["correlation_total_seconds"] = (
            float(values["correlation_total_seconds"]) + timing.total_seconds
        )
        values["exposure_decode_seconds"] = (
            float(values["exposure_decode_seconds"]) + timing.exposure_decode_seconds
        )
        values["pairwise_correlation_seconds"] = (
            float(values["pairwise_correlation_seconds"])
            + timing.pairwise_correlation_seconds
        )

    def record_w_construction(self, round_id: int, elapsed_seconds: float) -> None:
        """Add one W-matrix construction timing sample to a round."""
        _require_non_negative_float(elapsed_seconds, "elapsed_seconds")
        values = self._round(round_id)
        values["w_construction_seconds"] = float(
            values["w_construction_seconds"]
        ) + float(elapsed_seconds)

    def record_flow(
        self,
        round_id: int,
        elapsed_seconds: float,
        result: GraphFlowResult,
    ) -> None:
        """Add one Flow timing and its exact result metadata to a round."""
        _require_non_negative_float(elapsed_seconds, "elapsed_seconds")
        if not isinstance(result, GraphFlowResult):
            raise CoreValidationError("result must be GraphFlowResult")
        values = self._round(round_id)
        values["flow_total_seconds"] = float(values["flow_total_seconds"]) + float(
            elapsed_seconds
        )
        cast_iterations = values["flow_iterations"]
        cast_converged = values["flow_converged"]
        if not isinstance(cast_iterations, list) or not isinstance(
            cast_converged, list
        ):
            raise CoreValidationError("invalid internal telemetry accumulator")
        cast_iterations.append(result.iterations_run)
        cast_converged.append(result.converged)

    def records(self) -> tuple[GraphTelemetryRecord, ...]:
        """Return accumulated records in stable round order."""
        return tuple(
            GraphTelemetryRecord(
                run_id=self._run_id,
                round_id=round_id,
                correlation_total_seconds=float(values["correlation_total_seconds"]),
                exposure_decode_seconds=float(values["exposure_decode_seconds"]),
                pairwise_correlation_seconds=float(
                    values["pairwise_correlation_seconds"]
                ),
                w_construction_seconds=float(values["w_construction_seconds"]),
                flow_total_seconds=float(values["flow_total_seconds"]),
                flow_iterations=tuple(values["flow_iterations"]),  # type: ignore[arg-type]
                flow_converged=tuple(values["flow_converged"]),  # type: ignore[arg-type]
            )
            for round_id, values in sorted(self._rounds.items())
        )

    def persist(self) -> tuple[GraphTelemetryRecord, ...]:
        """Persist records best-effort; telemetry failures never escape to search."""
        persisted: list[GraphTelemetryRecord] = []
        for record in self.records():
            try:
                ref = graph_telemetry_ref(record)
                if not self._objects.exists(ref):
                    self._objects.put_object(
                        GRAPH_TELEMETRY_OBJECT_TYPE,
                        ref.object_id,
                        record,
                    )
                persisted.append(record)
            except Exception:  # noqa: BLE001 - telemetry is observational only
                continue
        return tuple(persisted)

    def _round(self, round_id: int) -> dict[str, object]:
        if type(round_id) is not int or round_id < 0:
            raise CoreValidationError("round_id must be non-negative")
        return self._rounds.setdefault(
            round_id,
            {
                "correlation_total_seconds": 0.0,
                "exposure_decode_seconds": 0.0,
                "pairwise_correlation_seconds": 0.0,
                "w_construction_seconds": 0.0,
                "flow_total_seconds": 0.0,
                "flow_iterations": [],
                "flow_converged": [],
            },
        )


def load_graph_telemetry(
    objects: LocalExperimentRepository,
    *,
    run_id: str,
) -> tuple[GraphTelemetryRecord, ...]:
    """Load persisted telemetry records for one run."""
    if not isinstance(objects, LocalExperimentRepository):
        raise CoreValidationError("objects must be LocalExperimentRepository")
    _require_text(run_id, "run_id")
    records = []
    for object_id in objects.list_object_ids(GRAPH_TELEMETRY_OBJECT_TYPE):
        record = objects.get_object(
            ObjectRef(
                GRAPH_TELEMETRY_OBJECT_TYPE,
                object_id,
                GRAPH_TELEMETRY_SCHEMA_VERSION,
            ),
            GraphTelemetryRecord,
        )
        if record.run_id == run_id:
            records.append(record)
    return tuple(sorted(records, key=lambda item: item.round_id))


def write_graph_telemetry_report(
    run_root: Path,
    records: Iterable[GraphTelemetryRecord],
    *,
    run_id: str,
) -> tuple[Path, Path]:
    """Write derived telemetry CSV and manifest without touching Graph snapshots."""
    _require_text(run_id, "run_id")
    ordered = tuple(
        sorted(
            (record for record in records if record.run_id == run_id),
            key=lambda item: item.round_id,
        )
    )
    report_root = run_root / "graph"
    report_root.mkdir(parents=True, exist_ok=True)
    columns = (
        "run_id",
        "round_id",
        "correlation_total_seconds",
        "exposure_decode_seconds",
        "pairwise_correlation_seconds",
        "w_construction_seconds",
        "flow_total_seconds",
        "flow_iterations",
        "flow_converged",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(
        {
            "run_id": record.run_id,
            "round_id": record.round_id,
            "correlation_total_seconds": record.correlation_total_seconds,
            "exposure_decode_seconds": record.exposure_decode_seconds,
            "pairwise_correlation_seconds": record.pairwise_correlation_seconds,
            "w_construction_seconds": record.w_construction_seconds,
            "flow_total_seconds": record.flow_total_seconds,
            "flow_iterations": "|".join(str(value) for value in record.flow_iterations),
            "flow_converged": "|".join(
                str(value).lower() for value in record.flow_converged
            ),
        }
        for record in ordered
    )
    csv_path = report_root / "telemetry.csv"
    csv_path.write_text(output.getvalue(), encoding="utf-8")
    manifest_path = report_root / "telemetry.json"
    manifest_path.write_text(
        to_json(
            {
                "run_id": run_id,
                "csv_path": "telemetry.csv",
                "record_refs": tuple(graph_telemetry_ref(record) for record in ordered),
                "row_count": len(ordered),
                "schema_version": GRAPH_TELEMETRY_REPORT_SCHEMA_VERSION,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")


def _require_non_negative_float(value: object, field_name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CoreValidationError(f"{field_name} must be finite")
    if float(value) < 0.0:
        raise CoreValidationError(f"{field_name} must be non-negative")


__all__ = [
    "GRAPH_TELEMETRY_OBJECT_TYPE",
    "GRAPH_TELEMETRY_REPORT_SCHEMA_VERSION",
    "GRAPH_TELEMETRY_SCHEMA_VERSION",
    "GraphTelemetryRecord",
    "GraphTelemetryRecorder",
    "graph_telemetry_id",
    "graph_telemetry_ref",
    "load_graph_telemetry",
    "write_graph_telemetry_report",
]
