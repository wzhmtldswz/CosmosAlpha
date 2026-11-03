"""Graph-specific expression equivalence and duplicate indexing."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from cosmosalpha.agents import FactorSpec
from cosmosalpha.core import CoreValidationError, stable_sha256
from cosmosalpha.factors import CallNode, ExpressionNode, parse_expression
from cosmosalpha.factors.expression import normalize_expression

GRAPH_DUPLICATE_KEY_SCHEMA_VERSION = "graph-duplicate-key/1.0"
GRAPH_LIBRARY_KEY_SNAPSHOT_SCHEMA_VERSION = "graph-library-key-snapshot/1.0"
COMMUTATIVE_OPERATOR_NAMES = frozenset({"add", "mul", "corr", "cov"})
_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be a non-empty string without surrounding whitespace"
        )


def _require_schema_version(value: object, field_name: str) -> None:
    _require_text(value, field_name)
    if _SCHEMA_VERSION_PATTERN.fullmatch(value) is None:
        raise CoreValidationError(
            f"{field_name} must match '<schema-name>/<major>.<minor>'"
        )


def canonical_graph_expression(node: ExpressionNode) -> str:
    """Canonicalize only commutative operands for Graph duplicate identity."""
    if not isinstance(node, (CallNode,)):
        return normalize_expression(node)

    arguments = tuple(
        canonical_graph_expression(argument) for argument in node.arguments
    )
    if node.function in COMMUTATIVE_OPERATOR_NAMES & {"add", "mul"}:
        if len(arguments) != 2:
            raise CoreValidationError(
                f"validated {node.function} expression must have two operands"
            )
        arguments = tuple(sorted(arguments))
    elif node.function in COMMUTATIVE_OPERATOR_NAMES & {"corr", "cov"}:
        if len(arguments) != 3:
            raise CoreValidationError(
                f"validated {node.function} expression must have two operands and a window"
            )
        arguments = (*tuple(sorted(arguments[:2])), arguments[2])
    return f"{node.function}({', '.join(arguments)})"


@dataclass(frozen=True, slots=True)
class GraphDuplicateKey:
    """Versioned expression equivalence key independent of global factor_id."""

    canonical_expression: str
    frequency: str
    operator_set_version: str
    field_schema_version: str
    expression_schema_version: str
    schema_version: str = GRAPH_DUPLICATE_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all identity dimensions used by Graph duplicate checking."""
        for field_name in (
            "canonical_expression",
            "frequency",
            "operator_set_version",
            "field_schema_version",
            "expression_schema_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        for field_name in (
            "operator_set_version",
            "field_schema_version",
            "expression_schema_version",
            "schema_version",
        ):
            _require_schema_version(getattr(self, field_name), field_name)
        if self.schema_version != GRAPH_DUPLICATE_KEY_SCHEMA_VERSION:
            raise CoreValidationError("invalid GraphDuplicateKey schema_version")

    @classmethod
    def from_validated_ast(
        cls,
        node: ExpressionNode,
        *,
        frequency: str,
        operator_set_version: str,
        field_schema_version: str,
        expression_schema_version: str,
    ) -> GraphDuplicateKey:
        """Build a key from an AST already accepted by factor validation."""
        return cls(
            canonical_expression=canonical_graph_expression(node),
            frequency=frequency,
            operator_set_version=operator_set_version,
            field_schema_version=field_schema_version,
            expression_schema_version=expression_schema_version,
        )

    @classmethod
    def from_factor_spec(cls, factor_spec: FactorSpec) -> GraphDuplicateKey:
        """Build a key from the authoritative validated FactorSpec."""
        if not isinstance(factor_spec, FactorSpec):
            raise CoreValidationError("factor_spec must be FactorSpec")
        return cls.from_validated_ast(
            parse_expression(factor_spec.expression),
            frequency=factor_spec.frequency,
            operator_set_version=factor_spec.operator_set_version,
            field_schema_version=factor_spec.field_schema_version,
            expression_schema_version=factor_spec.expression_schema_version,
        )

    @property
    def digest(self) -> str:
        """Return a stable digest suitable for node expression_hash fields."""
        return stable_sha256(
            {
                "canonical_expression": self.canonical_expression,
                "frequency": self.frequency,
                "operator_set_version": self.operator_set_version,
                "field_schema_version": self.field_schema_version,
                "expression_schema_version": self.expression_schema_version,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class FactorLibraryCanonicalKeySnapshot:
    """Immutable run-start view of Factor Library duplicate identities."""

    snapshot_id: str
    source_revision: str
    keys: tuple[GraphDuplicateKey, ...]
    schema_version: str = GRAPH_LIBRARY_KEY_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and deterministically order the external identity index."""
        _require_text(self.snapshot_id, "snapshot_id")
        _require_text(self.source_revision, "source_revision")
        if not isinstance(self.keys, tuple) or not all(
            isinstance(key, GraphDuplicateKey) for key in self.keys
        ):
            raise CoreValidationError("keys must be a tuple of GraphDuplicateKey")
        if len(self.keys) != len(set(self.keys)):
            raise CoreValidationError("snapshot keys must be unique")
        if self.schema_version != GRAPH_LIBRARY_KEY_SNAPSHOT_SCHEMA_VERSION:
            raise CoreValidationError(
                "invalid FactorLibraryCanonicalKeySnapshot schema_version"
            )
        ordered = tuple(sorted(self.keys, key=lambda key: key.digest))
        expected_id = stable_sha256(
            {
                "source_revision": self.source_revision,
                "keys": ordered,
                "schema_version": self.schema_version,
            }
        )
        if self.snapshot_id != expected_id:
            raise CoreValidationError("snapshot_id does not match snapshot content")
        object.__setattr__(self, "keys", ordered)

    @classmethod
    def from_factor_specs(
        cls,
        factor_specs: Iterable[FactorSpec],
        *,
        source_revision: str,
    ) -> FactorLibraryCanonicalKeySnapshot:
        """Build a stable snapshot from persisted, validated FactorSpecs."""
        values = tuple(factor_specs)
        if not all(isinstance(spec, FactorSpec) for spec in values):
            raise CoreValidationError("factor_specs must contain FactorSpec values")
        keys = tuple(GraphDuplicateKey.from_factor_spec(spec) for spec in values)
        ordered = tuple(sorted(set(keys), key=lambda key: key.digest))
        snapshot_id = stable_sha256(
            {
                "source_revision": source_revision,
                "keys": ordered,
                "schema_version": GRAPH_LIBRARY_KEY_SNAPSHOT_SCHEMA_VERSION,
            }
        )
        return cls(snapshot_id, source_revision, ordered)


@dataclass(frozen=True, slots=True)
class DuplicateIndexEntry:
    """One persisted Graph duplicate key to node mapping."""

    key: GraphDuplicateKey
    node_id: str
    schema_version: str = "duplicate-index-entry/1.0"

    def __post_init__(self) -> None:
        """Validate a duplicate index mapping."""
        if not isinstance(self.key, GraphDuplicateKey):
            raise CoreValidationError("key must be GraphDuplicateKey")
        _require_text(self.node_id, "node_id")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != "duplicate-index-entry/1.0":
            raise CoreValidationError("invalid DuplicateIndexEntry schema_version")


@dataclass(frozen=True, slots=True)
class DuplicateIndexSnapshot:
    """Stable JSON representation of the mutable runtime duplicate index."""

    entries: tuple[DuplicateIndexEntry, ...]
    invalid_keys: tuple[GraphDuplicateKey, ...]
    external_keys: tuple[GraphDuplicateKey, ...] = ()
    external_snapshot_id: str = ""
    schema_version: str = "duplicate-index-snapshot/1.0"

    def __post_init__(self) -> None:
        """Validate and deterministically order persisted index state."""
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, DuplicateIndexEntry) for entry in self.entries
        ):
            raise CoreValidationError("entries must be a tuple of DuplicateIndexEntry")
        if not isinstance(self.invalid_keys, tuple) or not all(
            isinstance(key, GraphDuplicateKey) for key in self.invalid_keys
        ):
            raise CoreValidationError(
                "invalid_keys must be a tuple of GraphDuplicateKey"
            )
        entry_keys = tuple(entry.key for entry in self.entries)
        if len(entry_keys) != len(set(entry_keys)):
            raise CoreValidationError("duplicate index entries must have unique keys")
        node_ids = tuple(entry.node_id for entry in self.entries)
        if len(node_ids) != len(set(node_ids)):
            raise CoreValidationError(
                "duplicate index entries must have unique node ids"
            )
        if len(self.invalid_keys) != len(set(self.invalid_keys)):
            raise CoreValidationError("invalid_keys must be unique")
        if not isinstance(self.external_keys, tuple) or not all(
            isinstance(key, GraphDuplicateKey) for key in self.external_keys
        ):
            raise CoreValidationError(
                "external_keys must be a tuple of GraphDuplicateKey"
            )
        if len(self.external_keys) != len(set(self.external_keys)):
            raise CoreValidationError("external_keys must be unique")
        if type(self.external_snapshot_id) is not str:
            raise CoreValidationError("external_snapshot_id must be text")
        if set(entry_keys) & set(self.invalid_keys):
            raise CoreValidationError("a key cannot be both indexed and invalid")
        if set(entry_keys) & set(self.external_keys):
            raise CoreValidationError("a key cannot be both indexed and external")
        sorted_entries = tuple(sorted(self.entries, key=lambda entry: entry.key.digest))
        sorted_invalid = tuple(sorted(self.invalid_keys, key=lambda key: key.digest))
        sorted_external = tuple(sorted(self.external_keys, key=lambda key: key.digest))
        object.__setattr__(self, "entries", sorted_entries)
        object.__setattr__(self, "invalid_keys", sorted_invalid)
        object.__setattr__(self, "external_keys", sorted_external)
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != "duplicate-index-snapshot/1.0":
            raise CoreValidationError("invalid DuplicateIndexSnapshot schema_version")


class DuplicateIndex:
    """Mutable in-memory index with an immutable persistence snapshot."""

    def __init__(
        self,
        *,
        entries: Iterable[DuplicateIndexEntry] = (),
        invalid_keys: Iterable[GraphDuplicateKey] = (),
        external_keys: Iterable[GraphDuplicateKey] = (),
        external_snapshot_id: str = "",
    ) -> None:
        """Create an index from optional persisted mappings."""
        snapshot = DuplicateIndexSnapshot(
            tuple(entries),
            tuple(invalid_keys),
            tuple(external_keys),
            external_snapshot_id,
        )
        self._node_by_key = {entry.key: entry.node_id for entry in snapshot.entries}
        self._invalid_keys = set(snapshot.invalid_keys)
        self._external_keys = set(snapshot.external_keys)
        self._external_snapshot_id = snapshot.external_snapshot_id

    @classmethod
    def from_snapshot(cls, snapshot: DuplicateIndexSnapshot) -> DuplicateIndex:
        """Restore a runtime index from a validated snapshot."""
        if not isinstance(snapshot, DuplicateIndexSnapshot):
            raise CoreValidationError("snapshot must be DuplicateIndexSnapshot")
        return cls(
            entries=snapshot.entries,
            invalid_keys=snapshot.invalid_keys,
            external_keys=snapshot.external_keys,
            external_snapshot_id=snapshot.external_snapshot_id,
        )

    def lookup(self, key: GraphDuplicateKey) -> str | None:
        """Return the existing node id for a key, if any."""
        self._require_key(key)
        return self._node_by_key.get(key)

    def contains(self, key: GraphDuplicateKey) -> bool:
        """Return whether a key is indexed or blocked as invalid."""
        self._require_key(key)
        return (
            key in self._node_by_key
            or key in self._invalid_keys
            or key in self._external_keys
        )

    def is_invalid(self, key: GraphDuplicateKey) -> bool:
        """Return whether a key is blocked by this run's invalid-expression set."""
        self._require_key(key)
        return key in self._invalid_keys

    def add_node(self, key: GraphDuplicateKey, node_id: str) -> None:
        """Add an idempotent key-to-node mapping."""
        self._require_key(key)
        _require_text(node_id, "node_id")
        if key in self._invalid_keys:
            raise CoreValidationError("invalid expression key cannot become a node")
        if key in self._external_keys:
            raise CoreValidationError(
                "external Factor Library key cannot become a node"
            )
        existing = self._node_by_key.get(key)
        if existing is not None and existing != node_id:
            raise CoreValidationError("GraphDuplicateKey already maps to another node")
        if node_id in self._node_by_key.values() and existing != node_id:
            raise CoreValidationError("node_id already maps to another duplicate key")
        self._node_by_key[key] = node_id

    def mark_invalid(self, key: GraphDuplicateKey) -> None:
        """Block a validated-but-ineligible expression from future work."""
        self._require_key(key)
        if key in self._node_by_key:
            raise CoreValidationError("indexed node key cannot be marked invalid")
        self._invalid_keys.add(key)

    def snapshot(self) -> DuplicateIndexSnapshot:
        """Return a deterministic, JSON-serializable index snapshot."""
        return DuplicateIndexSnapshot(
            entries=tuple(
                DuplicateIndexEntry(key=key, node_id=node_id)
                for key, node_id in self._node_by_key.items()
            ),
            invalid_keys=tuple(self._invalid_keys),
            external_keys=tuple(self._external_keys),
            external_snapshot_id=self._external_snapshot_id,
        )

    def __len__(self) -> int:
        """Return the number of indexed Graph nodes."""
        return len(self._node_by_key)

    @staticmethod
    def _require_key(key: GraphDuplicateKey) -> None:
        if not isinstance(key, GraphDuplicateKey):
            raise CoreValidationError("key must be GraphDuplicateKey")


__all__ = [
    "COMMUTATIVE_OPERATOR_NAMES",
    "DuplicateIndex",
    "DuplicateIndexEntry",
    "DuplicateIndexSnapshot",
    "GraphDuplicateKey",
    "FactorLibraryCanonicalKeySnapshot",
    "GRAPH_LIBRARY_KEY_SNAPSHOT_SCHEMA_VERSION",
    "canonical_graph_expression",
]
