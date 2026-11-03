"""Canonical expression structure facts and local structure neighborhoods."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cosmosalpha.core import CoreValidationError
from cosmosalpha.factors import (
    CallNode,
    ExpressionNode,
    FieldNode,
    NumberNode,
    normalize_expression,
    parse_expression,
)
from cosmosalpha.pipeline.graph.core import DenseMatrix

STRUCTURE_CANONICAL_FORM_SCHEMA_VERSION = "structure-canonical-form/1.0"
STRUCTURE_FACT_STORE_SCHEMA_VERSION = "structure-fact-store/1.0"
STRUCTURE_COMMUTATIVE_OPERATOR_NAMES = frozenset({"add", "mul", "corr", "cov"})
_WINDOWED_OPERATOR_NAMES = frozenset(
    {
        "delay",
        "delta",
        "corr",
        "cov",
        "decay_linear",
        "ts_min",
        "ts_max",
        "ts_argmin",
        "ts_argmax",
        "ts_rank",
        "ts_sum",
        "ts_stddev",
        "ts_zscore",
    }
)

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


def _canonical_expression(node: ExpressionNode) -> str:
    if isinstance(node, (FieldNode, NumberNode)):
        return normalize_expression(node)
    if not isinstance(node, CallNode):
        raise CoreValidationError("unsupported expression AST node")
    arguments = tuple(_canonical_expression(argument) for argument in node.arguments)
    if node.function in STRUCTURE_COMMUTATIVE_OPERATOR_NAMES:
        if node.function in {"add", "mul"}:
            if len(arguments) != 2:
                raise CoreValidationError(
                    f"validated {node.function} expression must have two operands"
                )
            arguments = tuple(sorted(arguments))
        elif len(arguments) != 3:
            raise CoreValidationError(
                f"validated {node.function} expression must have two operands "
                "and a window"
            )
        else:
            arguments = (*tuple(sorted(arguments[:2])), arguments[2])
    return f"{node.function}({', '.join(arguments)})"


def _validate_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or not all(
        type(item) is str and item for item in value
    ):
        raise CoreValidationError(f"{field_name} must be a tuple of strings")


def _validate_matrix(value: object, field_name: str, *, size: int) -> None:
    if not isinstance(value, tuple) or len(value) != size:
        raise CoreValidationError(f"{field_name} must be a {size}x{size} matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, tuple) or len(row) != size:
            raise CoreValidationError(
                f"{field_name}[{row_index}] must contain {size} values"
            )
        for column_index, item in enumerate(row):
            if type(item) is not float or not math.isfinite(item):
                raise CoreValidationError(
                    f"{field_name}[{row_index}][{column_index}] must be finite float"
                )


@dataclass(frozen=True, slots=True)
class StructureCanonicalForm:
    """Canonical structure features independent of Graph factor identity."""

    canonical_expression: str
    subtree_features: tuple[str, ...]
    operator_features: tuple[str, ...]
    field_features: tuple[str, ...]
    window_features: tuple[int, ...]
    schema_version: str = STRUCTURE_CANONICAL_FORM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate deterministic, sorted structure feature tuples."""
        _require_text(self.canonical_expression, "canonical_expression")
        for field_name, value in (
            ("subtree_features", self.subtree_features),
            ("operator_features", self.operator_features),
            ("field_features", self.field_features),
        ):
            _validate_string_tuple(value, field_name)
            if value != tuple(sorted(set(value))):
                raise CoreValidationError(f"{field_name} must be unique and sorted")
        if not isinstance(self.window_features, tuple) or not all(
            type(window) is int and window > 0 for window in self.window_features
        ):
            raise CoreValidationError("window_features must contain positive integers")
        if self.window_features != tuple(sorted(set(self.window_features))):
            raise CoreValidationError("window_features must be unique and sorted")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != STRUCTURE_CANONICAL_FORM_SCHEMA_VERSION:
            raise CoreValidationError("invalid StructureCanonicalForm schema_version")

    @classmethod
    def from_expression(
        cls, expression: str | ExpressionNode
    ) -> StructureCanonicalForm:
        """Derive canonical subtree, operator, field and window features."""
        node = (
            parse_expression(expression) if isinstance(expression, str) else expression
        )
        if not isinstance(node, (FieldNode, NumberNode, CallNode)):
            raise CoreValidationError("expression must be a supported AST node")
        subtrees: set[str] = set()
        operators: set[str] = set()
        fields: set[str] = set()
        windows: set[int] = set()

        def visit(current: ExpressionNode) -> None:
            subtrees.add(_canonical_expression(current))
            if isinstance(current, FieldNode):
                fields.add(current.name)
                return
            if isinstance(current, NumberNode):
                return
            operators.add(current.function)
            for argument in current.arguments:
                visit(argument)
            last_argument = current.arguments[-1] if current.arguments else None
            if (
                current.function in _WINDOWED_OPERATOR_NAMES
                and isinstance(last_argument, NumberNode)
                and type(last_argument.value) is int
            ):
                windows.add(last_argument.value)

        visit(node)
        return cls(
            canonical_expression=_canonical_expression(node),
            subtree_features=tuple(sorted(subtrees)),
            operator_features=tuple(sorted(operators)),
            field_features=tuple(sorted(fields)),
            window_features=tuple(sorted(windows)),
        )


@dataclass(frozen=True, slots=True)
class StructureFactStore:
    """Complete dense structure similarity and distance facts."""

    node_ids: tuple[str, ...]
    similarity: DenseMatrix
    distance: DenseMatrix
    schema_version: str = STRUCTURE_FACT_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate dense fact matrices without applying neighborhood pruning."""
        if not isinstance(self.node_ids, tuple) or not all(
            type(node_id) is str and node_id for node_id in self.node_ids
        ):
            raise CoreValidationError("node_ids must be a tuple of non-empty strings")
        if self.node_ids != tuple(sorted(self.node_ids)):
            raise CoreValidationError("node_ids must be sorted deterministically")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise CoreValidationError("node_ids must be unique")
        size = len(self.node_ids)
        _validate_matrix(self.similarity, "similarity", size=size)
        _validate_matrix(self.distance, "distance", size=size)
        for left in range(size):
            if self.similarity[left][left] != 1.0:
                raise CoreValidationError("similarity diagonal must be 1.0")
            if self.distance[left][left] != 0.0:
                raise CoreValidationError("distance diagonal must be 0.0")
            for right in range(size):
                if not 0.0 <= self.similarity[left][right] <= 1.0:
                    raise CoreValidationError("similarity values must be in [0, 1]")
                if not 0.0 <= self.distance[left][right] <= 1.0:
                    raise CoreValidationError("distance values must be in [0, 1]")
                if not math.isclose(
                    self.distance[left][right],
                    1.0 - self.similarity[left][right],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise CoreValidationError("distance must equal 1 - similarity")
                if not math.isclose(
                    self.similarity[left][right],
                    self.similarity[right][left],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise CoreValidationError("similarity must be symmetric")
                if not math.isclose(
                    self.distance[left][right],
                    self.distance[right][left],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise CoreValidationError("distance must be symmetric")
        _require_schema_version(self.schema_version, "schema_version")
        if self.schema_version != STRUCTURE_FACT_STORE_SCHEMA_VERSION:
            raise CoreValidationError("invalid StructureFactStore schema_version")


def structure_jaccard_similarity(
    left: StructureCanonicalForm,
    right: StructureCanonicalForm,
) -> float:
    """Return Jaccard similarity over canonical subexpression sets."""
    if not isinstance(left, StructureCanonicalForm) or not isinstance(
        right, StructureCanonicalForm
    ):
        raise CoreValidationError("structure forms must be StructureCanonicalForm")
    left_subtrees = set(left.subtree_features)
    right_subtrees = set(right.subtree_features)
    union = left_subtrees | right_subtrees
    if not union:
        return 1.0
    return len(left_subtrees & right_subtrees) / len(union)


def structure_jaccard_distance(
    left: StructureCanonicalForm,
    right: StructureCanonicalForm,
) -> float:
    """Return one minus canonical subtree Jaccard similarity."""
    return 1.0 - structure_jaccard_similarity(left, right)


def build_structure_fact_store(
    forms: Mapping[str, StructureCanonicalForm],
) -> StructureFactStore:
    """Build complete dense structure facts in sorted node-id order."""
    if not isinstance(forms, Mapping):
        raise CoreValidationError("forms must be a mapping")
    form_keys = tuple(forms.keys())
    if not all(type(node_id) is str and node_id for node_id in form_keys):
        raise CoreValidationError("forms keys must be non-empty strings")
    node_ids = tuple(sorted(form_keys))
    if not all(
        isinstance(forms[node_id], StructureCanonicalForm) for node_id in node_ids
    ):
        raise CoreValidationError("forms values must be StructureCanonicalForm")
    size = len(node_ids)
    similarity = [[0.0 for _column in range(size)] for _row in range(size)]
    distance = [[0.0 for _column in range(size)] for _row in range(size)]
    for left in range(size):
        for right in range(left, size):
            score = structure_jaccard_similarity(
                forms[node_ids[left]], forms[node_ids[right]]
            )
            similarity[left][right] = score
            similarity[right][left] = score
            distance[left][right] = 1.0 - score
            distance[right][left] = 1.0 - score
    return StructureFactStore(
        node_ids=node_ids,
        similarity=tuple(tuple(row) for row in similarity),
        distance=tuple(tuple(row) for row in distance),
    )


def project_structure_neighborhood(
    facts: StructureFactStore,
    *,
    edge_threshold: float,
    max_neighbors: int,
) -> DenseMatrix:
    """Project full structure facts to thresholded union-symmetrized Top-K W."""
    if not isinstance(facts, StructureFactStore):
        raise CoreValidationError("facts must be StructureFactStore")
    if (
        type(edge_threshold) not in (int, float)
        or not math.isfinite(float(edge_threshold))
        or not 0.0 <= float(edge_threshold) <= 1.0
    ):
        raise CoreValidationError("edge_threshold must be between 0 and 1")
    if type(max_neighbors) is not int or max_neighbors < 1:
        raise CoreValidationError("max_neighbors must be at least 1")
    size = len(facts.node_ids)
    selected = [[False for _column in range(size)] for _row in range(size)]
    for left in range(size):
        candidates = [
            right
            for right in range(size)
            if right != left and facts.similarity[left][right] >= edge_threshold
        ]
        candidates.sort(
            key=lambda right: (-facts.similarity[left][right], facts.node_ids[right])
        )
        for right in candidates[:max_neighbors]:
            selected[left][right] = True
    weights = [[0.0 for _column in range(size)] for _row in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            if selected[left][right] or selected[right][left]:
                weights[left][right] = facts.similarity[left][right]
                weights[right][left] = facts.similarity[left][right]
    return tuple(tuple(row) for row in weights)


__all__ = [
    "STRUCTURE_CANONICAL_FORM_SCHEMA_VERSION",
    "STRUCTURE_COMMUTATIVE_OPERATOR_NAMES",
    "STRUCTURE_FACT_STORE_SCHEMA_VERSION",
    "StructureCanonicalForm",
    "StructureFactStore",
    "build_structure_fact_store",
    "project_structure_neighborhood",
    "structure_jaccard_distance",
    "structure_jaccard_similarity",
]
