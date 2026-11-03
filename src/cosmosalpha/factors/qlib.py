"""Compile validated CosmosAlpha factor expressions for registered Qlib operators."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cosmosalpha.agents import FactorSpec
from cosmosalpha.core import CoreValidationError
from cosmosalpha.factors.expression import (
    BASE_OPERATOR_SET_VERSION,
    EXPRESSION_SCHEMA_VERSION,
    CallNode,
    ExpressionNode,
    FieldCatalog,
    FieldNode,
    NumberNode,
    OperatorCatalog,
    build_base_field_catalog,
    build_base_operator_catalog,
    normalize_expression,
    parse_expression,
    validate_expression,
)

QLIB_COMPILER_VERSION = "qlib-expression-compiler/1.0"
COMPILED_EXPRESSION_SCHEMA_VERSION = "compiled-expression/1.0"
_QLIB_FUNCTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class QlibCompilationError(ValueError):
    """Raised when a factor cannot be compiled to the registered Qlib backend."""


@dataclass(frozen=True, slots=True)
class CompiledExpression:
    """Versioned Qlib representation derived from one validated factor spec."""

    factor_id: str
    source_expression: str
    target_backend: str
    compiled_expression: str
    compiler_version: str
    operator_registry_version: str
    schema_version: str = COMPILED_EXPRESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate derived-expression identity and version fields."""
        for name in (
            "factor_id",
            "source_expression",
            "target_backend",
            "compiled_expression",
            "compiler_version",
            "operator_registry_version",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise CoreValidationError(f"{name} must be a non-empty string")
        if self.target_backend != "qlib":
            raise CoreValidationError("target_backend must be 'qlib'")
        if self.schema_version != COMPILED_EXPRESSION_SCHEMA_VERSION:
            raise CoreValidationError(
                f"schema_version must be {COMPILED_EXPRESSION_SCHEMA_VERSION!r}"
            )


@dataclass(frozen=True, slots=True)
class QlibOperatorBinding:
    """Bind one catalog compiler alias to a registered Qlib function name."""

    compiler_alias: str
    qlib_function: str

    def __post_init__(self) -> None:
        """Validate a deterministic Qlib function binding."""
        for name in ("compiler_alias", "qlib_function"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise CoreValidationError(f"{name} must be a non-empty string")
        if _QLIB_FUNCTION_PATTERN.fullmatch(self.qlib_function) is None:
            raise CoreValidationError("qlib_function must be a Qlib identifier")


@dataclass(frozen=True, slots=True)
class QlibOperatorRegistry:
    """Complete registry of Qlib function bindings for one operator catalog."""

    bindings: tuple[QlibOperatorBinding, ...]
    version: str = "qlib-operator-registry/1.0"

    def __post_init__(self) -> None:
        """Validate unique compiler aliases and registry version."""
        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise CoreValidationError("bindings must be a non-empty tuple")
        if not all(
            isinstance(binding, QlibOperatorBinding) for binding in self.bindings
        ):
            raise CoreValidationError(
                "bindings must contain QlibOperatorBinding values"
            )
        aliases = tuple(binding.compiler_alias for binding in self.bindings)
        if len(set(aliases)) != len(aliases):
            raise CoreValidationError("Qlib compiler aliases must be unique")
        if type(self.version) is not str or not self.version:
            raise CoreValidationError("version must be a non-empty string")

    def get(self, compiler_alias: str) -> QlibOperatorBinding | None:
        """Return a binding for one catalog compiler alias."""
        for binding in self.bindings:
            if binding.compiler_alias == compiler_alias:
                return binding
        return None


def build_base_qlib_operator_registry() -> QlibOperatorRegistry:
    """Build bindings for custom operators preserving CosmosAlpha semantics.

    Every operator uses an explicitly registered custom Qlib function.  This
    prevents a protected operation or rolling minimum-period rule from being
    silently replaced by a superficially similar Qlib native operator.
    """
    catalog = build_base_operator_catalog()
    return QlibOperatorRegistry(
        bindings=tuple(
            QlibOperatorBinding(
                compiler_alias=operator.qlib_compiler_alias,
                qlib_function="CosmosAlpha"
                + "".join(part.capitalize() for part in operator.name.split("_")),
            )
            for operator in catalog.operators
        )
    )


class QlibExpressionCompiler:
    """Compile only fully validated factor specs into Qlib formula text."""

    def __init__(
        self,
        *,
        field_catalog: FieldCatalog | None = None,
        operator_catalog: OperatorCatalog | None = None,
        operator_registry: QlibOperatorRegistry | None = None,
    ) -> None:
        """Create a compiler and verify complete backend binding coverage."""
        self._fields = field_catalog or build_base_field_catalog()
        self._operators = operator_catalog or build_base_operator_catalog()
        self._registry = operator_registry or build_base_qlib_operator_registry()
        self._validate_backend_coverage()

    def compile(self, factor_spec: FactorSpec) -> CompiledExpression:
        """Compile one validated factor specification into Qlib expression text.

        Args:
            factor_spec: Authoritative factor definition produced by validation.

        Returns:
            Versioned Qlib expression using the registered custom operators.

        Raises:
            QlibCompilationError: If the spec cannot be revalidated or a
                required Qlib operator binding is unavailable.
        """
        if not isinstance(factor_spec, FactorSpec):
            raise QlibCompilationError("UNVALIDATED_FACTOR_SPEC: expected FactorSpec")
        self._require_revalidated_spec(factor_spec)
        node = parse_expression(factor_spec.expression)
        return CompiledExpression(
            factor_id=factor_spec.factor_id,
            source_expression=factor_spec.expression,
            target_backend="qlib",
            compiled_expression=self._compile_node(node),
            compiler_version=QLIB_COMPILER_VERSION,
            operator_registry_version=self._registry.version,
        )

    def _validate_backend_coverage(self) -> None:
        missing = [
            operator.qlib_compiler_alias
            for operator in self._operators.operators
            if self._registry.get(operator.qlib_compiler_alias) is None
        ]
        if missing:
            raise QlibCompilationError(
                "UNSUPPORTED_BACKEND: missing Qlib bindings for " + ", ".join(missing)
            )

    def _require_revalidated_spec(self, factor_spec: FactorSpec) -> None:
        if factor_spec.operator_set_version != self._operators.version:
            raise QlibCompilationError(
                "UNVALIDATED_FACTOR_SPEC: operator catalog differs"
            )
        if factor_spec.field_schema_version != self._fields.schema_version:
            raise QlibCompilationError("UNVALIDATED_FACTOR_SPEC: field catalog differs")
        if factor_spec.expression_schema_version != EXPRESSION_SCHEMA_VERSION:
            raise QlibCompilationError(
                "UNVALIDATED_FACTOR_SPEC: expression schema differs"
            )
        report = validate_expression(
            factor_spec.expression,
            self._fields,
            self._operators,
            proposal_id=factor_spec.proposal_id,
            frequency=factor_spec.frequency,
        )
        if (
            not report.is_valid
            or report.factor_id != factor_spec.factor_id
            or report.normalized_expression != factor_spec.expression
            or report.required_fields != factor_spec.required_fields
        ):
            raise QlibCompilationError(
                "UNVALIDATED_FACTOR_SPEC: factor spec does not match validation"
            )

    def _compile_node(self, node: ExpressionNode) -> str:
        if isinstance(node, FieldNode):
            return f"${node.name}"
        if isinstance(node, NumberNode):
            return normalize_expression(node)
        if isinstance(node, CallNode):
            operator = self._operators.get(node.function)
            if operator is None:
                raise QlibCompilationError(
                    f"UNSUPPORTED_BACKEND: unregistered operator {node.function}"
                )
            binding = self._registry.get(operator.qlib_compiler_alias)
            if binding is None:
                raise QlibCompilationError(
                    f"UNSUPPORTED_BACKEND: no Qlib binding for {node.function}"
                )
            arguments = ", ".join(self._compile_node(arg) for arg in node.arguments)
            return f"{binding.qlib_function}({arguments})"
        raise QlibCompilationError("UNSUPPORTED_BACKEND: unsupported AST node")


def base_qlib_compiler() -> QlibExpressionCompiler:
    """Return a compiler for the complete base Qlib operator backend."""
    return QlibExpressionCompiler()


def base_operator_registry_version() -> str:
    """Return the base Qlib operator registry version for startup diagnostics."""
    if BASE_OPERATOR_SET_VERSION != build_base_operator_catalog().version:
        raise QlibCompilationError(
            "UNSUPPORTED_BACKEND: unexpected base operator version"
        )
    return build_base_qlib_operator_registry().version
