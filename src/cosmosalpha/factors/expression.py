"""Expression catalogs, parser, normalization, validation, and stable IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from cosmosalpha.agents import (
    FactorDesignDraft,
    FactorProposal,
    FactorSpec,
    FactorValidationResult,
    FactorVariableDraft,
    ValidationIssue,
    ValidationReport,
    WindowChoiceDraft,
)
from cosmosalpha.core import CoreValidationError, stable_sha256

EXPRESSION_SCHEMA_VERSION = "factor-expression/1.0"
BASE_OPERATOR_SET_VERSION = "operator-set-base/1.0"
BASE_FIELD_SCHEMA_VERSION = "field-catalog/1.0"
OPERATOR_CATALOG_SCHEMA_VERSION = "operator-catalog/1.0"
DEFAULT_TIME_WINDOWS: tuple[int, ...] = (5, 10, 20)
BASE_EXPRESSION_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
DEFAULT_OPTIONAL_EXPRESSION_FIELDS: tuple[str, ...] = (
    "vwap",
    "turn",
    "free_turn",
    "returns",
)
DEFAULT_CONTEXT_FIELDS: tuple[str, ...] = (
    "industry_id",
    "log_mkt_cap",
    "ret20",
    "turn20",
    "vol20",
)
_FIELD_DESCRIPTIONS = {
    "open": "Post-adjusted opening price for the trading day.",
    "high": "Post-adjusted highest price for the trading day.",
    "low": "Post-adjusted lowest price for the trading day.",
    "close": "Post-adjusted closing price for the trading day.",
    "volume": "Trading volume for the trading day.",
    "vwap": "Post-adjusted volume-weighted average price for the trading day.",
    "turn": "Trading-day turnover rate relative to total shares outstanding.",
    "free_turn": "Trading-day turnover rate relative to free-float shares.",
    "returns": "Daily return supplied by the configured market data source.",
    "industry_id": "Source-versioned industry classification for same-day grouping; historical availability depends on the source, and missing industries are excluded.",
    "log_mkt_cap": "Point-in-time natural logarithm of market capitalization.",
    "ret20": "Trailing 20-trading-day return measured through the current day.",
    "turn20": "Trailing 20-trading-day mean turnover measured through the current day.",
    "vol20": (
        "Trailing 20-trading-day sample standard deviation of daily returns "
        "measured through the current day."
    ),
}

_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_SCHEMA_VERSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]*/(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)


class ExpressionSyntaxError(ValueError):
    """Raised when expression text does not match the supported grammar."""


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


def _require_schema_version(value: object, field_name: str) -> str:
    version = _require_string(value, field_name)
    if _SCHEMA_VERSION_PATTERN.fullmatch(version) is None:
        raise CoreValidationError(
            f"{field_name} must match '<schema-name>/<major>.<minor>'"
        )
    return version


def _require_tuple_of_strings(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise CoreValidationError(f"{field_name} must be a tuple")
    for index, item in enumerate(value):
        _require_string(item, f"{field_name}[{index}]")


class OperatorScope(StrEnum):
    """Execution scope for an expression operator."""

    ELEMENTWISE = "elementwise"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    INDUSTRY_GROUP = "industry_group"


class OperatorParameterKind(StrEnum):
    """Kinds of parameters accepted by expression operators."""

    EXPRESSION = "expression"
    WINDOW = "window"


@dataclass(frozen=True, slots=True)
class OperatorParameterSpec:
    """One named parameter in an operator's canonical DSL signature."""

    name: str
    kind: OperatorParameterKind

    def __post_init__(self) -> None:
        """Validate parameter metadata."""
        _require_identifier(self.name, "parameter name")
        if not isinstance(self.kind, OperatorParameterKind):
            raise CoreValidationError("kind must be OperatorParameterKind")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Field available to factor expressions or operator context."""

    name: str
    description: str
    available: bool = True
    context_only: bool = False

    def __post_init__(self) -> None:
        """Validate field metadata."""
        _require_identifier(self.name, "field name")
        _require_string(self.description, "description")
        if type(self.available) is not bool:
            raise CoreValidationError("available must be a boolean")
        if type(self.context_only) is not bool:
            raise CoreValidationError("context_only must be a boolean")


@dataclass(frozen=True, slots=True)
class FieldCatalog:
    """Catalog of expression fields and context-only fields."""

    fields: tuple[FieldSpec, ...]
    schema_version: str = BASE_FIELD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate field catalog consistency."""
        if not isinstance(self.fields, tuple):
            raise CoreValidationError("fields must be a tuple")
        names: set[str] = set()
        for field in self.fields:
            if not isinstance(field, FieldSpec):
                raise CoreValidationError("every field must be FieldSpec")
            if field.name in names:
                raise CoreValidationError(f"duplicate field: {field.name!r}")
            names.add(field.name)
        _require_schema_version(self.schema_version, "schema_version")

    def has_expression_field(self, name: str) -> bool:
        """Return whether a name is available for direct expression use."""
        field = self.get(name)
        return field is not None and field.available and not field.context_only

    def has_context_field(self, name: str) -> bool:
        """Return whether a name is available as operator context."""
        field = self.get(name)
        return field is not None and field.available

    def get(self, name: str) -> FieldSpec | None:
        """Return field metadata by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Metadata for a supported expression operator."""

    name: str
    arity: int
    description: str
    parameter_schema: tuple[OperatorParameterSpec, ...]
    scope: OperatorScope
    supported_frequencies: tuple[str, ...]
    required_context_fields: tuple[str, ...]
    supported_windows: tuple[int, ...]
    implementation_version: str
    supported_backends: tuple[str, ...]
    qlib_compiler_alias: str
    gplearn_name_template: str
    missing_policy: str

    def __post_init__(self) -> None:
        """Validate operator metadata."""
        _require_identifier(self.name, "operator name")
        if type(self.arity) is not int or self.arity < 0:
            raise CoreValidationError("arity must be a non-negative integer")
        _require_string(self.description, "description")
        if not isinstance(self.parameter_schema, tuple) or not all(
            isinstance(parameter, OperatorParameterSpec)
            for parameter in self.parameter_schema
        ):
            raise CoreValidationError(
                "parameter_schema must be a tuple of OperatorParameterSpec"
            )
        if len(self.parameter_schema) != self.arity:
            raise CoreValidationError("parameter_schema must match arity")
        parameter_names = tuple(parameter.name for parameter in self.parameter_schema)
        if len(set(parameter_names)) != len(parameter_names):
            raise CoreValidationError("parameter_schema names must be unique")
        if not isinstance(self.scope, OperatorScope):
            raise CoreValidationError("scope must be OperatorScope")
        _require_tuple_of_strings(self.supported_frequencies, "supported_frequencies")
        _require_tuple_of_strings(
            self.required_context_fields, "required_context_fields"
        )
        if not isinstance(self.supported_windows, tuple):
            raise CoreValidationError("supported_windows must be a tuple")
        for window in self.supported_windows:
            if type(window) is not int or window <= 0:
                raise CoreValidationError("supported_windows must contain integers")
        window_parameters = tuple(
            parameter
            for parameter in self.parameter_schema
            if parameter.kind is OperatorParameterKind.WINDOW
        )
        if bool(self.supported_windows) != bool(window_parameters):
            raise CoreValidationError(
                "supported_windows and window parameter must be declared together"
            )
        if window_parameters and self.parameter_schema[-1] is not window_parameters[0]:
            raise CoreValidationError("window must be the final parameter")
        _require_string(self.implementation_version, "implementation_version")
        _require_tuple_of_strings(self.supported_backends, "supported_backends")
        _require_string(self.qlib_compiler_alias, "qlib_compiler_alias")
        _require_string(self.gplearn_name_template, "gplearn_name_template")
        _require_string(self.missing_policy, "missing_policy")

    def gplearn_name(self, window: int | None = None) -> str:
        """Return the fixed GPlearn function name for this operator use."""
        if "{window}" in self.gplearn_name_template:
            if window is None:
                raise CoreValidationError("window is required for GPlearn name")
            return self.gplearn_name_template.format(window=window)
        return self.gplearn_name_template

    @property
    def canonical_signature(self) -> str:
        """Return the canonical DSL signature shown to an LLM."""
        parameters = ", ".join(parameter.name for parameter in self.parameter_schema)
        return f"{self.name}({parameters})"


@dataclass(frozen=True, slots=True)
class OperatorCatalog:
    """Catalog of supported expression operators."""

    operators: tuple[OperatorSpec, ...]
    version: str = BASE_OPERATOR_SET_VERSION
    schema_version: str = OPERATOR_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate operator catalog consistency."""
        if not isinstance(self.operators, tuple):
            raise CoreValidationError("operators must be a tuple")
        names: set[str] = set()
        for operator in self.operators:
            if not isinstance(operator, OperatorSpec):
                raise CoreValidationError("every operator must be OperatorSpec")
            if operator.name in names:
                raise CoreValidationError(f"duplicate operator: {operator.name!r}")
            names.add(operator.name)
        _require_schema_version(self.version, "version")
        _require_schema_version(self.schema_version, "schema_version")

    def get(self, name: str) -> OperatorSpec | None:
        """Return operator metadata by name."""
        for operator in self.operators:
            if operator.name == name:
                return operator
        return None

    def contains(self, name: str) -> bool:
        """Return whether an operator is registered."""
        return self.get(name) is not None

    def gplearn_operator_names(self) -> tuple[str, ...]:
        """Return expanded GPlearn fixed operator names in catalog order."""
        names: list[str] = []
        for operator in self.operators:
            if operator.supported_windows:
                names.extend(
                    operator.gplearn_name(window)
                    for window in operator.supported_windows
                )
            else:
                names.append(operator.gplearn_name())
        return tuple(names)


@dataclass(frozen=True, slots=True)
class FieldNode:
    """Expression AST node for a field reference."""

    name: str

    def __post_init__(self) -> None:
        """Validate field node."""
        _require_identifier(self.name, "field name")


@dataclass(frozen=True, slots=True)
class NumberNode:
    """Expression AST node for a numeric literal."""

    value: int | float

    def __post_init__(self) -> None:
        """Validate number node."""
        if type(self.value) not in (int, float):
            raise CoreValidationError("number value must be int or float")


@dataclass(frozen=True, slots=True)
class CallNode:
    """Expression AST node for a function call."""

    function: str
    arguments: tuple[ExpressionNode, ...]

    def __post_init__(self) -> None:
        """Validate call node."""
        _require_identifier(self.function, "function name")
        if not isinstance(self.arguments, tuple):
            raise CoreValidationError("arguments must be a tuple")
        for argument in self.arguments:
            if not isinstance(argument, (FieldNode, NumberNode, CallNode)):
                raise CoreValidationError("every argument must be an expression node")


type ExpressionNode = FieldNode | NumberNode | CallNode


def build_base_field_catalog(
    *,
    optional_fields: tuple[str, ...] = DEFAULT_OPTIONAL_EXPRESSION_FIELDS,
    include_context_fields: bool = True,
) -> FieldCatalog:
    """Build the default field catalog for price-volume expressions.

    Args:
        optional_fields: Optional expression fields to expose for this run.
        include_context_fields: Whether to include context-only exposure fields.

    Returns:
        Field catalog containing required fields and selected optional fields.
    """
    allowed_optional = set(DEFAULT_OPTIONAL_EXPRESSION_FIELDS)
    unknown_optional = sorted(set(optional_fields) - allowed_optional)
    if unknown_optional:
        raise CoreValidationError(
            "unknown optional expression field(s): " + ", ".join(unknown_optional)
        )

    fields = [
        FieldSpec(name=name, description=_FIELD_DESCRIPTIONS[name])
        for name in BASE_EXPRESSION_FIELDS
    ]
    fields.extend(
        FieldSpec(name=name, description=_FIELD_DESCRIPTIONS[name])
        for name in optional_fields
    )
    if include_context_fields:
        fields.extend(
            FieldSpec(
                name=name,
                description=_FIELD_DESCRIPTIONS[name],
                context_only=True,
            )
            for name in DEFAULT_CONTEXT_FIELDS
        )
    return FieldCatalog(fields=tuple(fields))


def build_base_operator_catalog() -> OperatorCatalog:
    """Build OPERATOR_SET_BASE metadata aligned with the GPLEARN reference."""
    operators = [
        _operator(
            "add", 2, "Elementwise addition of two inputs.", OperatorScope.ELEMENTWISE
        ),
        _operator(
            "sub",
            2,
            "Elementwise subtraction of the second input from the first.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "mul",
            2,
            "Elementwise multiplication of two inputs.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "div",
            2,
            "Protected elementwise division; near-zero denominators return one.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator("abs", 1, "Elementwise absolute value.", OperatorScope.ELEMENTWISE),
        _operator(
            "sqrt",
            1,
            "Elementwise square root of the input absolute value.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "log",
            1,
            "Protected elementwise natural logarithm of the input absolute value.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "inv",
            1,
            "Protected elementwise reciprocal with zero output near zero.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "rank",
            1,
            "Daily cross-sectional percentile rank with average ranks for ties.",
            OperatorScope.CROSS_SECTIONAL,
        ),
        _operator(
            "rank_sub",
            2,
            "Difference between the daily cross-sectional ranks of two inputs.",
            OperatorScope.CROSS_SECTIONAL,
        ),
        _operator(
            "rank_div",
            2,
            "Protected ratio of the daily cross-sectional ranks of two inputs.",
            OperatorScope.CROSS_SECTIONAL,
        ),
        _operator(
            "sigmoid",
            1,
            "Elementwise logistic transformation.",
            OperatorScope.ELEMENTWISE,
        ),
        _operator(
            "indneutralize",
            1,
            "Subtract the same-day industry-group mean from each eligible input.",
            OperatorScope.INDUSTRY_GROUP,
            required_context_fields=("industry_id",),
        ),
    ]
    for name, arity, description in (
        ("delay", 2, "Value of an input a fixed number of trading days ago."),
        ("delta", 2, "Change from a fixed number of trading days ago."),
        ("corr", 3, "Rolling correlation of two inputs by instrument."),
        ("cov", 3, "Rolling covariance of two inputs by instrument."),
        (
            "decay_linear",
            2,
            "Rolling linearly weighted mean with larger weight on recent values.",
        ),
        ("ts_min", 2, "Rolling minimum by instrument."),
        ("ts_max", 2, "Rolling maximum by instrument."),
        ("ts_argmin", 2, "Position of the rolling minimum within each window."),
        ("ts_argmax", 2, "Position of the rolling maximum within each window."),
        ("ts_rank", 2, "Percentile rank of the current value within its window."),
        ("ts_sum", 2, "Rolling sum by instrument."),
        ("ts_stddev", 2, "Rolling sample standard deviation by instrument."),
        (
            "ts_zscore",
            2,
            "Current value standardized by its rolling mean and sample deviation.",
        ),
    ):
        operators.append(
            _operator(
                name,
                arity,
                description,
                OperatorScope.TIME_SERIES,
                supported_windows=DEFAULT_TIME_WINDOWS,
                gplearn_name_template=f"{name}_{{window}}",
            )
        )
    return OperatorCatalog(operators=tuple(operators))


def parse_expression(text: str) -> ExpressionNode:
    """Parse a CosmosAlpha expression into an AST.

    Args:
        text: Expression text.

    Returns:
        Parsed expression AST.

    Raises:
        ExpressionSyntaxError: If text violates the expression grammar.
    """
    parser = _ExpressionParser(text)
    return parser.parse()


def normalize_expression(node: ExpressionNode) -> str:
    """Return the deterministic canonical expression text."""
    if isinstance(node, FieldNode):
        return node.name
    if isinstance(node, NumberNode):
        return _format_number(node.value)
    if isinstance(node, CallNode):
        arguments = ", ".join(normalize_expression(arg) for arg in node.arguments)
        return f"{node.function}({arguments})"
    raise CoreValidationError("unsupported expression node")


def compute_factor_id(
    *,
    normalized_expression: str,
    frequency: str,
    operator_set_version: str,
    field_schema_version: str,
    expression_schema_version: str = EXPRESSION_SCHEMA_VERSION,
) -> str:
    """Compute the stable SHA-256 factor id from canonical identity fields."""
    return stable_sha256(
        {
            "normalized_expression": normalized_expression,
            "frequency": frequency,
            "operator_set_version": operator_set_version,
            "field_schema_version": field_schema_version,
            "expression_schema_version": expression_schema_version,
        }
    )


def validate_factor_proposal(
    proposal: FactorProposal,
    field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    *,
    existing_expressions: tuple[str, ...] = (),
    max_depth: int = 12,
    max_operator_nodes: int = 32,
    max_fields: int = 6,
    validator_version: str = "expression-validator/1.0",
) -> FactorValidationResult:
    """Validate a design-linked proposal and derive its factor specification."""
    if not isinstance(proposal, FactorProposal):
        raise CoreValidationError("proposal must be FactorProposal")
    report = validate_expression(
        proposal.proposed_expression,
        field_catalog,
        operator_catalog,
        proposal_id=proposal.proposal_id,
        frequency=proposal.frequency,
        existing_expressions=existing_expressions,
        max_depth=max_depth,
        max_operator_nodes=max_operator_nodes,
        max_fields=max_fields,
        validator_version=validator_version,
    )
    factor_spec = None
    if report.is_valid:
        assert report.factor_id is not None and report.normalized_expression is not None
        factor_spec = FactorSpec(
            factor_id=report.factor_id,
            proposal_id=proposal.proposal_id,
            hypothesis_id=proposal.hypothesis_id,
            name=proposal.name,
            expression=report.normalized_expression,
            description=proposal.description,
            frequency=proposal.frequency,
            required_fields=report.required_fields,
            expected_direction=proposal.expected_direction,
            operator_set_version=operator_catalog.version,
            field_schema_version=field_catalog.schema_version,
            expression_schema_version=EXPRESSION_SCHEMA_VERSION,
            schema_version="factor-spec/1.0",
        )
    return FactorValidationResult(
        proposal_id=proposal.proposal_id,
        report=report,
        factor_spec=factor_spec,
        schema_version="factor-validation-result/1.0",
    )


def validate_expression(
    expression: str,
    field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    *,
    proposal_id: str,
    frequency: str,
    existing_expressions: tuple[str, ...] = (),
    max_depth: int = 12,
    max_operator_nodes: int = 32,
    max_fields: int = 6,
    validator_version: str = "expression-validator/1.0",
) -> ValidationReport:
    """Validate expression semantics without requiring generation provenance."""
    _require_string(proposal_id, "proposal_id")
    _require_string(frequency, "frequency")
    if not isinstance(field_catalog, FieldCatalog):
        raise CoreValidationError("field_catalog must be FieldCatalog")
    if not isinstance(operator_catalog, OperatorCatalog):
        raise CoreValidationError("operator_catalog must be OperatorCatalog")
    try:
        parsed = parse_expression(expression)
    except ExpressionSyntaxError as error:
        return _invalid_report(
            proposal_id=proposal_id,
            code="SYNTAX_ERROR",
            message=str(error),
            validator_version=validator_version,
        )
    normalized_expression = normalize_expression(parsed)
    analysis = _ExpressionAnalysis()
    _analyze(parsed, analysis, depth=1)
    validations = (
        _validate_field_members,
        _validate_operator_members,
        _validate_arities_and_windows,
        _validate_frequency,
        _validate_context_fields,
        _validate_future_access,
        _validate_complexity,
        _validate_duplicate_expression,
        _validate_backend_support,
    )
    for validation in validations:
        issue = validation(
            parsed,
            analysis,
            field_catalog,
            operator_catalog,
            frequency,
            normalized_expression,
            existing_expressions,
            max_depth,
            max_operator_nodes,
            max_fields,
        )
        if issue is not None:
            return _invalid_report(
                proposal_id=proposal_id,
                code=issue[0],
                message=issue[1],
                validator_version=validator_version,
                required_fields=tuple(sorted(analysis.fields)),
                required_lookback=analysis.required_lookback,
            )
    return ValidationReport(
        proposal_id=proposal_id,
        factor_id=compute_factor_id(
            normalized_expression=normalized_expression,
            frequency=frequency,
            operator_set_version=operator_catalog.version,
            field_schema_version=field_catalog.schema_version,
        ),
        is_valid=True,
        normalized_expression=normalized_expression,
        required_fields=tuple(sorted(analysis.fields)),
        required_lookback=analysis.required_lookback,
        issues=(),
        validator_version=validator_version,
        schema_version="validation-report/1.0",
    )


def build_fixed_factor_design_draft(
    expression: str,
    *,
    name: str,
    description: str,
    expected_direction: str,
    source: str,
    operator_catalog: OperatorCatalog,
) -> FactorDesignDraft:
    """Describe a supplied expression without generating or changing its DSL."""
    analysis = _ExpressionAnalysis()
    _analyze(parse_expression(expression), analysis, depth=1)
    windows = {
        call.arguments[-1].value
        for call in analysis.operators
        if (operator := operator_catalog.get(call.function)) is not None
        and operator.supported_windows
        and call.arguments
        and isinstance(call.arguments[-1], NumberNode)
        and type(call.arguments[-1].value) is int
        and call.arguments[-1].value > 0
    }
    return FactorDesignDraft(
        name=name,
        description=description,
        mechanism_mapping=f"Fixed expression supplied by {source}.",
        signal_definition=expression,
        variables=tuple(
            FactorVariableDraft(field, "Input to the supplied expression.")
            for field in sorted(analysis.fields)
        ),
        formulation=expression,
        window_choices=tuple(
            WindowChoiceDraft(window, "Lookback specified by the supplied expression.")
            for window in sorted(windows)
        ),
        expected_direction=expected_direction,
        failure_conditions=(),
    )


@dataclass(slots=True)
class _ExpressionAnalysis:
    fields: set[str]
    operators: list[CallNode]
    numbers: list[NumberNode]
    max_depth: int
    operator_nodes: int
    required_lookback: int

    def __init__(self) -> None:
        self.fields = set()
        self.operators = []
        self.numbers = []
        self.max_depth = 0
        self.operator_nodes = 0
        self.required_lookback = 0


class _ExpressionParser:
    def __init__(self, text: str) -> None:
        if type(text) is not str:
            raise ExpressionSyntaxError("expression must be a string")
        if not text or not text.strip():
            raise ExpressionSyntaxError("expression must be non-empty")
        if len(text) > 512:
            raise ExpressionSyntaxError("expression exceeds 512 characters")
        self._text = text
        self._position = 0

    def parse(self) -> ExpressionNode:
        node = self._parse_expression()
        self._skip_ws()
        if not self._eof():
            raise self._error("unexpected trailing input")
        return node

    def _parse_expression(self) -> ExpressionNode:
        self._skip_ws()
        if self._eof():
            raise self._error("expected expression")
        number_match = _NUMBER_PATTERN.match(self._text, self._position)
        if number_match is not None:
            raw = number_match.group(0)
            self._position = number_match.end()
            return NumberNode(float(raw) if "." in raw else int(raw))

        identifier_match = _IDENTIFIER_PATTERN.match(self._text, self._position)
        if identifier_match is None:
            raise self._error("expected lowercase identifier or number")
        name = identifier_match.group(0)
        self._position = identifier_match.end()
        self._skip_ws()
        if self._peek("("):
            self._position += 1
            arguments: list[ExpressionNode] = []
            self._skip_ws()
            if self._peek(")"):
                self._position += 1
                return CallNode(name, tuple(arguments))
            while True:
                arguments.append(self._parse_expression())
                self._skip_ws()
                if self._peek(","):
                    self._position += 1
                    continue
                if self._peek(")"):
                    self._position += 1
                    return CallNode(name, tuple(arguments))
                raise self._error("expected ',' or ')'")
        return FieldNode(name)

    def _skip_ws(self) -> None:
        while not self._eof() and self._text[self._position].isspace():
            self._position += 1

    def _peek(self, text: str) -> bool:
        return self._text.startswith(text, self._position)

    def _eof(self) -> bool:
        return self._position >= len(self._text)

    def _error(self, message: str) -> ExpressionSyntaxError:
        return ExpressionSyntaxError(f"{message} at position {self._position}")


def _operator(
    name: str,
    arity: int,
    description: str,
    scope: OperatorScope,
    *,
    required_context_fields: tuple[str, ...] = (),
    supported_windows: tuple[int, ...] = (),
    gplearn_name_template: str | None = None,
) -> OperatorSpec:
    expression_parameter_count = arity - int(bool(supported_windows))
    expression_parameter_names = ("x", "y")[:expression_parameter_count]
    parameter_schema = tuple(
        OperatorParameterSpec(parameter_name, OperatorParameterKind.EXPRESSION)
        for parameter_name in expression_parameter_names
    )
    if supported_windows:
        parameter_schema += (
            OperatorParameterSpec("window", OperatorParameterKind.WINDOW),
        )
    return OperatorSpec(
        name=name,
        arity=arity,
        description=description,
        parameter_schema=parameter_schema,
        scope=scope,
        supported_frequencies=("1d",),
        required_context_fields=required_context_fields,
        supported_windows=supported_windows,
        implementation_version="gplearn-base/1.0",
        supported_backends=("qlib",),
        qlib_compiler_alias=f"qlib_{name}",
        gplearn_name_template=gplearn_name_template or name,
        missing_policy="preserve_nan",
    )


def _require_identifier(value: object, field_name: str) -> str:
    identifier = _require_string(value, field_name)
    if _IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise CoreValidationError(f"{field_name} must be a lowercase identifier")
    return identifier


def _format_number(value: int | float) -> str:
    if type(value) is int:
        return str(value)
    if value.is_integer():
        return f"{value:.1f}"
    return repr(value)


def _analyze(
    node: ExpressionNode, analysis: _ExpressionAnalysis, *, depth: int
) -> None:
    analysis.max_depth = max(analysis.max_depth, depth)
    if isinstance(node, FieldNode):
        analysis.fields.add(node.name)
        return
    if isinstance(node, NumberNode):
        analysis.numbers.append(node)
        return
    if isinstance(node, CallNode):
        analysis.operators.append(node)
        analysis.operator_nodes += 1
        for argument in node.arguments:
            _analyze(argument, analysis, depth=depth + 1)


def _validate_field_members(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    field_catalog: FieldCatalog,
    _operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    max_fields: int,
) -> tuple[str, str] | None:
    if not analysis.fields:
        return ("UNKNOWN_FIELD", "expression must reference at least one field")
    for field in sorted(analysis.fields):
        if not field_catalog.has_expression_field(field):
            return ("UNKNOWN_FIELD", f"unknown or unavailable field: {field}")
    if len(analysis.fields) > max_fields:
        return ("COMPLEXITY_EXCEEDED", "expression references too many fields")
    return None


def _validate_operator_members(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for call in analysis.operators:
        if not operator_catalog.contains(call.function):
            return ("UNKNOWN_OPERATOR", f"unknown operator: {call.function}")
    return None


def _validate_arities_and_windows(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for call in analysis.operators:
        operator = operator_catalog.get(call.function)
        if operator is None:
            continue
        if len(call.arguments) != operator.arity:
            return ("INVALID_ARITY", f"{call.function} expects {operator.arity} args")
        if operator.supported_windows:
            window_arg = call.arguments[-1]
            if (
                not isinstance(window_arg, NumberNode)
                or type(window_arg.value) is not int
            ):
                return ("INVALID_PARAMETER_TYPE", "window must be an integer literal")
            if window_arg.value not in operator.supported_windows:
                return ("UNSUPPORTED_WINDOW", f"unsupported window: {window_arg.value}")
            analysis.required_lookback = max(
                analysis.required_lookback, window_arg.value
            )
    return None


def _validate_frequency(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for call in analysis.operators:
        operator = operator_catalog.get(call.function)
        if operator is not None and frequency not in operator.supported_frequencies:
            return (
                "UNSUPPORTED_FREQUENCY",
                f"{call.function} does not support {frequency}",
            )
    return None


def _validate_context_fields(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for call in analysis.operators:
        operator = operator_catalog.get(call.function)
        if operator is None:
            continue
        for context_field in operator.required_context_fields:
            if not field_catalog.has_context_field(context_field):
                return (
                    "MISSING_OPERATOR_CONTEXT",
                    f"{call.function} requires context field {context_field}",
                )
    return None


def _validate_future_access(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    _operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for number in analysis.numbers:
        if type(number.value) is int and number.value < 0:
            return ("FUTURE_DATA_ACCESS", "negative windows are forbidden")
    for field in analysis.fields:
        if field.startswith("future_") or field in {"next_return", "forward_return"}:
            return ("FUTURE_DATA_ACCESS", f"future-looking field is forbidden: {field}")
    return None


def _validate_complexity(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    _operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    max_depth: int,
    max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    if analysis.max_depth > max_depth:
        return ("COMPLEXITY_EXCEEDED", "expression AST depth limit exceeded")
    if analysis.operator_nodes > max_operator_nodes:
        return ("COMPLEXITY_EXCEEDED", "operator node limit exceeded")
    return None


def _validate_duplicate_expression(
    _node: ExpressionNode,
    _analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    _operator_catalog: OperatorCatalog,
    _frequency: str,
    normalized_expression: str,
    existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    normalized_existing: set[str] = set()
    for expression in existing_expressions:
        try:
            normalized_existing.add(normalize_expression(parse_expression(expression)))
        except ExpressionSyntaxError:
            normalized_existing.add(expression)
    if normalized_expression in normalized_existing:
        return ("DUPLICATE_EXPRESSION", "expression already exists")
    return None


def _validate_backend_support(
    _node: ExpressionNode,
    analysis: _ExpressionAnalysis,
    _field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    _frequency: str,
    _normalized_expression: str,
    _existing_expressions: tuple[str, ...],
    _max_depth: int,
    _max_operator_nodes: int,
    _max_fields: int,
) -> tuple[str, str] | None:
    for call in analysis.operators:
        operator = operator_catalog.get(call.function)
        if operator is not None and "qlib" not in operator.supported_backends:
            return ("UNSUPPORTED_BACKEND", f"{call.function} has no qlib backend")
    return None


def _invalid_report(
    *,
    proposal_id: str,
    code: str,
    message: str,
    validator_version: str,
    required_fields: tuple[str, ...] = (),
    required_lookback: int = 0,
) -> ValidationReport:
    return ValidationReport(
        proposal_id=proposal_id,
        factor_id=None,
        is_valid=False,
        normalized_expression=None,
        required_fields=required_fields,
        required_lookback=required_lookback,
        issues=(
            ValidationIssue(
                code=code,
                severity="error",
                location="expression",
                message=message,
            ),
        ),
        validator_version=validator_version,
        schema_version="validation-report/1.0",
    )
