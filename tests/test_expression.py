"""Tests for factor field/operator catalogs and expression validation."""

from __future__ import annotations

import unittest
from dataclasses import replace

from cosmosalpha.agents import FactorProposal, FactorValidationResult
from cosmosalpha.core import ObjectRef
from cosmosalpha.factors import (
    BASE_OPERATOR_SET_VERSION,
    EXPRESSION_SCHEMA_VERSION,
    FieldCatalog,
    build_base_field_catalog,
    build_base_operator_catalog,
    compute_factor_id,
    normalize_expression,
    parse_expression,
    validate_factor_proposal,
)


def make_proposal(expression: str = "delay(close,5)") -> FactorProposal:
    """Create a proposal for expression validation tests."""
    return FactorProposal(
        proposal_id="proposal-001",
        name="test_factor",
        hypothesis_id="hypothesis-001",
        proposed_expression=expression,
        description="A test factor.",
        frequency="1d",
        expected_direction="positive",
        schema_version="factor-proposal/2.0",
        factor_design_ref=ObjectRef("factor_design", "design-001", "factor-design/1.0"),
        generation_attempt=1,
    )


def issue_code(result: FactorValidationResult) -> str:
    """Return the first validation issue code."""
    self = result
    return self.report.issues[0].code


class FactorCatalogTest(unittest.TestCase):
    """Verify field and operator catalogs."""

    def test_base_field_catalog_exposes_price_volume_fields(self) -> None:
        catalog = build_base_field_catalog()

        for field in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "turn",
            "free_turn",
            "returns",
        ):
            self.assertTrue(catalog.has_expression_field(field))
        for exposure in ("industry_id", "log_mkt_cap", "ret20", "turn20", "vol20"):
            self.assertFalse(catalog.has_expression_field(exposure))
            self.assertTrue(catalog.has_context_field(exposure))

    def test_optional_fields_can_be_narrowed_for_a_run(self) -> None:
        catalog = build_base_field_catalog(optional_fields=("vwap", "returns"))

        self.assertTrue(catalog.has_expression_field("vwap"))
        self.assertTrue(catalog.has_expression_field("returns"))
        self.assertFalse(catalog.has_expression_field("turn"))

    def test_base_operator_catalog_matches_gplearn_base_set(self) -> None:
        catalog = build_base_operator_catalog()
        expanded_names = catalog.gplearn_operator_names()

        self.assertEqual(catalog.version, BASE_OPERATOR_SET_VERSION)
        self.assertEqual(len(expanded_names), 52)
        for name in (
            "add",
            "rank_div",
            "indneutralize",
            "delay_5",
            "corr_10",
            "cov_20",
            "ts_zscore_20",
        ):
            self.assertIn(name, expanded_names)
        for name in ("scale", "ts_mean_5", "ts_rollzscore_10", "ts_product_20"):
            self.assertNotIn(name, expanded_names)
        self.assertTrue(catalog.contains("delay"))
        self.assertFalse(catalog.contains("delay_5"))
        self.assertFalse(catalog.contains("scale"))


class ExpressionParserTest(unittest.TestCase):
    """Verify parser and canonical normalization."""

    def test_parse_and_normalize_expression(self) -> None:
        self.assertEqual(
            normalize_expression(parse_expression("delay(close,5)")),
            "delay(close, 5)",
        )
        self.assertEqual(
            normalize_expression(
                parse_expression("div(delta(close, 5),delay(close,5))")
            ),
            "div(delta(close, 5), delay(close, 5))",
        )

    def test_rejects_non_contract_syntax(self) -> None:
        invalid = (
            "DIV(delta(close,5),delay(close,5))",
            "$close",
            "close / delay(close, 5)",
            "delay('close', 5)",
            "close.value",
            "delay(close, n=5)",
            "eval(close)",
        )

        for expression in invalid:
            with self.subTest(expression=expression):
                result = validate_factor_proposal(
                    make_proposal(expression),
                    build_base_field_catalog(),
                    build_base_operator_catalog(),
                )
                if expression == "eval(close)":
                    self.assertEqual(issue_code(result), "UNKNOWN_OPERATOR")
                else:
                    self.assertEqual(issue_code(result), "SYNTAX_ERROR")


class FactorValidationTest(unittest.TestCase):
    """Verify deterministic expression validation and factor IDs."""

    def test_valid_expression_generates_factor_spec_and_stable_id(self) -> None:
        field_catalog = build_base_field_catalog()
        operator_catalog = build_base_operator_catalog()
        result = validate_factor_proposal(
            make_proposal("div(delta(close,5), delay(close, 5))"),
            field_catalog,
            operator_catalog,
        )

        self.assertTrue(result.report.is_valid)
        self.assertIsNotNone(result.factor_spec)
        assert result.factor_spec is not None
        self.assertEqual(
            result.factor_spec.expression,
            "div(delta(close, 5), delay(close, 5))",
        )
        self.assertEqual(result.factor_spec.required_fields, ("close",))
        self.assertEqual(result.report.required_lookback, 5)
        self.assertEqual(result.report.factor_id, result.factor_spec.factor_id)
        self.assertEqual(
            result.factor_spec.operator_set_version, BASE_OPERATOR_SET_VERSION
        )
        self.assertEqual(
            result.factor_spec.expression_schema_version,
            EXPRESSION_SCHEMA_VERSION,
        )

    def test_narrowed_optional_field_and_exposure_field_are_rejected(self) -> None:
        for expression in ("vwap", "industry_id"):
            with self.subTest(expression=expression):
                result = validate_factor_proposal(
                    make_proposal(expression),
                    build_base_field_catalog(optional_fields=()),
                    build_base_operator_catalog(),
                )
                self.assertFalse(result.report.is_valid)
                self.assertEqual(issue_code(result), "UNKNOWN_FIELD")

        enabled = validate_factor_proposal(
            make_proposal("returns"),
            build_base_field_catalog(),
            build_base_operator_catalog(),
        )
        self.assertTrue(enabled.report.is_valid)

    def test_operator_arity_window_context_and_constant_failures(self) -> None:
        cases = {
            "unknown(close)": "UNKNOWN_OPERATOR",
            "delay(close)": "INVALID_ARITY",
            "delay(close, 0)": "UNSUPPORTED_WINDOW",
            "delay(close, 6)": "UNSUPPORTED_WINDOW",
            "delay(close, -5)": "UNSUPPORTED_WINDOW",
            "delay(close, 5.0)": "INVALID_PARAMETER_TYPE",
            "1": "UNKNOWN_FIELD",
        }

        for expression, expected_code in cases.items():
            with self.subTest(expression=expression):
                result = validate_factor_proposal(
                    make_proposal(expression),
                    build_base_field_catalog(),
                    build_base_operator_catalog(),
                )
                self.assertFalse(result.report.is_valid)
                self.assertEqual(issue_code(result), expected_code)

        missing_context = validate_factor_proposal(
            make_proposal("indneutralize(close)"),
            build_base_field_catalog(include_context_fields=False),
            build_base_operator_catalog(),
        )
        self.assertEqual(issue_code(missing_context), "MISSING_OPERATOR_CONTEXT")

    def test_complexity_and_duplicate_expression_are_rejected(self) -> None:
        catalog = build_base_field_catalog()
        operators = build_base_operator_catalog()

        complex_result = validate_factor_proposal(
            make_proposal("rank(rank(rank(close)))"),
            catalog,
            operators,
            max_depth=3,
        )
        self.assertEqual(issue_code(complex_result), "COMPLEXITY_EXCEEDED")

        duplicate = validate_factor_proposal(
            make_proposal("delay(close, 5)"),
            catalog,
            operators,
            existing_expressions=("delay(close,5)",),
        )
        self.assertEqual(issue_code(duplicate), "DUPLICATE_EXPRESSION")

    def test_factor_id_is_stable_and_sensitive_to_identity_inputs(self) -> None:
        proposal = make_proposal("delay(close,5)")
        first = validate_factor_proposal(
            proposal,
            build_base_field_catalog(),
            build_base_operator_catalog(),
        )
        reordered_catalog = FieldCatalog(
            fields=tuple(reversed(build_base_field_catalog().fields))
        )
        reordered = validate_factor_proposal(
            proposal,
            reordered_catalog,
            build_base_operator_catalog(),
        )
        changed_expression = validate_factor_proposal(
            make_proposal("delay(close,10)"),
            build_base_field_catalog(),
            build_base_operator_catalog(),
        )

        self.assertEqual(first.report.factor_id, reordered.report.factor_id)
        self.assertNotEqual(first.report.factor_id, changed_expression.report.factor_id)
        self.assertNotEqual(
            compute_factor_id(
                normalized_expression="delay(close, 5)",
                frequency="1d",
                operator_set_version="operator-set-base/2.0",
                field_schema_version="field-catalog/1.0",
            ),
            first.report.factor_id,
        )

    def test_unsupported_frequency_and_backend_are_rejected(self) -> None:
        unsupported_frequency = validate_factor_proposal(
            replace(make_proposal("delay(close,5)"), frequency="1h"),
            build_base_field_catalog(),
            build_base_operator_catalog(),
        )
        self.assertEqual(issue_code(unsupported_frequency), "UNSUPPORTED_FREQUENCY")

        operator_catalog = build_base_operator_catalog()
        first_operator = operator_catalog.operators[0]
        no_backend_operator = replace(first_operator, supported_backends=())
        no_backend_catalog = replace(
            operator_catalog,
            operators=(no_backend_operator, *operator_catalog.operators[1:]),
        )
        unsupported_backend = validate_factor_proposal(
            make_proposal("add(close, 1)"),
            build_base_field_catalog(),
            no_backend_catalog,
        )
        self.assertEqual(issue_code(unsupported_backend), "UNSUPPORTED_BACKEND")


if __name__ == "__main__":
    unittest.main()
