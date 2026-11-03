"""Fixed numerical, mask and causality regression checks."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pytest

from cosmosalpha.core import (
    decode_record_artifact,
)
from cosmosalpha.factors import (
    CalculationFrame,
    FactorCalculationError,
    evaluate_factor_spec,
)
from cosmosalpha.storage import LocalArtifactRepository
from tests.fixtures import make_frame, make_spec


class FactorOperatorExecutionTest(unittest.TestCase):
    """Verify base operator execution semantics."""

    def test_elementwise_operators_preserve_missing_and_invalid_domains(self) -> None:
        frame = make_frame(
            datetimes=("2020-01-01",) * 5,
            instruments=("a", "b", "c", "d", "e"),
            fields={
                "close": (4.0, -1.0, 0.0, math.nan, 9.0),
                "open": (2.0, 2.0, 0.0, 1.0, math.nan),
            },
        )

        np.testing.assert_allclose(
            evaluate_factor_spec(
                make_spec("div(close, open)", required_fields=("close", "open")),
                frame,
            ),
            np.array([2.0, -0.5, math.nan, math.nan, math.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("sqrt(close)"), frame),
            np.array([2.0, math.nan, 0.0, math.nan, 3.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("log(close)"), frame),
            np.array([math.log(4.0), math.nan, math.nan, math.nan, math.log(9.0)]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("inv(close)"), frame),
            np.array([0.25, -1.0, math.nan, math.nan, 1.0 / 9.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("sigmoid(close)"), frame),
            1.0 / (1.0 + np.exp(-np.array([4.0, -1.0, 0.0, math.nan, 9.0]))),
            equal_nan=True,
        )

    def test_rank_operators_are_by_date_and_use_eligible_tie_ranks(self) -> None:
        frame = make_frame(
            datetimes=("2020-01-01",) * 3 + ("2020-01-02",) * 3,
            instruments=("a", "b", "c", "a", "b", "c"),
            fields={
                "close": (1.0, 2.0, 100.0, 5.0, 5.0, 7.0),
                "open": (3.0, 2.0, 1.0, 7.0, 5.0, 5.0),
            },
            eligible=(True, True, False, True, True, True),
        )

        rank_close = evaluate_factor_spec(make_spec("rank(close)"), frame)
        np.testing.assert_allclose(
            rank_close,
            np.array([0.5, 1.0, math.nan, 0.5, 0.5, 1.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(
                make_spec("rank_sub(close, open)", required_fields=("close", "open")),
                frame,
            ),
            rank_close - np.array([1.0, 0.5, math.nan, 1.0, 0.5, 0.5]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(
                make_spec("rank_div(close, open)", required_fields=("close", "open")),
                frame,
            ),
            rank_close / np.array([1.0, 0.5, math.nan, 1.0, 0.5, 0.5]),
            equal_nan=True,
        )

    def test_indneutralize_uses_date_industry_and_eligible_samples(self) -> None:
        frame = make_frame(
            datetimes=("2020-01-01",) * 4 + ("2020-01-02",) * 4,
            instruments=("a", "b", "c", "d", "a", "b", "c", "d"),
            fields={"close": (1.0, 3.0, 100.0, 10.0, 2.0, 6.0, 8.0, 9.0)},
            industry=(1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0),
            eligible=(True, True, False, True, True, True, True, False),
        )

        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("indneutralize(close)"), frame),
            np.array([-1.0, 1.0, math.nan, 0.0, -2.0, 2.0, 0.0, math.nan]),
            equal_nan=True,
        )

        missing_context = make_frame(
            datetimes=("2020-01-01",),
            instruments=("a",),
            fields={"close": (1.0,)},
            industry=(math.nan,),
        )
        # A supplied, entirely unknown industry column is not absent context.
        np.testing.assert_equal(
            evaluate_factor_spec(make_spec("indneutralize(close)"), missing_context),
            [math.nan],
        )
        missing_context = CalculationFrame(
            datetimes=missing_context.datetimes,
            instruments=missing_context.instruments,
            fields=missing_context.fields,
            context_fields={},
            eligible_mask=missing_context.eligible_mask,
        )
        with self.assertRaises(FactorCalculationError):
            evaluate_factor_spec(make_spec("indneutralize(close)"), missing_context)

        # Record conversion must not turn an absent field into a supplied NaN column.
        missing_frame = CalculationFrame.from_records(
            market_records=(
                {"datetime": "2020-01-01", "instrument": "a", "close": 1.0},
            ),
            exposure_records=({"datetime": "2020-01-01", "instrument": "a"},),
            mask_records=(
                {"datetime": "2020-01-01", "instrument": "a", "eligible_mask": True},
            ),
            required_fields=("close",),
        )
        self.assertNotIn("industry_id", missing_frame.context_fields)
        with self.assertRaisesRegex(FactorCalculationError, "MISSING_OPERATOR_CONTEXT"):
            evaluate_factor_spec(make_spec("indneutralize(close)"), missing_frame)

    def test_indneutralize_does_not_group_unknown_industries(self) -> None:
        frame = make_frame(
            datetimes=("2020-01-01",) * 4,
            instruments=("a", "b", "c", "d"),
            fields={"close": (1.0, 3.0, 100.0, 200.0)},
            industry=(801780.0, 801780.0, math.nan, math.nan),
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("indneutralize(close)"), frame),
            np.array([-1.0, 1.0, math.nan, math.nan]),
            equal_nan=True,
        )

    def test_rolling_operators_respect_groups_and_min_periods(self) -> None:
        frame = make_frame(
            datetimes=tuple(f"2020-01-{day:02d}" for day in range(1, 7)),
            instruments=("a",) * 6,
            fields={"close": (1.0, 2.0, 3.0, math.nan, 5.0, 6.0)},
        )

        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("delay(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, math.nan, 1.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("delta(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, math.nan, 5.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_sum(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 11.0, 16.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_min(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 1.0, 2.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_max(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 5.0, 6.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_argmin(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 1.0, 1.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_argmax(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 5.0, 5.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_rank(close, 5)"), frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 1.0, 1.0]),
            equal_nan=True,
        )
        tied_frame = make_frame(
            datetimes=frame.datetimes,
            instruments=frame.instruments,
            fields={"close": (1.0, 2.0, 2.0, math.nan, 2.0, 3.0)},
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_rank(close, 5)"), tied_frame),
            np.array([math.nan, math.nan, math.nan, math.nan, 0.75, 1.0]),
            equal_nan=True,
        )
        with self.assertRaises(FactorCalculationError):
            evaluate_factor_spec(make_spec("ts_sum(close, 6)"), frame)

    def test_rolling_sample_statistics_and_future_information_constraint(self) -> None:
        base_frame = make_frame(
            datetimes=tuple(f"2020-01-{day:02d}" for day in range(1, 7)),
            instruments=("a",) * 6,
            fields={
                "close": (1.0, 2.0, 3.0, 4.0, 5.0, 999.0),
                "open": (2.0, 4.0, 6.0, 8.0, 10.0, -999.0),
            },
        )
        changed_future = make_frame(
            datetimes=base_frame.datetimes,
            instruments=base_frame.instruments,
            fields={
                "close": (1.0, 2.0, 3.0, 4.0, 5.0, -999.0),
                "open": (2.0, 4.0, 6.0, 8.0, 10.0, 999.0),
            },
        )

        std = evaluate_factor_spec(make_spec("ts_stddev(close, 5)"), base_frame)
        zscore = evaluate_factor_spec(make_spec("ts_zscore(close, 5)"), base_frame)
        cov = evaluate_factor_spec(
            make_spec("cov(close, open, 5)", required_fields=("close", "open")),
            base_frame,
        )
        corr = evaluate_factor_spec(
            make_spec("corr(close, open, 5)", required_fields=("close", "open")),
            base_frame,
        )

        self.assertAlmostEqual(std[4], math.sqrt(2.5))
        self.assertAlmostEqual(zscore[4], 2.0 / math.sqrt(2.5))
        self.assertAlmostEqual(cov[4], 5.0)
        self.assertAlmostEqual(corr[4], 1.0)

        missing_frame = make_frame(
            datetimes=tuple(f"2020-02-{day:02d}" for day in range(1, 6)),
            instruments=("a",) * 5,
            fields={
                "close": (1.0, 2.0, math.nan, 4.0, 5.0),
                "open": (2.0, 4.0, math.nan, 8.0, 10.0),
            },
        )
        expected_std = math.sqrt(10.0 / 3.0)
        self.assertAlmostEqual(
            evaluate_factor_spec(make_spec("ts_stddev(close, 5)"), missing_frame)[4],
            expected_std,
        )
        self.assertAlmostEqual(
            evaluate_factor_spec(make_spec("ts_zscore(close, 5)"), missing_frame)[4],
            2.0 / expected_std,
        )
        self.assertAlmostEqual(
            evaluate_factor_spec(
                make_spec("cov(close, open, 5)", required_fields=("close", "open")),
                missing_frame,
            )[4],
            20.0 / 3.0,
        )
        self.assertAlmostEqual(
            evaluate_factor_spec(
                make_spec("corr(close, open, 5)", required_fields=("close", "open")),
                missing_frame,
            )[4],
            1.0,
        )
        np.testing.assert_allclose(
            evaluate_factor_spec(make_spec("ts_sum(close, 5)"), base_frame)[:5],
            evaluate_factor_spec(make_spec("ts_sum(close, 5)"), changed_future)[:5],
            equal_nan=True,
        )


def test_trailing_operator_does_not_read_future_values() -> None:
    dates = tuple(f"2020-01-{index:02d}" for index in range(1, 11))
    fields = {"close": tuple(float(index) for index in range(10))}
    first = make_frame(datetimes=dates, instruments=("a",) * 10, fields=fields)
    second = make_frame(
        datetimes=dates,
        instruments=("a",) * 10,
        fields={"close": fields["close"][:6] + (900.0,) * 4},
    )
    spec = make_spec("ts_sum(close, 5)")
    np.testing.assert_allclose(
        evaluate_factor_spec(spec, first)[:6],
        evaluate_factor_spec(spec, second)[:6],
        equal_nan=True,
    )


def test_forward_returns_align_by_instrument_and_leave_tail_missing(
    tmp_path: Path,
) -> None:
    from cosmosalpha.evaluation import ForwardReturnCalculator
    from cosmosalpha.pipeline.smoke import _build_smoke_snapshot

    artifacts = LocalArtifactRepository(tmp_path)
    snapshot = _build_smoke_snapshot(artifacts)
    market = decode_record_artifact(artifacts.get_bytes(snapshot.market_data.data_ref))
    result = ForwardReturnCalculator(artifacts).calculate(snapshot)
    records = decode_record_artifact(artifacts.get_bytes(result.return_values_ref))
    assert len(records) == len(market)
    for index in range(16):
        assert records[index]["forward_return"] == pytest.approx(
            market[index + 8]["close"] / market[index]["close"] - 1.0
        )
    assert all(row["forward_return"] is None for row in records[-8:])


def test_neutralization_preserves_missing_industry_and_current_date_only() -> None:
    from cosmosalpha.evaluation import preprocess_feature_values

    rng = np.random.default_rng(17)
    industry = np.tile(np.repeat(np.arange(3, dtype=float), 8), 2)
    industry[23] = np.nan
    dates = ("2020-01-01",) * 24 + ("2020-01-02",) * 24
    raw = rng.normal(size=48)
    continuous = rng.normal(size=(48, 4))
    arguments = dict(
        datetimes=dates,
        industry=industry,
        continuous=continuous,
        eligible=np.ones(48, dtype=bool),
        mad_multiplier=5.0,
        min_cross_section_samples=8,
    )
    first = preprocess_feature_values(raw, **arguments)
    altered = raw.copy()
    altered[24:] = 1e9
    second = preprocess_feature_values(altered, **arguments)
    np.testing.assert_allclose(first[:24], second[:24], equal_nan=True)
    assert np.isnan(first[23])
    finite = first[:23]
    assert np.isfinite(finite).all()
    assert float(np.mean(finite)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(finite)) == pytest.approx(1.0, abs=1e-12)
