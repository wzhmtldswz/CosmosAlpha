"""Deterministic IC/RankIC evaluator and acceptance policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Protocol

import numpy as np

from cosmosalpha.agents import (
    AcceptanceDecision,
    AcceptanceMetricThreshold,
    AcceptancePolicySnapshot,
    EvaluationResult,
    FactorPreprocessingResult,
    MetricValue,
)
from cosmosalpha.core import ArtifactRef, ObjectRef, WarningInfo
from cosmosalpha.data import DataSnapshot
from cosmosalpha.evaluation._statistics import (
    mean_daily_cross_sectional_pearson,
)
from cosmosalpha.evaluation._statistics import (
    safe_pearson as _safe_pearson,
)
from cosmosalpha.evaluation.preprocessing import (
    EvaluationInputCache,
    EvaluationInputs,
    _load_evaluation_inputs,
    _read_table,
    _table_numeric_array,
    _validate_table_alignment,
)

EVALUATION_PROTOCOL_ID = "single_factor_ic_v2"
EVALUATOR_VERSION = "factor-evaluator/2.3"
ACCEPTANCE_POLICY_ID = "deterministic_multi_metric_v1"
ACCEPTANCE_POLICY_VERSION = "2.1.0"
DEFAULT_ACCEPTANCE_METRIC = "rank_ic_mean"
DEFAULT_ACCEPTANCE_THRESHOLD = 0.02
_REFERENCE_FACTOR_LOAD_WORKERS = 8

METRIC_DESCRIPTIONS = {
    "ic_mean": "Mean daily Pearson correlation between factor values and forward returns.",
    "rank_ic_mean": "Mean daily rank correlation between factor values and forward returns.",
    "ic_ir": "Mean daily IC divided by its sample standard deviation.",
    "rank_ic_ir": "Mean daily RankIC divided by its sample standard deviation.",
    "rank_ic_positive_rate": "Fraction of evaluated dates with positive RankIC.",
    "directional_positive_rate": (
        "Fraction of evaluated dates with positive direction-aligned RankIC."
    ),
    "factor_orientation": (
        "Sign used to align factor exposure with its mean daily RankIC."
    ),
    "top20_mean_return": (
        "Equal-date-weighted forward return of the direction-aligned top 20% stocks."
    ),
    "rank_ic_std": "Sample standard deviation of daily RankIC (ddof=1).",
    "rank_ic_oos_mean": "Mean RankIC across chronological out-of-sample segments.",
    "rank_ic_oos_min": "Worst mean RankIC among chronological out-of-sample segments.",
    "coverage": "Fraction of eligible samples included in factor evaluation.",
    "factor_max_abs_daily_correlation": (
        "Maximum absolute equal-date mean of daily cross-sectional correlations "
        "with accepted factors on the same data snapshot."
    ),
}


class FactorEvaluationError(Exception):
    """Raised when deterministic factor evaluation cannot be performed."""


class ArtifactRepository(Protocol):
    """Minimal artifact repository interface for evaluation components."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read artifact bytes."""


class ReferenceFactorValuesProvider(Protocol):
    """Provide accepted peer factor values for one comparable data snapshot."""

    def reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Return deterministic peer factor value artifact references."""


@dataclass(frozen=True, slots=True)
class EvaluationTiming:
    """Non-persisted timing from the latest factor evaluation."""

    total_seconds: float
    input_seconds: float
    metric_seconds: float
    reference_correlation_seconds: float


@dataclass(frozen=True, slots=True)
class MetricRequirement:
    """One versioned lower/upper bound required by a multi-metric policy."""

    metric_name: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        """Validate a non-ambiguous finite metric range."""
        if type(self.metric_name) is not str or not self.metric_name:
            raise FactorEvaluationError("metric_name must be a non-empty string")
        if self.minimum is None and self.maximum is None:
            raise FactorEvaluationError("metric requirement needs a bound")
        for field_name in ("minimum", "maximum"):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(float(value))
            ):
                raise FactorEvaluationError(f"{field_name} must be finite when set")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise FactorEvaluationError("minimum must not exceed maximum")


@dataclass(frozen=True, slots=True)
class GraphNodeEligibility:
    """Technical Graph admission result independent of library acceptance."""

    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the deterministic eligibility result."""
        if type(self.eligible) is not bool:
            raise FactorEvaluationError("eligible must be a boolean")
        if not isinstance(self.reason_codes, tuple) or not all(
            type(code) is str and code for code in self.reason_codes
        ):
            raise FactorEvaluationError("reason_codes must be a tuple of strings")


class GraphNodeEligibilityGate:
    """Decide whether an evaluated factor is technically usable by Graph."""

    _INTRINSIC_METRICS = (
        "rank_ic_mean",
        "rank_ic_std",
        "top20_mean_return",
    )

    def __init__(self, *, min_effective_date_ratio: float) -> None:
        """Create a gate with an explicit evaluation date-coverage threshold."""
        if (
            type(min_effective_date_ratio) not in (int, float)
            or not math.isfinite(float(min_effective_date_ratio))
            or not 0.0 <= float(min_effective_date_ratio) <= 1.0
        ):
            raise FactorEvaluationError(
                "min_effective_date_ratio must be between 0 and 1"
            )
        self._min_effective_date_ratio = float(min_effective_date_ratio)

    def evaluate(self, evaluation_result: EvaluationResult) -> GraphNodeEligibility:
        """Return technical Graph admission without applying AcceptancePolicy."""
        if not isinstance(evaluation_result, EvaluationResult):
            raise FactorEvaluationError("evaluation_result must be EvaluationResult")
        metrics = {metric.name: metric.value for metric in evaluation_result.metrics}
        reasons = [
            f"MISSING_{_metric_token(name)}_METRIC"
            for name in self._INTRINSIC_METRICS
            if name not in metrics
        ]
        if evaluation_result.effective_date_count < 2:
            reasons.append("INSUFFICIENT_RANK_IC_STD_SAMPLE")
        if evaluation_result.total_date_count in (None, 0):
            reasons.append("INSUFFICIENT_VALID_DATE_COVERAGE")
        elif (
            evaluation_result.effective_date_count / evaluation_result.total_date_count
            < self._min_effective_date_ratio
        ):
            reasons.append("INSUFFICIENT_VALID_DATE_COVERAGE")
        return GraphNodeEligibility(not reasons, tuple(reasons))


class FactorEvaluator:
    """Evaluate preprocessed factor values with daily IC and RankIC metrics."""

    def __init__(
        self,
        artifact_repository: ArtifactRepository,
        *,
        min_cross_section_samples: int = 20,
        min_daily_effective_stock_ratio: float = 0.8,
        oos_segment_count: int = 2,
        reference_factor_values_provider: ReferenceFactorValuesProvider | None = None,
        input_cache: EvaluationInputCache | None = None,
    ) -> None:
        """Create a deterministic factor evaluator.

        Args:
            artifact_repository: Repository used to resolve JSON artifacts.
            min_cross_section_samples: Minimum finite factor/return pairs required
                for one date to contribute metrics.
            min_daily_effective_stock_ratio: Minimum fraction of eligible stocks
                with finite factor/return pairs required for one date.
            oos_segment_count: Number of chronological holdout segments summarized.
            reference_factor_values_provider: Optional accepted-factor source used for
                same-snapshot correlation metrics.
            input_cache: Optional run-local cache shared with factor preprocessing.
        """
        if type(min_cross_section_samples) is not int or min_cross_section_samples < 2:
            raise FactorEvaluationError("min_cross_section_samples must be >= 2")
        if (
            type(min_daily_effective_stock_ratio) not in (int, float)
            or not math.isfinite(float(min_daily_effective_stock_ratio))
            or not 0.0 <= float(min_daily_effective_stock_ratio) <= 1.0
        ):
            raise FactorEvaluationError(
                "min_daily_effective_stock_ratio must be between 0 and 1"
            )
        if type(oos_segment_count) is not int or oos_segment_count < 1:
            raise FactorEvaluationError("oos_segment_count must be >= 1")
        self._artifacts = artifact_repository
        self._min_samples = min_cross_section_samples
        self._min_daily_ratio = float(min_daily_effective_stock_ratio)
        self._oos_segment_count = oos_segment_count
        self._reference_provider = reference_factor_values_provider
        self._input_cache = input_cache or EvaluationInputCache()
        self._last_timing = EvaluationTiming(0.0, 0.0, 0.0, 0.0)

    @property
    def last_timing(self) -> EvaluationTiming:
        """Return timing from the latest successful evaluation."""
        return self._last_timing

    def set_reference_factor_values_provider(
        self,
        provider: ReferenceFactorValuesProvider | None,
    ) -> None:
        """Replace the peer-factor view used by subsequent evaluations.

        Graph creates one evaluator per worker Loop and calls this before the
        worker starts.  The provider itself is immutable for the lifetime of a
        round; this method only avoids rebuilding the Loop to inject that view.
        """
        self._reference_provider = provider

    def evaluate(
        self,
        preprocessing_result: FactorPreprocessingResult,
        preprocessing_result_ref: ObjectRef,
        data_snapshot: DataSnapshot,
        *,
        evaluation_start_date: str | None = None,
        evaluation_end_date: str | None = None,
    ) -> EvaluationResult:
        """Evaluate one preprocessed factor result.

        Args:
            preprocessing_result: Preprocessed factor artifact metadata.
            preprocessing_result_ref: Reference to the preprocessed factor object.
            data_snapshot: Snapshot containing the aligned evaluation artifacts.
            evaluation_start_date: Optional inclusive evaluation start date. When
                omitted, the snapshot request start date is used.
            evaluation_end_date: Optional inclusive evaluation end date. When
                omitted, the snapshot request end date is used.

        Returns:
            Deterministic metrics for the requested date window.

        Raises:
            FactorEvaluationError: If the date window is invalid or evaluation
                cannot be performed.
        """
        started_at = perf_counter()
        self._validate_inputs(
            preprocessing_result,
            preprocessing_result_ref,
            data_snapshot,
        )
        evaluation_start, evaluation_end = _evaluation_date_range(
            data_snapshot,
            evaluation_start_date=evaluation_start_date,
            evaluation_end_date=evaluation_end_date,
        )
        phase_started_at = perf_counter()
        inputs = self._evaluation_inputs(preprocessing_result, data_snapshot)
        factor_values = self._input_cache.get_factor_values(
            inputs,
            preprocessing_result.evaluated_factor_values_ref,
            lambda: _load_aligned_factor_values(
                self._artifacts,
                preprocessing_result.evaluated_factor_values_ref,
                inputs,
                "evaluated_factor_values",
            ),
        )
        forward_returns = inputs.forward_returns
        eligible = inputs.eligible
        input_seconds = perf_counter() - phase_started_at

        phase_started_at = perf_counter()
        evaluation_groups: list[slice] = []
        insufficient_dates = 0
        low_coverage_dates = 0
        for date_value, indices in inputs.date_slices.items():
            if not (evaluation_start <= date_value <= evaluation_end):
                continue
            if (
                int(
                    np.count_nonzero(
                        eligible[indices] & np.isfinite(forward_returns[indices])
                    )
                )
                < self._min_samples
            ):
                continue
            evaluation_groups.append(indices)
        if not evaluation_groups:
            raise FactorEvaluationError(
                "NO_VALID_EVALUATION_DATES: no evaluable dates in requested range"
            )

        daily_ic: list[float] = []
        daily_rank_ic: list[float] = []
        daily_top20_indices: list[np.ndarray] = []
        effective_sample_count = 0
        degenerate_dates = 0
        for indices in evaluation_groups:
            factor_day = factor_values[indices]
            return_day = forward_returns[indices]
            effective_stock_mask = (
                eligible[indices] & np.isfinite(factor_day) & np.isfinite(return_day)
            )
            effective_stock_count = int(np.count_nonzero(effective_stock_mask))
            eligible_count = int(np.count_nonzero(eligible[indices]))
            if (
                effective_stock_count < self._min_samples
                or eligible_count == 0
                or effective_stock_count / eligible_count < self._min_daily_ratio
            ):
                low_coverage_dates += 1
                continue
            keep = eligible[indices] & np.isfinite(factor_day) & np.isfinite(return_day)
            valid_count = int(np.count_nonzero(keep))
            if valid_count < self._min_samples:
                insufficient_dates += 1
                continue
            factor_valid = factor_day[keep]
            return_valid = return_day[keep]
            ic_value = _safe_pearson(factor_valid, return_valid)
            rank_ic_value = _safe_pearson(
                _average_ranks(factor_valid),
                _average_ranks(return_valid),
            )
            if not math.isfinite(ic_value) or not math.isfinite(rank_ic_value):
                degenerate_dates += 1
                continue
            daily_ic.append(ic_value)
            daily_rank_ic.append(rank_ic_value)
            start = 0 if indices.start is None else indices.start
            daily_top20_indices.append(np.flatnonzero(keep) + start)
            effective_sample_count += int(factor_valid.size)

        if not daily_ic:
            raise FactorEvaluationError("NO_VALID_EVALUATION_DATES: no finite IC dates")

        warnings = _evaluation_warnings(
            insufficient_dates=insufficient_dates,
            low_coverage_dates=low_coverage_dates,
            degenerate_dates=degenerate_dates,
            min_samples=self._min_samples,
            min_daily_ratio=self._min_daily_ratio,
        )
        evaluable_sample_count = sum(
            int(
                np.count_nonzero(
                    eligible[indices] & np.isfinite(forward_returns[indices])
                )
            )
            for indices in evaluation_groups
        )
        coverage = effective_sample_count / evaluable_sample_count
        metric_seconds = perf_counter() - phase_started_at
        phase_started_at = perf_counter()
        reference_refs = self._reference_factor_values_refs(
            data_snapshot.snapshot_id,
            exclude_factor_id=preprocessing_result.factor_id,
        )
        correlation_dates = tuple(
            date_value
            for date_value in inputs.date_slices
            if evaluation_start <= date_value <= evaluation_end
        )
        max_abs_daily_correlation, correlation_date_count = (
            _max_abs_daily_factor_correlation(
                factor_values,
                inputs,
                correlation_dates,
                reference_refs,
                self._artifacts,
                self._input_cache,
                min_cross_section_samples=self._min_samples,
                min_daily_effective_stock_ratio=self._min_daily_ratio,
            )
        )
        reference_correlation_seconds = perf_counter() - phase_started_at
        orientation = factor_orientation(float(np.mean(np.asarray(daily_rank_ic))))
        result = EvaluationResult(
            factor_id=preprocessing_result.factor_id,
            data_snapshot_id=data_snapshot.snapshot_id,
            raw_factor_values_ref=preprocessing_result.input_factor_values_ref,
            evaluated_factor_values_ref=preprocessing_result.evaluated_factor_values_ref,
            factor_preprocessing_result_ref=preprocessing_result_ref,
            forward_return_ref=preprocessing_result.forward_return_ref,
            evaluation_protocol_id=EVALUATION_PROTOCOL_ID,
            preprocessing_protocol_id=preprocessing_result.preprocessing_protocol_id,
            neutralization_applied=preprocessing_result.neutralization_applied,
            metrics=_build_metrics(
                daily_ic,
                daily_rank_ic,
                effective_sample_count,
                coverage=float(coverage),
                oos_segment_count=self._oos_segment_count,
                max_abs_daily_correlation=max_abs_daily_correlation,
                correlation_date_count=correlation_date_count,
                orientation=orientation,
                daily_top20_indices=daily_top20_indices,
                factor_values=factor_values,
                forward_returns=forward_returns,
                instruments=inputs.instruments,
            ),
            effective_date_count=len(daily_ic),
            effective_sample_count=effective_sample_count,
            coverage=float(coverage),
            warnings=warnings,
            evaluator_version=EVALUATOR_VERSION,
            schema_version="evaluation-result/1.1",
            total_date_count=len(evaluation_groups),
        )
        self._last_timing = EvaluationTiming(
            total_seconds=perf_counter() - started_at,
            input_seconds=input_seconds,
            metric_seconds=metric_seconds,
            reference_correlation_seconds=reference_correlation_seconds,
        )
        return result

    def _evaluation_inputs(
        self,
        preprocessing_result: FactorPreprocessingResult,
        data_snapshot: DataSnapshot,
    ) -> EvaluationInputs:
        key = (
            data_snapshot.snapshot_id,
            preprocessing_result.forward_return_ref.artifact_id,
            preprocessing_result.mask_data_ref.artifact_id,
        )
        return self._input_cache.get_evaluation_inputs(
            key,
            lambda: _load_evaluation_inputs(
                self._artifacts,
                data_snapshot.snapshot_id,
                preprocessing_result.forward_return_ref,
                preprocessing_result.mask_data_ref,
                FactorEvaluationError,
            ),
        )

    def _validate_inputs(
        self,
        preprocessing_result: FactorPreprocessingResult,
        preprocessing_result_ref: ObjectRef,
        data_snapshot: DataSnapshot,
    ) -> None:
        if not isinstance(preprocessing_result, FactorPreprocessingResult):
            raise FactorEvaluationError(
                "preprocessing_result must be FactorPreprocessingResult"
            )
        if not isinstance(preprocessing_result_ref, ObjectRef):
            raise FactorEvaluationError("preprocessing_result_ref must be ObjectRef")
        if not isinstance(data_snapshot, DataSnapshot):
            raise FactorEvaluationError("data_snapshot must be DataSnapshot")
        if preprocessing_result.data_snapshot_id != data_snapshot.snapshot_id:
            raise FactorEvaluationError("DATA_SNAPSHOT_MISMATCH")
        if not preprocessing_result.neutralization_applied:
            raise FactorEvaluationError("NEUTRALIZATION_REQUIRED")

    def _reference_factor_values_refs(
        self,
        data_snapshot_id: str,
        *,
        exclude_factor_id: str,
    ) -> tuple[ArtifactRef, ...]:
        if self._reference_provider is None:
            return ()
        refs = self._reference_provider.reference_factor_values_refs(
            data_snapshot_id,
            exclude_factor_id=exclude_factor_id,
        )
        if not isinstance(refs, tuple) or not all(
            isinstance(ref, ArtifactRef) for ref in refs
        ):
            raise FactorEvaluationError("INVALID_REFERENCE_FACTOR_VALUES_REFS")
        return refs


def _evaluation_date_range(
    data_snapshot: DataSnapshot,
    *,
    evaluation_start_date: str | None,
    evaluation_end_date: str | None,
) -> tuple[str, str]:
    """Resolve and validate an inclusive evaluation window."""
    request_start = date.fromisoformat(data_snapshot.request.start_date)
    request_end = date.fromisoformat(data_snapshot.request.end_date)
    start = _parse_evaluation_date(
        (
            data_snapshot.request.start_date
            if evaluation_start_date is None
            else evaluation_start_date
        ),
        "evaluation_start_date",
    )
    end = _parse_evaluation_date(
        (
            data_snapshot.request.end_date
            if evaluation_end_date is None
            else evaluation_end_date
        ),
        "evaluation_end_date",
    )
    if start > end:
        raise FactorEvaluationError("EVALUATION_DATE_RANGE_INVALID")
    if start < request_start or end > request_end:
        raise FactorEvaluationError("EVALUATION_DATE_RANGE_OUT_OF_BOUNDS")
    return start.isoformat(), end.isoformat()


def _parse_evaluation_date(value: str, field_name: str) -> date:
    if type(value) is not str:
        raise FactorEvaluationError(f"{field_name} must use YYYY-MM-DD date format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise FactorEvaluationError(
            f"{field_name} must use YYYY-MM-DD date format"
        ) from error
    if parsed.isoformat() != value:
        raise FactorEvaluationError(f"{field_name} must use YYYY-MM-DD date format")
    return parsed


class AcceptancePolicy:
    """Apply deterministic metric thresholds to an evaluation result."""

    def __init__(
        self,
        *,
        metric_name: str = DEFAULT_ACCEPTANCE_METRIC,
        threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        min_effective_sample_count: int = 100,
        min_effective_date_ratio: float = 0.8,
        min_coverage: float = 0.0,
        use_absolute_value: bool = False,
        requirements: tuple[MetricRequirement, ...] | None = None,
        policy_id: str = ACCEPTANCE_POLICY_ID,
        policy_version: str = ACCEPTANCE_POLICY_VERSION,
    ) -> None:
        """Create a versioned deterministic acceptance policy.

        ``requirements`` enables multi-metric decisions. When it is omitted, the
        original primary-metric threshold behavior is retained for compatibility.
        """
        if type(metric_name) is not str or not metric_name:
            raise FactorEvaluationError("metric_name must be a non-empty string")
        if type(threshold) not in (int, float) or not math.isfinite(float(threshold)):
            raise FactorEvaluationError("threshold must be finite")
        if (
            type(min_effective_sample_count) is not int
            or min_effective_sample_count < 1
        ):
            raise FactorEvaluationError("min_effective_sample_count must be >= 1")
        if (
            type(min_effective_date_ratio) not in (int, float)
            or not math.isfinite(float(min_effective_date_ratio))
            or not 0.0 <= float(min_effective_date_ratio) <= 1.0
        ):
            raise FactorEvaluationError(
                "min_effective_date_ratio must be between 0 and 1"
            )
        if (
            type(min_coverage) not in (int, float)
            or not math.isfinite(float(min_coverage))
            or not 0.0 <= float(min_coverage) <= 1.0
        ):
            raise FactorEvaluationError("min_coverage must be in [0, 1]")
        if requirements is not None:
            if not isinstance(requirements, tuple) or not requirements:
                raise FactorEvaluationError("requirements must be a non-empty tuple")
            if not all(isinstance(rule, MetricRequirement) for rule in requirements):
                raise FactorEvaluationError(
                    "requirements must contain MetricRequirement"
                )
            names = tuple(rule.metric_name for rule in requirements)
            if len(set(names)) != len(names):
                raise FactorEvaluationError("metric requirements must be unique")
        if type(policy_id) is not str or not policy_id:
            raise FactorEvaluationError("policy_id must be a non-empty string")
        if type(policy_version) is not str or not policy_version:
            raise FactorEvaluationError("policy_version must be a non-empty string")
        if type(use_absolute_value) is not bool:
            raise FactorEvaluationError("use_absolute_value must be a boolean")
        self._metric_name = metric_name
        self._threshold = float(threshold)
        self._min_samples = min_effective_sample_count
        self._min_date_ratio = float(min_effective_date_ratio)
        self._min_coverage = float(min_coverage)
        self._use_absolute_value = use_absolute_value
        self._requirements = requirements
        self._policy_id = policy_id
        self._policy_version = policy_version

    def decide(
        self,
        evaluation_result: EvaluationResult,
        evaluation_result_ref: ObjectRef,
    ) -> AcceptanceDecision:
        """Create a deterministic acceptance decision for one evaluation."""
        if not isinstance(evaluation_result, EvaluationResult):
            raise FactorEvaluationError("evaluation_result must be EvaluationResult")
        if not isinstance(evaluation_result_ref, ObjectRef):
            raise FactorEvaluationError("evaluation_result_ref must be ObjectRef")

        if self._requirements is None:
            reasons, accepted = self._legacy_decision(evaluation_result)
        else:
            reasons = self._multi_metric_reasons(evaluation_result)
            if evaluation_result.effective_sample_count < self._min_samples:
                reasons.append("INSUFFICIENT_SAMPLES")
            if evaluation_result.coverage < self._min_coverage:
                reasons.append("LOW_COVERAGE")
            self._append_date_ratio_reason(evaluation_result, reasons)
            accepted = not any(
                reason.startswith(("MISSING_", "FAIL_", "INSUFFICIENT_", "LOW_"))
                for reason in reasons
            )

        return AcceptanceDecision(
            factor_id=evaluation_result.factor_id,
            evaluation_result_ref=evaluation_result_ref,
            accepted=accepted,
            reason_codes=tuple(reasons),
            policy_id=self._policy_id,
            policy_version=self._policy_version,
            schema_version="acceptance-decision/1.0",
        )

    def snapshot(self) -> AcceptancePolicySnapshot:
        """Return the exact policy configuration used by Feedback and Replay."""
        thresholds = (
            (
                AcceptanceMetricThreshold(
                    metric_name=self._metric_name,
                    minimum=self._threshold,
                    maximum=None,
                ),
            )
            if self._requirements is None
            else tuple(
                AcceptanceMetricThreshold(
                    metric_name=requirement.metric_name,
                    minimum=(
                        None
                        if requirement.minimum is None
                        else float(requirement.minimum)
                    ),
                    maximum=(
                        None
                        if requirement.maximum is None
                        else float(requirement.maximum)
                    ),
                )
                for requirement in self._requirements
            )
        )
        return AcceptancePolicySnapshot(
            policy_id=self._policy_id,
            policy_version=self._policy_version,
            acceptance_metric=self._metric_name,
            metric_thresholds=thresholds,
            min_effective_sample_count=self._min_samples,
            min_coverage=self._min_coverage,
            min_effective_date_ratio=self._min_date_ratio,
            use_absolute_value=self._use_absolute_value,
            schema_version="acceptance-policy-snapshot/1.1",
        )

    def _legacy_decision(
        self,
        evaluation_result: EvaluationResult,
    ) -> tuple[list[str], bool]:
        """Retain the original single-metric policy reason-code behavior."""
        reasons: list[str] = []
        metric = _metric_by_name(evaluation_result, self._metric_name)
        if metric is None:
            reasons.append("MISSING_ACCEPTANCE_METRIC")
        if evaluation_result.effective_sample_count < self._min_samples:
            reasons.append("INSUFFICIENT_SAMPLES")
        if evaluation_result.coverage < self._min_coverage:
            reasons.append("LOW_COVERAGE")
        self._append_date_ratio_reason(evaluation_result, reasons)
        if reasons or metric is None:
            return reasons, False
        metric_value = abs(metric.value) if self._use_absolute_value else metric.value
        passed = metric_value >= self._threshold
        return (
            [_threshold_reason(self._metric_name, passed=passed)],
            passed,
        )

    def _append_date_ratio_reason(
        self,
        evaluation_result: EvaluationResult,
        reasons: list[str],
    ) -> None:
        """Append the effective-date coverage result to deterministic reasons."""
        if self._min_date_ratio <= 0.0:
            return
        if evaluation_result.total_date_count is None:
            reasons.append("MISSING_TOTAL_DATE_COUNT")
            return
        if evaluation_result.total_date_count == 0:
            reasons.append("LOW_EFFECTIVE_DATE_RATIO")
            return
        ratio = (
            evaluation_result.effective_date_count / evaluation_result.total_date_count
        )
        if ratio < self._min_date_ratio:
            reasons.append("LOW_EFFECTIVE_DATE_RATIO")

    def _multi_metric_reasons(self, evaluation_result: EvaluationResult) -> list[str]:
        if self._requirements is None:
            raise FactorEvaluationError("multi-metric requirements are not configured")
        reasons: list[str] = []
        for requirement in self._requirements:
            metric = _metric_by_name(evaluation_result, requirement.metric_name)
            token = _metric_token(requirement.metric_name)
            if metric is None:
                reasons.append(f"MISSING_{token}_METRIC")
                continue
            if requirement.minimum is not None:
                reasons.append(
                    f"PASS_{token}_MINIMUM"
                    if metric.value >= requirement.minimum
                    else f"FAIL_{token}_MINIMUM"
                )
            if requirement.maximum is not None:
                reasons.append(
                    f"PASS_{token}_MAXIMUM"
                    if metric.value <= requirement.maximum
                    else f"FAIL_{token}_MAXIMUM"
                )
        return reasons


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([values.size], dtype=np.int64)))
    average_ranks = (starts + 1 + ends) / 2.0
    ranks[order] = np.repeat(average_ranks, ends - starts)
    return ranks


def _build_metrics(
    daily_ic: list[float],
    daily_rank_ic: list[float],
    effective_sample_count: int,
    *,
    coverage: float,
    oos_segment_count: int,
    max_abs_daily_correlation: float,
    correlation_date_count: int,
    orientation: int,
    daily_top20_indices: list[np.ndarray],
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    instruments: np.ndarray,
) -> tuple[MetricValue, ...]:
    ic_values = np.array(daily_ic, dtype=float)
    rank_ic_values = np.array(daily_rank_ic, dtype=float)
    oos_means = _chronological_segment_means(rank_ic_values, oos_segment_count)
    metrics = [
        MetricValue(
            name="ic_mean",
            value=float(np.mean(ic_values)),
            direction="higher_is_better",
            sample_count=effective_sample_count,
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="rank_ic_mean",
            value=float(np.mean(rank_ic_values)),
            direction="higher_is_better",
            sample_count=effective_sample_count,
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="ic_ir",
            value=_information_ratio(ic_values),
            direction="higher_is_better",
            sample_count=len(daily_ic),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="rank_ic_ir",
            value=_information_ratio(rank_ic_values),
            direction="higher_is_better",
            sample_count=len(daily_rank_ic),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="rank_ic_positive_rate",
            value=float(np.mean(rank_ic_values > 0.0)),
            direction="higher_is_better",
            sample_count=len(daily_rank_ic),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="directional_positive_rate",
            value=float(np.mean(orientation * rank_ic_values > 0.0)),
            direction="higher_is_better",
            sample_count=len(daily_rank_ic),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="factor_orientation",
            value=float(orientation),
            direction="neutral",
            sample_count=len(daily_rank_ic),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="rank_ic_oos_mean",
            value=float(np.mean(oos_means)),
            direction="higher_is_better",
            sample_count=len(oos_means),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="rank_ic_oos_min",
            value=float(np.min(oos_means)),
            direction="higher_is_better",
            sample_count=len(oos_means),
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="coverage",
            value=coverage,
            direction="higher_is_better",
            sample_count=effective_sample_count,
            calculation_version=EVALUATOR_VERSION,
        ),
        MetricValue(
            name="factor_max_abs_daily_correlation",
            value=max_abs_daily_correlation,
            direction="lower_is_better",
            sample_count=correlation_date_count,
            calculation_version=EVALUATOR_VERSION,
        ),
    ]
    if rank_ic_values.size >= 2:
        metrics.insert(
            7,
            MetricValue(
                name="rank_ic_std",
                value=float(np.std(rank_ic_values, ddof=1)),
                direction="lower_is_better",
                sample_count=len(daily_rank_ic),
                calculation_version=EVALUATOR_VERSION,
            ),
        )
    top20_returns = _top20_daily_returns(
        orientation,
        daily_top20_indices,
        factor_values,
        forward_returns,
        instruments,
    )
    if top20_returns:
        metrics.append(
            MetricValue(
                name="top20_mean_return",
                value=float(np.mean(np.asarray(top20_returns, dtype=float))),
                direction="higher_is_better",
                sample_count=len(top20_returns),
                calculation_version=EVALUATOR_VERSION,
            )
        )
    return tuple(metrics)


def factor_orientation(rank_ic_mean: float) -> int:
    """Return the frozen sign used to align a factor's direction."""
    if type(rank_ic_mean) not in (int, float) or not math.isfinite(float(rank_ic_mean)):
        raise FactorEvaluationError("rank_ic_mean must be finite")
    return 1 if float(rank_ic_mean) >= 0.0 else -1


def _top20_daily_returns(
    orientation: int,
    daily_indices: list[np.ndarray],
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    instruments: np.ndarray,
) -> list[float]:
    """Select direction-aligned top 20% exposure with deterministic tie breaks."""
    results: list[float] = []
    for indices in daily_indices:
        count = int(indices.size)
        top_count = math.floor(0.2 * count)
        if top_count == 0:
            continue
        day_factor = factor_values[indices]
        ordering_values = -orientation * day_factor
        cutoff = float(np.partition(ordering_values, top_count - 1)[top_count - 1])
        selected = np.flatnonzero(ordering_values < cutoff)
        remaining = top_count - int(selected.size)
        if remaining:
            ties = np.flatnonzero(ordering_values == cutoff)
            tie_order = np.argsort(instruments[indices[ties]], kind="stable")
            selected = np.concatenate((selected, ties[tie_order[:remaining]]))
        results.append(
            float(
                np.mean(
                    forward_returns[indices[selected]],
                )
            )
        )
    return results


def _information_ratio(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if not math.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(np.mean(values) / std)


def _chronological_segment_means(
    values: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    """Return contiguous chronological segment means without mixing observations."""
    segments = tuple(
        segment for segment in np.array_split(values, segment_count) if segment.size
    )
    if not segments:
        raise FactorEvaluationError("NO_VALID_OOS_SEGMENTS")
    return np.array([float(np.mean(segment)) for segment in segments], dtype=float)


def _max_abs_daily_factor_correlation(
    factor_values: np.ndarray,
    inputs: EvaluationInputs,
    evaluation_dates: tuple[str, ...],
    reference_refs: tuple[ArtifactRef, ...],
    artifacts: ArtifactRepository,
    input_cache: EvaluationInputCache,
    *,
    min_cross_section_samples: int,
    min_daily_effective_stock_ratio: float,
) -> tuple[float, int]:
    """Return max absolute Graph-style daily correlation against peer factors."""
    if not reference_refs:
        return 0.0, 0
    max_correlation = 0.0
    max_valid_date_count = 0
    reference_values_batch = input_cache.get_factor_values_batch(
        inputs,
        reference_refs,
        lambda ref: _load_aligned_factor_values(
            artifacts,
            ref,
            inputs,
            "reference_factor_values",
        ),
        max_workers=_REFERENCE_FACTOR_LOAD_WORKERS,
    )
    for reference_values in reference_values_batch:
        daily_pairs = (
            (
                factor_values[inputs.date_slices[date_value]][
                    inputs.eligible[inputs.date_slices[date_value]]
                ],
                reference_values[inputs.date_slices[date_value]][
                    inputs.eligible[inputs.date_slices[date_value]]
                ],
                int(np.count_nonzero(inputs.eligible[inputs.date_slices[date_value]])),
            )
            for date_value in evaluation_dates
        )
        correlation, valid_date_count = mean_daily_cross_sectional_pearson(
            daily_pairs,
            min_cross_section_samples=min_cross_section_samples,
            min_daily_effective_stock_ratio=min_daily_effective_stock_ratio,
        )
        if not math.isfinite(correlation):
            continue
        absolute_correlation = abs(correlation)
        if absolute_correlation > max_correlation or (
            absolute_correlation == max_correlation
            and valid_date_count > max_valid_date_count
        ):
            max_correlation = absolute_correlation
            max_valid_date_count = valid_date_count
    return max_correlation, max_valid_date_count


def _load_aligned_factor_values(
    artifacts: ArtifactRepository,
    ref: ArtifactRef,
    inputs: EvaluationInputs,
    label: str,
) -> np.ndarray:
    table = _read_table(
        artifacts.get_bytes(ref),
        label,
        FactorEvaluationError,
    )
    _validate_table_alignment(table, inputs, FactorEvaluationError)
    return _table_numeric_array(
        table,
        "factor_value",
        FactorEvaluationError,
        missing_ok=True,
    )


def _evaluation_warnings(
    *,
    insufficient_dates: int,
    low_coverage_dates: int,
    degenerate_dates: int,
    min_samples: int,
    min_daily_ratio: float,
) -> tuple[WarningInfo, ...]:
    warnings: list[WarningInfo] = []
    if insufficient_dates:
        warnings.append(
            WarningInfo(
                "INSUFFICIENT_SAMPLES",
                (
                    f"{insufficient_dates} date(s) skipped because finite pairs "
                    f"were below {min_samples}"
                ),
            )
        )
    if low_coverage_dates:
        warnings.append(
            WarningInfo(
                "LOW_DAILY_CROSS_SECTION_COVERAGE",
                (
                    f"{low_coverage_dates} date(s) skipped because finite factor "
                    f"values were below {min_daily_ratio:.1%} of eligible stocks "
                    f"or below {min_samples} stocks"
                ),
            )
        )
    if degenerate_dates:
        warnings.append(
            WarningInfo(
                "DEGENERATE_CROSS_SECTION",
                f"{degenerate_dates} date(s) skipped because correlation was undefined",
            )
        )
    return tuple(warnings)


def _metric_by_name(
    evaluation_result: EvaluationResult,
    metric_name: str,
) -> MetricValue | None:
    for metric in evaluation_result.metrics:
        if metric.name == metric_name:
            return metric
    return None


def _threshold_reason(metric_name: str, *, passed: bool) -> str:
    prefix = "PASS" if passed else "FAIL"
    if metric_name == "rank_ic_mean":
        return f"{prefix}_RANKIC_THRESHOLD"
    if metric_name == "ic_mean":
        return f"{prefix}_IC_THRESHOLD"
    token = _metric_token(metric_name)
    return f"{prefix}_{token}_THRESHOLD"


def _metric_token(metric_name: str) -> str:
    """Convert a metric name to a stable AcceptanceDecision reason-code token."""
    return "".join(
        character for character in metric_name.upper() if character.isalnum()
    )
