"""Bounded Graph replay, publication and resume contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cosmosalpha.agents import AcceptanceDecision, EvaluationResult
from cosmosalpha.pipeline.graph import GraphResult
from cosmosalpha.pipeline.runtime import run_formal_runtime
from cosmosalpha.pipeline.smoke import run_fixed_replay_smoke_loop
from cosmosalpha.storage import LocalExperimentRepository
from tests.fixtures import graph_runtime_config, mock_config


def test_graph_cold_start_duplicate_budget_and_resume(tmp_path: Path) -> None:
    config = graph_runtime_config(tmp_path)
    result = run_formal_runtime(config, run_id="graph", resume=False)
    assert result.architecture_name == "graph"
    assert result.graph_result_ref is not None
    objects = LocalExperimentRepository(tmp_path / "graph")
    graph = objects.get_object(result.graph_result_ref, GraphResult)
    assert len(graph.graph.nodes) == 1
    assert graph.attempts_used == 1
    assert result.factor_library_entry_refs == ()
    assert Path(result.factor_library_csv_path).is_file()
    resumed = run_formal_runtime(config, run_id="graph", resume=True)
    assert resumed.resumed
    assert resumed.graph_result_ref == result.graph_result_ref
    assert resumed.experiment_record_refs == result.experiment_record_refs


def test_zero_rank_ic_is_rejected_at_the_production_threshold(tmp_path: Path) -> None:
    result = run_formal_runtime(mock_config(tmp_path), run_id="rejected", resume=False)
    objects = LocalExperimentRepository(tmp_path / "rejected")
    evaluation = objects.get_object(result.evaluation_result_refs[0], EvaluationResult)
    assert next(
        m.value for m in evaluation.metrics if m.name == "rank_ic_mean"
    ) == pytest.approx(0)
    decision = objects.get_object(
        result.acceptance_decision_refs[0], AcceptanceDecision
    )
    assert not decision.accepted
    assert "FAIL_RANKIC_THRESHOLD" in decision.reason_codes
    assert not result.factor_library_entry_refs


def test_explicitly_accepted_fixed_sample_is_published(tmp_path: Path) -> None:
    result = run_fixed_replay_smoke_loop(tmp_path)
    assert result.result.experiment_record is not None
    assert result.result.experiment_record.status.value == "accepted"
    objects = LocalExperimentRepository(tmp_path)
    ref = result.result.candidate_results[0].acceptance_decision_ref
    assert ref is not None
    assert objects.get_object(ref, AcceptanceDecision).accepted


def test_graph_acceptance_gate_mismatch_is_not_bypassed(tmp_path: Path) -> None:
    config = graph_runtime_config(tmp_path)
    config = replace(
        config, evaluation=replace(config.evaluation, acceptance_threshold=0.02)
    )
    with pytest.raises(Exception, match="GRAPH_ACCEPTANCE_GATE_MISMATCH"):
        run_formal_runtime(config, run_id="mismatch", resume=False)
