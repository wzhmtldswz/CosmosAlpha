"""Offline terminal progress, stream isolation and spawned-worker regressions."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cosmosalpha.agents import AcceptanceDecision, EvaluationResult, FactorSpec
from cosmosalpha.cli import main
from cosmosalpha.core import ArtifactRef
from cosmosalpha.core.logging import (
    LOGGER,
    console_logging,
    contextual,
    log_context,
    progress_stage,
)
from cosmosalpha.factors import build_base_field_catalog, build_base_operator_catalog
from cosmosalpha.llm import AuditedLLMExecutor, LLMCallRecorder, LLMRetryPolicy
from cosmosalpha.llm.client import LLMRequest, LLMResponse
from cosmosalpha.llm.errors import LLMClientError, LLMOutputError
from cosmosalpha.llm.prompts import PromptMessage, PromptRenderResult
from cosmosalpha.pipeline.graph.controller import (
    GraphAttemptExecution,
    GraphAttemptRequest,
)
from cosmosalpha.pipeline.mcts.controller import MCTSController
from cosmosalpha.pipeline.mcts.core import MCTSNode
from cosmosalpha.storage import LocalArtifactRepository, LocalExperimentRepository
from tests.fixtures import graph_runtime_config, mock_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-private-test-value"
DRIVER = """
import sys
import random
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from cosmosalpha.cli import main
from cosmosalpha.core import MCTSConfig
from tests.fixtures import mock_config, graph_runtime_config

if __name__ == "__main__":
    random.seed(7)
    architecture, level, output, fail, resume = sys.argv[1:]
    config = mock_config(Path(output))
    config = replace(config,
        workflow=replace(config.workflow, max_cycles=2),
    )
    if architecture == "graph":
        config = graph_runtime_config(Path(output))
    elif architecture == "parallel_branches":
        config = replace(config,
            architecture=replace(config.architecture, name=architecture, directions=("price", "volume")),
            runtime=replace(config.runtime, max_workers=2),
        )
    elif architecture == "mcts":
        config = replace(config, architecture=replace(config.architecture, name=architecture),
            mcts=MCTSConfig(max_iterations=2, initial_budget=1, budget_increment=0))
    if fail == "yes":
        config = replace(config, prompts=replace(config.prompts, max_input_tokens=1))
    arguments = ["run", "--config", "offline.toml", "--run-id", "progress", "--log-level", level]
    if resume == "yes":
        arguments.append("--resume")
    with patch("cosmosalpha.cli.load_experiment_config", return_value=config):
        raise SystemExit(main(arguments))
"""


def run_cli(
    root: Path,
    architecture: str,
    *,
    level: str = "INFO",
    fail: bool = False,
    resume: bool = False,
    redirect: bool = False,
) -> tuple[subprocess.CompletedProcess[str], str]:
    root.mkdir(parents=True, exist_ok=True)
    script = root / "offline_cli.py"
    script.write_text(DRIVER, encoding="utf-8")
    env = dict(os.environ)
    env.pop("COSMOSALPHA_LLM_API_KEY", None)
    env.pop("COSMOSALPHA_LLM_BASE_URL", None)
    env.update(
        PYTHONPATH=os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONIOENCODING="ascii:backslashreplace" if redirect else "utf-8",
    )
    arguments = [
        sys.executable,
        str(script),
        architecture,
        level,
        str(root / "工作 results"),
        "yes" if fail else "no",
        "yes" if resume else "no",
    ]
    if redirect:
        path = root / "progress.log"
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                arguments,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=stream,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        return result, path.read_text(encoding="utf-8")
    result = subprocess.run(
        arguments,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    return result, result.stderr


@pytest.mark.parametrize(
    "architecture", ("chain", "graph", "parallel_branches", "mcts")
)
def test_formal_cli_progress_and_spawned_branches(
    tmp_path: Path, architecture: str
) -> None:
    result, logs = run_cli(tmp_path, architecture)
    assert result.returncode == 0, result.stdout + logs
    payload = json.loads(result.stdout)
    assert payload["architecture"] == architecture
    assert logs.count("Mining started") == logs.count("Mining completed") == 1
    assert "Data preparation started" in logs and "Data ready stocks=8 dates=4" in logs
    assert "Hypothesis generation started" in logs
    assert "Factor evaluation completed" in logs and "rank_ic=" in logs
    assert "elapsed=" in logs and "run=progress" in logs
    assert all(ord(char) >= 32 or char == "\n" for char in logs)
    assert "Step restored" not in logs
    assert "LLM request started" not in logs
    if architecture == "parallel_branches":
        for index in (1, 2):
            branch = f"branch-{index:03d}"
            assert f"branch={branch} Branch started" in logs
            assert f"branch={branch} cycle={branch}-cycle-002" in logs
            assert f"branch={branch} Branch finished status=completed" in logs
        assert "completed=2 failed=0" in logs
    elif architecture == "graph":
        assert "Graph initialization completed nodes=1/1 expansion_attempts=0" in logs
        assert "attempts=1/1 new_nodes=0 nodes=1 duplicates=1" in logs
        assert "Graph stopped reason=GLOBAL_ATTEMPT_BUDGET_EXHAUSTED" in logs
    elif architecture == "mcts":
        assert "MCTS root ready nodes=1 budget=1" in logs
        assert "MCTS round started index=1/2" in logs
        assert "MCTS stopped reason=" in logs
    else:
        assert "Cycle started index=2/2" in logs
        assert "Chain stopped reason=CYCLE_LIMIT_REACHED cycles=2" in logs


def test_graph_resume_and_redirected_non_ascii_path(tmp_path: Path) -> None:
    first, _ = run_cli(tmp_path, "graph")
    assert first.returncode == 0
    resumed, logs = run_cli(tmp_path, "graph", resume=True, redirect=True)
    assert resumed.returncode == 0, resumed.stdout + logs
    assert json.loads(resumed.stdout)["resumed"]
    assert "Graph restored nodes=1 attempts=1/1" in logs
    assert "Graph initialization started" not in logs
    assert "Expression generation started" not in logs
    assert "Mining completed" in logs and "Logging error" not in logs
    assert "\\u5de5\\u4f5c" in logs


def test_graph_failure_before_loop_is_reported_without_failing_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = graph_runtime_config(tmp_path)

    def failed_batch(
        requests: tuple[GraphAttemptRequest, ...], **_: object
    ) -> tuple[GraphAttemptExecution, ...]:
        return tuple(
            GraphAttemptExecution(request, None, "WORKFLOW_METADATA_FAILED")
            for request in requests
        )

    with (
        patch("cosmosalpha.cli.load_experiment_config", return_value=config),
        patch(
            "cosmosalpha.pipeline.runtime.LocalLoopAttemptExecutor.run_batch",
            side_effect=failed_batch,
        ),
    ):
        assert main(["run", "--config", "mock.toml", "--run-id", "failed-attempt"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["architecture"] == "graph"
    assert "Graph expansion failed" in output.err
    assert "reason=WORKFLOW_METADATA_FAILED" in output.err
    assert "Mining completed" in output.err and "Mining terminated" not in output.err


@pytest.mark.parametrize("architecture", ("graph", "parallel_branches"))
def test_recoverable_and_terminal_failure_logs_exit_cleanly(
    tmp_path: Path, architecture: str
) -> None:
    result, logs = run_cli(tmp_path, architecture, fail=True)
    assert result.returncode == 1, result.stdout + logs
    assert json.loads(result.stdout)["error_code"] == "RUNTIME_FAILED"
    assert "WARNING" in logs and "ERROR" in logs
    assert "Mining terminated" in logs and "Mining completed" not in logs
    assert "Hypothesis generation completed" not in logs
    if architecture == "parallel_branches":
        assert logs.count("Branch finished status=failed") == 2


@pytest.mark.parametrize("level", ("DEBUG", "WARNING", "ERROR"))
def test_cli_log_levels_keep_stdout_json(tmp_path: Path, level: str) -> None:
    result, logs = run_cli(tmp_path, "chain", level=level)
    assert result.returncode == 0, result.stdout + logs
    assert json.loads(result.stdout)["architecture"] == "chain"
    if level == "DEBUG":
        assert "DEBUG" in logs and "experiment_publish started" in logs
    else:
        assert not logs


def business_signature(run_directory: Path) -> list[tuple[object, ...]]:
    objects = LocalExperimentRepository(run_directory)
    rows = []
    for record in objects.list_experiments():
        expressions = tuple(
            objects.get_object(ref, FactorSpec).expression for ref in record.factor_refs
        )
        metrics = tuple(
            tuple(
                (m.name, m.value)
                for m in objects.get_object(ref, EvaluationResult).metrics
            )
            for ref in record.evaluation_result_refs
        )
        decisions = []
        for ref in record.acceptance_decision_refs:
            decision = objects.get_object(ref, AcceptanceDecision)
            decisions.append((decision.accepted, decision.reason_codes))
        rows.append(
            (
                record.cycle_id,
                record.status.value,
                expressions,
                metrics,
                tuple(decisions),
            )
        )
    return sorted(rows)


@pytest.mark.parametrize(
    "architecture", ("chain", "graph", "parallel_branches", "mcts")
)
def test_logging_level_does_not_change_research_results(
    tmp_path: Path, architecture: str
) -> None:
    signatures = []
    for level in ("INFO", "ERROR"):
        result, logs = run_cli(tmp_path / level, architecture, level=level)
        assert result.returncode == 0, result.stdout + logs
        signatures.append(
            business_signature(Path(json.loads(result.stdout)["run_directory"]))
        )
    assert signatures[0] == signatures[1]


def test_cli_restores_application_logging_and_preflight_stays_quiet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = mock_config(tmp_path)
    monkeypatch.setattr("cosmosalpha.cli.load_experiment_config", lambda _: config)
    root = logging.getLogger()
    before = (
        LOGGER.level,
        LOGGER.propagate,
        tuple(LOGGER.handlers),
        tuple(root.handlers),
        root.level,
    )
    assert (
        main(["run", "--config", "mock.toml", "--run-id", "check", "--preflight-only"])
        == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["ready"] and not output.err
    for run_id in ("first", "second"):
        assert main(["run", "--config", "mock.toml", "--run-id", run_id]) == 0
        output = capsys.readouterr()
        assert json.loads(output.out)["run_id"] == run_id
        assert output.err.count("Mining started") == 1
        assert (
            LOGGER.level,
            LOGGER.propagate,
            tuple(LOGGER.handlers),
            tuple(root.handlers),
            root.level,
        ) == before
    with patch("cosmosalpha.cli.run_formal_runtime", side_effect=RuntimeError(SECRET)):
        assert main(["run", "--config", "mock.toml", "--run-id", "failed"]) == 1
    output = capsys.readouterr()
    assert SECRET not in output.err
    assert (
        LOGGER.level,
        LOGGER.propagate,
        tuple(LOGGER.handlers),
        tuple(root.handlers),
        root.level,
    ) == before


def test_thread_context_and_single_line_safe_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    barrier = Barrier(2)

    @contextual
    def work(*, run_id: str, cycle_id: str) -> None:
        barrier.wait(timeout=5)
        with progress_stage("Work"):
            assert "Work started" in stream.getvalue()
        with pytest.raises(RuntimeError), progress_stage("Broken"):
            raise RuntimeError(SECRET)

    with console_logging("INFO"), log_context(run="parent"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(work, run_id=f"run-{i}", cycle_id=f"cycle-{i}")
                for i in (1, 2)
            ]
            for future in futures:
                future.result(timeout=5)
        LOGGER.info("Parent restored")
        LOGGER.info("No multiline\nmessage\r\x1b[31m")
    logs = stream.getvalue()
    for line in logs.splitlines():
        if "run=run-1" in line:
            assert "cycle=cycle-1" in line and "cycle-2" not in line
        if "run=run-2" in line:
            assert "cycle=cycle-2" in line and "cycle-1" not in line
    assert "run=parent Parent restored" in logs
    assert "Broken completed" not in logs and SECRET not in logs
    assert "\\x0a" in logs and "\x1b" not in logs


def llm_request() -> LLMRequest:
    return LLMRequest(
        prompt=PromptRenderResult(
            prompt_render_id="logging-request",
            prompt_id="hypothesis",
            template_version="1.0",
            template_checksum="a" * 64,
            input_schema="test/1.0",
            output_schema="test/1.0",
            messages=(PromptMessage("user", SECRET),),
            output_json_schema={},
            estimated_input_tokens=1,
            token_estimation_method="test",
            truncation_report="none",
            schema_version="prompt-render-result/1.0",
        ),
        model_alias="offline",
        max_output_tokens=10,
        response_format="json_object",
    )


@pytest.mark.parametrize("kind", ("timeout", "rate_limit", "format", "exhausted"))
def test_llm_retries_report_before_wait_without_exposing_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    request = llm_request()
    attempts = 0
    parses = 0
    sleeps: list[float] = []

    def complete(active: LLMRequest) -> LLMResponse:
        nonlocal attempts
        attempts += 1
        assert "LLM request started" in stream.getvalue()
        if kind != "format" and (attempts == 1 or kind == "exhausted"):
            code = "LLM_RATE_LIMITED" if kind == "rate_limit" else "LLM_TIMEOUT"
            raise LLMClientError(f"{code}: {SECRET}", raw_response=SECRET)
        return LLMResponse(
            content='{"ok":true}',
            model_alias=active.model_alias,
            prompt_render_id=active.prompt.prompt_render_id,
            finish_reason="stop",
            input_tokens=3,
            output_tokens=2,
            raw_response=SECRET,
        )

    def parse(content: str) -> dict[str, bool]:
        nonlocal parses
        parses += 1
        if kind == "format" and parses == 1:
            raise LLMOutputError(SECRET)
        return json.loads(content)

    def sleep(delay: float) -> None:
        assert "LLM retry" in stream.getvalue()
        sleeps.append(delay)

    executor = AuditedLLMExecutor(
        client=SimpleNamespace(complete=complete),
        recorder=LLMCallRecorder(LocalArtifactRepository(tmp_path)),
        object_repository=LocalExperimentRepository(tmp_path),
        retry_policy=LLMRetryPolicy(1, 1, 0, 1, 0.25, 0.25, max_total_retries=1),
        sleep=sleep,
        jitter=lambda delay: delay,
    )
    with console_logging("DEBUG"):
        if kind == "exhausted":
            with pytest.raises(LLMClientError):
                executor.complete_and_parse(request, parse)
        else:
            assert executor.complete_and_parse(request, parse) == {"ok": True}
    logs = stream.getvalue()
    assert attempts == 2 and sleeps == [0.25]
    assert logs.count("LLM retry") == 1
    assert "wait=0.25s" in logs and SECRET not in logs
    assert (
        "retryable=false" in logs
        if kind == "exhausted"
        else "LLM request completed" in logs
    )


def mcts_controller(tmp_path: Path) -> MCTSController:
    def complete(_: LLMRequest) -> LLMResponse:
        raise LLMClientError(SECRET)

    return MCTSController(
        loop=SimpleNamespace(),
        experiment_repository=LocalExperimentRepository(tmp_path),
        artifact_repository=LocalArtifactRepository(tmp_path),
        field_catalog=build_base_field_catalog(),
        operator_catalog=build_base_operator_catalog(),
        llm_client=SimpleNamespace(complete=complete),
    )


def test_mcts_independent_requests_and_fallback_are_visible(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = mcts_controller(tmp_path)
    parent = MCTSNode(factor_id="parent", expression="close", rank_ic_mean=0.0)
    with (
        caplog.at_level("INFO", logger="cosmosalpha"),
        patch.object(controller, "_sample_few_shot_paper", return_value=[]),
        patch(
            "cosmosalpha.pipeline.mcts.controller.generate_portrait", return_value=None
        ),
    ):
        assert controller._generate_alpha_portrait(parent, "effectiveness") is None
        assert controller._step_a_suggest(parent) == []
    logs = caplog.text
    assert "MCTS factor portrait started" in logs
    assert "using suggestion fallback" in logs
    assert (
        "MCTS suggestion started" in logs and "continuing without suggestions" in logs
    )
    assert "MCTS suggestion completed" not in logs and SECRET not in logs


def test_mcts_early_stop_reports_only_executed_rounds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = mcts_controller(tmp_path)
    root = MCTSNode(
        factor_id="root", expression="close", rank_ic_mean=0.0, overall_score=0.0
    )
    child = MCTSNode(
        factor_id="child", expression="volume", rank_ic_mean=0.0, overall_score=0.0
    )
    with (
        caplog.at_level("INFO", logger="cosmosalpha"),
        patch.object(controller, "_admit_to_fzoo", return_value=False),
        patch.object(controller, "_expand", return_value=child),
        patch.object(controller, "_persist_tree"),
    ):
        controller.run(
            root=root,
            data_snapshot=None,
            base_context=None,
            config_snapshot_ref=ArtifactRef(
                "test",
                "test",
                "artifact://test",
                "text/plain",
                "b" * 64,
                1,
                "artifact-ref/1.0",
            ),
            code_version="test",
            run_id="early",
            max_iterations=3,
            initial_budget=1,
            budget_increment=0,
        )
    assert (
        "MCTS stopped reason=BUDGET_EXHAUSTED rounds=1 nodes=2 budget=0" in caplog.text
    )
    assert "MCTS round started index=2" not in caplog.text
