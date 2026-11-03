"""Immutable storage, concurrent publication and recovery."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from cosmosalpha.pipeline.smoke import _research_context, run_fixed_replay_smoke_loop
from cosmosalpha.storage import (
    IntegrityError,
    LocalArtifactRepository,
    LocalExperimentRepository,
    ObjectConflictError,
    SqliteFactorLibrary,
    SqliteWorkflowMetadataRepository,
)


def test_artifact_identity_and_integrity(tmp_path: Path) -> None:
    repository = LocalArtifactRepository(tmp_path)
    first = repository.put_bytes(b"fixed", kind="sample", media_type="text/plain")
    second = repository.put_bytes(b"fixed", kind="sample", media_type="text/plain")
    assert first.artifact_id == second.artifact_id
    assert repository.get_bytes(first) == b"fixed"
    repository._artifact_path(first.artifact_id).write_bytes(b"broken")
    with pytest.raises(IntegrityError):
        repository.get_bytes(first)


def test_concurrent_objects_are_immutable_and_idempotent(tmp_path: Path) -> None:
    repository = LocalExperimentRepository(tmp_path)
    context = _research_context()
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(
            pool.map(
                lambda _: repository.put_object("research_context", "fixed", context),
                range(8),
            )
        )
    assert all(ref == refs[0] for ref in refs)
    assert repository.get_object(refs[0], type(context)) == context
    with pytest.raises(ObjectConflictError):
        repository.put_object(
            "research_context", "fixed", replace(context, research_goal="changed")
        )


def test_sqlite_repository_connections_close_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def track_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connect)
    objects = LocalExperimentRepository(tmp_path)
    artifacts = LocalArtifactRepository(tmp_path)
    workflow = SqliteWorkflowMetadataRepository.initialize(
        tmp_path / "workflow.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )
    factor_library = SqliteFactorLibrary.initialize(
        tmp_path / "factor-library.sqlite3",
        object_verifier=objects,
        artifact_verifier=artifacts,
    )

    assert workflow.latest_successful_attempt("run", "cycle", "step") is None
    assert factor_library.list_entries() == ()

    with pytest.raises(RuntimeError, match="rollback"):
        with workflow._connection() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, payload) VALUES (?, ?)",
                ("rollback", "{}"),
            )
            raise RuntimeError("rollback")
    verification = original_connect(tmp_path / "workflow.sqlite3")
    try:
        assert (
            verification.execute(
                "SELECT COUNT(*) FROM runs WHERE run_id = ?", ("rollback",)
            ).fetchone()[0]
            == 0
        )
    finally:
        verification.close()

    assert len(connections) == 5
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")

    (tmp_path / "workflow.sqlite3").unlink()
    (tmp_path / "factor-library.sqlite3").unlink()


def test_replay_resume_reuses_completed_experiment(tmp_path: Path) -> None:
    first = run_fixed_replay_smoke_loop(tmp_path)
    second = run_fixed_replay_smoke_loop(tmp_path)
    assert second.result.resumed
    assert first.result.experiment_ref == second.result.experiment_ref
    assert first.result.candidate_results == second.result.candidate_results
