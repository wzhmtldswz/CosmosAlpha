"""Immutable storage, concurrent publication and recovery."""

from __future__ import annotations

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


def test_replay_resume_reuses_completed_experiment(tmp_path: Path) -> None:
    first = run_fixed_replay_smoke_loop(tmp_path)
    second = run_fixed_replay_smoke_loop(tmp_path)
    assert second.result.resumed
    assert first.result.experiment_ref == second.result.experiment_ref
    assert first.result.candidate_results == second.result.candidate_results
