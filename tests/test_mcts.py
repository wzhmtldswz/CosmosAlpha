"""Offline MCTS extension, repair and tree persistence."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from cosmosalpha.core import (
    ConfigSnapshot,
    CoreValidationError,
    MCTSConfig,
    ObjectRef,
    SerializationError,
    from_json_value,
    load_experiment_config,
    load_experiment_config_values,
    to_json_value,
)
from cosmosalpha.pipeline.mcts.controller import MCTSController
from cosmosalpha.pipeline.mcts.smoke import run_mcts_smoke
from cosmosalpha.pipeline.runtime import run_formal_runtime
from cosmosalpha.storage import LocalExperimentRepository
from tests.fixtures import mock_config


def test_mcts_example_and_omitted_settings_keep_formal_defaults() -> None:
    values = tomllib.loads(Path("configs/mcts_experiment.toml").read_text())
    public = load_experiment_config_values(values)
    assert public.mcts == MCTSConfig()
    values.pop("mcts")
    values["schema_version"] = "experiment-config/1.5"
    migrated = load_experiment_config_values(values)
    assert migrated == public


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("max_iterations", 0),
        ("exploration_weight", -0.1),
        ("exploration_weight", float("inf")),
        ("initial_budget", 0),
        ("budget_increment", -1),
        ("breakthrough_relax", 0.0),
        ("breakthrough_relax", 1.1),
        ("unknown_search_setting", 1),
    ),
)
def test_invalid_mcts_settings_fail_during_config_loading(
    name: str, value: object
) -> None:
    values = tomllib.loads(Path("configs/mcts_experiment.toml").read_text())
    values["mcts"][name] = value
    with pytest.raises((CoreValidationError, SerializationError), match="mcts"):
        load_experiment_config_values(values)


@pytest.mark.parametrize("iterations", [1, 2])
def test_formal_mcts_passes_search_settings_and_persists_config(
    tmp_path: Path, iterations: int
) -> None:
    public = load_experiment_config("configs/mcts_experiment.toml")
    settings = MCTSConfig(
        max_iterations=iterations,
        exploration_weight=1.7,
        initial_budget=2,
        budget_increment=0,
        breakthrough_relax=0.8,
    )
    config = replace(
        mock_config(tmp_path),
        architecture=public.architecture,
        mcts=settings,
    )
    with (
        patch(
            "cosmosalpha.pipeline.runtime.MCTSController", wraps=MCTSController
        ) as constructor,
        patch.object(
            MCTSController, "run", autospec=True, side_effect=MCTSController.run
        ) as search,
    ):
        result = run_formal_runtime(config, run_id="mcts", resume=False)
    assert constructor.call_args.kwargs["exploration_weight"] == 1.7
    passed = search.call_args.kwargs
    for name in (
        "max_iterations",
        "initial_budget",
        "budget_increment",
        "breakthrough_relax",
    ):
        assert passed[name] == getattr(settings, name)
    assert result.architecture_name == "mcts"
    assert result.experiment_record_refs
    objects = LocalExperimentRepository(tmp_path / "mcts")
    assert result.mcts_tree_ref is not None and objects.exists(result.mcts_tree_ref)
    (snapshot_id,) = objects.list_object_ids("config_snapshot")
    snapshot = objects.get_object(
        ObjectRef("config_snapshot", snapshot_id, "config-snapshot/1.0"),
        ConfigSnapshot,
    )
    assert snapshot.config.mcts == settings
    assert from_json_value(to_json_value(snapshot), ConfigSnapshot) == snapshot


def test_mcts_replay_repairs_and_persists_tree(tmp_path: Path) -> None:
    result = run_mcts_smoke(tmp_path)
    summary = json.loads(result.summary_json())
    assert summary["tree_size"] == 3
    assert summary["iterations"] == 2
    assert {"mcts_factor_design", "factor_expression", "expression_repair"} <= set(
        summary["prompt_ids"]
    )
    assert (tmp_path / "objects/mcts_tree/mcts-tree.json").is_file()
