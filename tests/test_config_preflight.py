"""Public configuration, preflight and provider contracts."""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from cosmosalpha.core import (
    ConfigSnapshot,
    CoreValidationError,
    ObjectRef,
    SerializationError,
    from_json_value,
    load_experiment_config,
    load_experiment_config_values,
    to_json_value,
)
from cosmosalpha.data.qlib_industry import inspect_industry_provider
from cosmosalpha.llm import PromptError, PromptTemplateRegistry, resolve_prompt_manifest
from cosmosalpha.pipeline.graph import GraphConfig
from cosmosalpha.pipeline.preflight import run_preflight
from cosmosalpha.pipeline.runtime import RuntimeBuilder, run_formal_runtime
from cosmosalpha.storage import LocalExperimentRepository
from tests.fixtures import mock_config, qlib_market_config


def test_public_default_and_offline_fixture_are_separate(tmp_path: Path) -> None:
    public = load_experiment_config()
    assert public.architecture.name == "graph"
    assert public.data.provider == "qlib"
    assert public.llm.model_alias == "deepseek-flash"
    assert public.prompts.manifest == "builtin:v2"
    assert public.runtime.max_workers == 4
    assert public.graph is not None
    graph = from_json_value(public.graph, GraphConfig)
    assert graph.initialization.seed_mode == "cold"
    assert graph.search.max_expansion_attempts == 100
    config = mock_config(tmp_path)
    assert config.data.provider == "mock"
    assert run_preflight(config, run_id="offline").ready
    assert not (tmp_path / "offline").exists()
    assert {p.name for p in Path("configs").glob("*.toml")} == {
        "chain_experiment.toml",
        "graph_experiment.toml",
        "lightgbm_integration.example.toml",
        "mcts_experiment.toml",
        "parallel_branches_experiment.toml",
    }


def test_mining_examples_share_research_settings() -> None:
    graph = tomllib.loads(Path("configs/graph_experiment.toml").read_text())
    for architecture in ("chain", "parallel_branches", "mcts"):
        path = Path(f"configs/{architecture}_experiment.toml")
        values = tomllib.loads(path.read_text())
        config = load_experiment_config(path)
        assert config.architecture.name == architecture
        assert config.graph is None
        assert (config.mcts is not None) == (architecture == "mcts")
        assert config.runtime.max_workers == (
            2 if architecture == "parallel_branches" else 1
        )
        assert len(config.architecture.directions) == (
            2 if architecture == "parallel_branches" else 0
        )
        assert config.runtime.random_seed == graph["runtime"]["random_seed"]
        for section in graph.keys() - {"architecture", "graph", "runtime"}:
            assert values[section] == graph[section], (architecture, section)


@pytest.mark.parametrize(
    ("section", "name"),
    (
        ("runtime", "slurm_enabled"),
        ("runtime", "run_mode"),
        ("universe", "blacklist_enabled"),
        ("universe", "exclude_financial"),
        ("workflow", "replay_enabled"),
    ),
)
def test_unused_switches_are_rejected(section: str, name: str) -> None:
    values = to_json_value(load_experiment_config())
    values[section][name] = False
    with pytest.raises(SerializationError, match="unknown field"):
        load_experiment_config_values(values)
    for path in Path("configs").glob("*.toml"):
        public = tomllib.loads(path.read_text())
        assert name not in public.get(section, {})
        assert all(
            "provider_uri_fingerprint" not in public["qlib"][source]
            for source in ("market", "exposure", "mask")
        )


@pytest.mark.parametrize("source", ("market", "exposure", "mask"))
def test_source_fingerprint_fields_are_rejected(source: str) -> None:
    values = to_json_value(load_experiment_config())
    values["qlib"][source]["provider_uri_fingerprint"] = "obsolete-source-label"
    with pytest.raises(SerializationError, match="unknown field"):
        load_experiment_config_values(values)


def test_qlib_smoke_config_needs_no_source_fingerprints(tmp_path: Path) -> None:
    from cosmosalpha.data.qlib_smoke import load_qlib_snapshot_smoke_config

    config = qlib_market_config(tmp_path)
    request = {
        name: getattr(config.data, name)
        for name in (
            "market",
            "start_date",
            "end_date",
            "warmup_start_date",
            "frequency",
            "dataset_version",
            "price_adjustment",
        )
    }
    request["universe"] = config.universe.name
    request["fields"] = [field.field_name for field in config.qlib.market.fields]
    lines = ['instruments = ["SH600000"]', "[request]"]
    lines.extend(f"{name} = {json.dumps(value)}" for name, value in request.items())
    for name in ("market", "exposure", "mask"):
        source = getattr(config.qlib, name)
        lines.extend(
            [f"[sources.{name}]", f"provider_uri = {json.dumps(source.provider_uri)}"]
        )
        if name == "market":
            lines.extend(
                [
                    f"region = {json.dumps(config.qlib.region)}",
                    f"trading_calendar_version = {json.dumps(config.data.trading_calendar_version)}",
                ]
            )
        else:
            lines.append(f"[sources.{name}.fields]")
            lines.extend(
                f"{field.field_name} = {json.dumps(field.expression)}"
                for field in source.fields
            )
    path = tmp_path / "smoke.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_qlib_snapshot_smoke_config(path)
    assert loaded.request.dataset_version == config.data.dataset_version
    assert set(to_json_value(loaded.exposure_source)) == {"provider_uri", "fields"}
    assert set(to_json_value(loaded.mask_source)) == {"provider_uri", "fields"}


def test_runtime_starts_without_market_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cosmosalpha.pipeline.preflight as preflight

    config = qlib_market_config(tmp_path)
    find_spec = preflight.importlib.util.find_spec
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: object() if name == "qlib" else find_spec(name),
    )
    report = run_preflight(config, run_id="resolved-market")
    assert report.ready, report.issues
    RuntimeBuilder(config, run_id="resolved-market").build(resume=False)
    objects = LocalExperimentRepository(tmp_path / "resolved-market")
    snapshot = objects.get_object(
        ObjectRef("config_snapshot", report.config_snapshot_id, "config-snapshot/1.0"),
        ConfigSnapshot,
    )
    sources = to_json_value(snapshot.config.qlib)
    assert all(
        set(sources[source]) == {"provider_uri", "fields"}
        for source in ("market", "exposure", "mask")
    )
    assert not (
        Path(config.qlib.market.provider_uri) / "cosmosalpha_market_manifest.json"
    ).exists()


def test_qlib_provider_needs_no_manifest_and_tracks_content(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from cosmosalpha.data import (
        QlibFeatureSource,
        QlibFieldMapping,
        QlibMarketDataProvider,
    )
    from cosmosalpha.factors import FactorCalculator
    from cosmosalpha.pipeline.runtime import _market_data_request
    from cosmosalpha.storage import LocalArtifactRepository

    config = qlib_market_config(tmp_path)
    config = replace(config, data=replace(config.data, end_date="2020-01-04"))
    price_offset = 0.0
    excluded_day = 1
    industry_code = 1

    def load(
        source: QlibFeatureSource,
        *,
        region: str,
        instruments: object,
        start_time: str,
        end_time: str,
    ) -> object:
        rows = []
        for day in range(4):
            values = {
                "open": 10.0 + day + price_offset,
                "close": 10.0 + day + price_offset,
                "high": 11.0 + day + price_offset,
                "low": 9.0 + day + price_offset,
                "vwap": 10.0 + day + price_offset,
                "volume": 100.0,
                "turn": 0.1,
                "free_turn": 0.1,
                "returns": 0.1,
                "industry_id": industry_code,
                "log_mkt_cap": 23.0,
                "universe_mask": True,
                "eligible_mask": day != excluded_day,
                "entry_tradable_mask": True,
                "exit_tradable_mask": True,
            }
            rows.append([values[field.field_name] for field in source.fields])
        payload = {
            "index": [(f"2020-01-0{day + 1}", "SH600000") for day in range(4)],
            "data": rows,
        }
        return SimpleNamespace(
            columns=tuple(field.expression for field in source.fields),
            index=SimpleNamespace(names=("datetime", "instrument")),
            to_dict=lambda orient: payload,
        )

    def sidecar(name: str) -> QlibFeatureSource:
        source = getattr(config.qlib, name)
        return QlibFeatureSource(
            source.provider_uri,
            tuple(
                QlibFieldMapping(field.field_name, field.expression)
                for field in source.fields
            ),
        )

    artifacts = LocalArtifactRepository(tmp_path / "snapshot")
    provider = QlibMarketDataProvider(
        artifacts,
        provider_uri=config.qlib.market.provider_uri,
        trading_calendar_version=config.data.trading_calendar_version,
        exposure_source=sidecar("exposure"),
        mask_source=sidecar("mask"),
        instruments=("SH600000",),
        frame_loader=SimpleNamespace(load=load),
    )
    snapshot = provider.load_snapshot(_market_data_request(config))
    assert not (
        Path(config.qlib.market.provider_uri) / "cosmosalpha_market_manifest.json"
    ).exists()
    assert any(
        issue.code == "INDUSTRY_NOT_PIT_CERTIFIED"
        for issue in snapshot.quality_report.issues
    )
    assert FactorCalculator(artifacts).load_frame(
        snapshot, ("close",)
    ).eligible_mask.tolist() == [True, False, True, True]
    price_offset = 1.0
    changed = provider.load_snapshot(_market_data_request(config))
    assert changed.provider_uri_fingerprint == snapshot.provider_uri_fingerprint
    assert (
        changed.market_data.data_ref.checksum != snapshot.market_data.data_ref.checksum
    )
    assert changed.snapshot_id != snapshot.snapshot_id
    excluded_day = -1
    changed_mask = provider.load_snapshot(_market_data_request(config))
    assert (
        changed_mask.market_data.data_ref.checksum
        == changed.market_data.data_ref.checksum
    )
    assert (
        changed_mask.mask_data.mask_data_ref.checksum
        != changed.mask_data.mask_data_ref.checksum
    )
    assert changed_mask.snapshot_id != changed.snapshot_id
    industry_code = 2
    changed_exposure = provider.load_snapshot(_market_data_request(config))
    assert (
        changed_exposure.market_data.data_ref.checksum
        == changed_mask.market_data.data_ref.checksum
    )
    assert (
        changed_exposure.exposure_data.data_ref.checksum
        != changed_mask.exposure_data.data_ref.checksum
    )
    assert changed_exposure.snapshot_id != changed_mask.snapshot_id


def test_builtin_and_explicit_manifest_loading(tmp_path: Path) -> None:
    builtin = PromptTemplateRegistry.from_manifest("builtin:v2")
    explicit = PromptTemplateRegistry.from_manifest(
        resolve_prompt_manifest("builtin:v2")
    )
    assert builtin.prompt_ids() == explicit.prompt_ids()
    assert len(builtin.prompt_ids()) == 7
    assert (
        builtin.get("graph_factor_design").checksum
        == explicit.get("graph_factor_design").checksum
    )
    custom = tmp_path / "custom"
    custom.mkdir()
    builtin_path = resolve_prompt_manifest("builtin:v2")
    for template in builtin_path.parent.glob("*.yaml"):
        shutil.copyfile(template, custom / template.name)
    (custom / "manifest.yaml").rename(custom / "catalog.json")
    external = PromptTemplateRegistry.from_manifest(custom / "catalog.json")
    assert external.prompt_ids() == builtin.prompt_ids()
    with pytest.raises(PromptError):
        PromptTemplateRegistry.from_manifest(tmp_path / "missing.yaml")
    path = tmp_path / "manifest.yaml"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PromptError, match="PROMPT_VERSION_MISMATCH"):
        PromptTemplateRegistry.from_manifest(path)


def test_preflight_reports_missing_prompt_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = mock_config(tmp_path)
    monkeypatch.delenv("COSMOSALPHA_LLM_API_KEY", raising=False)
    changed = replace(
        config, llm=replace(config.llm, provider="openai-compatible", replay_only=False)
    )
    report = run_preflight(changed, run_id="missing-key")
    assert "LLM_API_KEY_NOT_CONFIGURED" in {i.code for i in report.issues}
    changed = replace(
        config, prompts=replace(config.prompts, manifest=str(tmp_path / "missing.yaml"))
    )
    report = run_preflight(changed, run_id="missing-prompt")
    assert "PROMPT_MANIFEST_MISSING" in {i.code for i in report.issues}
    assert not run_preflight(config, run_id="../unsafe").ready
    with pytest.raises(CoreValidationError):
        replace(config.architecture, name="unregistered")


def test_industry_manifest_accepts_quality_metadata_without_source_labels(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="INDUSTRY_MANIFEST_MISSING"):
        inspect_industry_provider(str(tmp_path))
    path = tmp_path / "cosmosalpha_industry_manifest.json"
    manifest = {
        "build_status": "complete",
        "point_in_time_certified": False,
        "usable_for_industry_calculations": True,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert inspect_industry_provider(str(tmp_path))["point_in_time_certified"] is False
    manifest["point_in_time_certified"] = True
    manifest["provider_fingerprint"] = "legacy-label-ignored"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert inspect_industry_provider(str(tmp_path))["point_in_time_certified"] is True


@pytest.mark.parametrize(
    "failure",
    ("malformed", "incomplete", "marker", "missing_pit", "invalid_pit", "unusable"),
)
def test_industry_manifest_keeps_quality_checks(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "cosmosalpha_industry_manifest.json"
    manifest = {"build_status": "complete", "point_in_time_certified": False}
    if failure == "incomplete":
        manifest["build_status"] = "building"
    elif failure == "marker":
        (tmp_path / "industry_build_incomplete.json").write_text("{}", encoding="utf-8")
    elif failure == "missing_pit":
        manifest.pop("point_in_time_certified")
    elif failure == "invalid_pit":
        manifest["point_in_time_certified"] = "false"
    elif failure == "unusable":
        manifest["usable_for_industry_calculations"] = False
    path.write_text(
        "{" if failure == "malformed" else json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="INDUSTRY_"):
        inspect_industry_provider(str(tmp_path))


def test_parallel_replay_uses_two_independent_branches(tmp_path: Path) -> None:
    public = load_experiment_config("configs/parallel_branches_experiment.toml")
    config = mock_config(tmp_path)
    config = replace(
        config,
        architecture=public.architecture,
        runtime=public.runtime,
        workflow=public.workflow,
    )
    result = run_formal_runtime(config, run_id="parallel", resume=False)
    assert result.architecture_name == "parallel_branches"
    assert len(result.branch_outcome_refs) == 2
    assert Path(result.factor_library_csv_path).is_file()
