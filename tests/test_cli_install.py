"""CLI behavior and isolated regular, editable, wheel and source distributions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest

from cosmosalpha import __version__
from cosmosalpha.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_environment() -> dict[str, str]:
    """Prevent source-path and credential leakage into isolated CLI checks."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("COSMOSALPHA_LLM_API_KEY", None)
    env.pop("COSMOSALPHA_LLM_BASE_URL", None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PIP_NO_INDEX="1",
        PIP_NO_CACHE_DIR="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return env


def run_command(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a bounded offline command and show its logs on failure."""
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=clean_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build package artifacts only inside a temporary source copy."""
    pytest.importorskip("setuptools")
    base = tmp_path_factory.mktemp("distributions")
    project = base / "project"
    shutil.copytree(
        PROJECT_ROOT,
        project,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "*.egg-info",
            ".git",
            ".venv",
            "runs",
        ),
    )
    output = base / "artifacts"
    output.mkdir()
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(output),
            str(project),
        ],
        base,
    )
    run_command(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_sdist; build_sdist("
            + repr(str(output))
            + ")",
        ],
        project,
    )
    return project, next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def test_cli_help_version_and_structured_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"cosmosalpha {__version__}"
    assert (
        main(
            [
                "run",
                "--config",
                "absent.toml",
                "--run-id",
                "missing",
                "--preflight-only",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["code"] == "CONFIG_LOAD_FAILED"


def test_distributions_include_one_complete_prompt_set(
    distributions: tuple[Path, Path, Path],
) -> None:
    _, wheel, source = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        prompts = [name for name in names if name.endswith(".yaml")]
        assert len(prompts) == 8
        assert "cosmosalpha/llm/templates/v2/manifest.yaml" in prompts
        assert any("licenses/LICENSE" in name for name in names)
    with tarfile.open(source) as archive:
        names = archive.getnames()
        assert len([name for name in names if name.endswith(".yaml")]) == 8
        assert {
            name.split("/configs/", 1)[1]
            for name in names
            if "/configs/" in name and name.endswith(".toml")
        } == {
            "chain_experiment.toml",
            "graph_experiment.toml",
            "lightgbm_integration.example.toml",
            "mcts_experiment.toml",
            "parallel_branches_experiment.toml",
        }
        assert any(name.endswith("/README_zh-CN.md") for name in names)
        assert {
            name.split("/docs/", 1)[1]
            for name in names
            if "/docs/" in name and name.endswith(".md")
        } == {"configuration.md", "graph.md", "mcts.md", "lightgbm.md"}


@pytest.mark.parametrize("installation", ["regular", "editable", "wheel"])
def test_installed_cli_and_prompts_outside_repository(
    distributions: tuple[Path, Path, Path],
    installation: str,
    tmp_path: Path,
) -> None:
    project, wheel, _ = distributions
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    arguments = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--force-reinstall",
    ]
    if installation == "editable":
        arguments.extend(["-e", str(project)])
    else:
        arguments.append(str(wheel if installation == "wheel" else project))
    outside = tmp_path / "outside"
    outside.mkdir()
    run_command(arguments, outside)
    probe = run_command(
        [
            str(python),
            "-c",
            "import json, cosmosalpha; from cosmosalpha.llm import resolve_prompt_manifest; "
            "print(json.dumps({'module':cosmosalpha.__file__,"
            "'manifest':str(resolve_prompt_manifest('builtin:v2'))}))",
        ],
        outside,
    )
    paths = json.loads(probe.stdout)
    expected = project / "src" if installation == "editable" else environment
    assert Path(paths["module"]).resolve().is_relative_to(expected.resolve())
    assert Path(paths["manifest"]).is_file()
    assert Path(paths["manifest"]).resolve().is_relative_to(expected.resolve())
    version = run_command(
        [str(python), "-m", "cosmosalpha", "--version"], outside
    ).stdout
    console = scripts / ("cosmosalpha.exe" if os.name == "nt" else "cosmosalpha")
    assert run_command([str(console), "--version"], outside).stdout == version
    for option in ("--smoke-loop", "--smoke-multi-cycle", "--mcts-smoke"):
        payload = json.loads(
            run_command([str(python), "-m", "cosmosalpha", option], outside).stdout
        )
        assert payload
    result = subprocess.run(
        [
            str(python),
            "-m",
            "cosmosalpha",
            "run",
            "--config",
            str(project / "configs/graph_experiment.toml"),
            "--run-id",
            "no-data",
            "--preflight-only",
        ],
        cwd=outside,
        env=clean_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert not any(issue["code"].startswith("PROMPT_") for issue in report["issues"])
    assert not (outside / "runs").exists()
