"""Formal-run configuration and dependency preflight checks."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from cosmosalpha.core import (
    CoreValidationError,
    ExperimentConfig,
    QlibSourceConfig,
    to_json,
)
from cosmosalpha.data.qlib_industry import inspect_industry_provider
from cosmosalpha.data.qlib_mask import read_mask_new_stock_min_age_years
from cosmosalpha.factors import QlibCompilationError, QlibExpressionCompiler
from cosmosalpha.llm import (
    PromptError,
    PromptTemplateRegistry,
    resolve_prompt_manifest,
)
from cosmosalpha.storage import RunStorageLayout

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SUPPORTED_DATA_PROVIDERS = {"mock", "qlib"}
_SUPPORTED_LLM_PROVIDERS = {"mock", "openai", "openai-compatible"}
_SUPPORTED_STORAGE_BACKENDS = {"local_mock", "local_sqlite"}
_PLACEHOLDER_PREFIX = "replace-with-"
_REQUIRED_QLIB_MARKET_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turn",
    "returns",
}


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """One structured reason that a formal run cannot start."""

    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        """Validate a machine-readable preflight issue."""
        if type(self.code) is not str or not self.code:
            raise CoreValidationError("preflight issue code must be non-empty")
        if type(self.message) is not str or not self.message:
            raise CoreValidationError("preflight issue message must be non-empty")
        if type(self.retryable) is not bool:
            raise CoreValidationError("preflight issue retryable must be boolean")


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Stable, non-secret result of formal-run startup validation."""

    run_id: str
    config_snapshot_id: str
    ready: bool
    checks: tuple[str, ...]
    issues: tuple[PreflightIssue, ...]
    schema_version: str = "run-preflight/1.0"

    def __post_init__(self) -> None:
        """Validate report shape and ready/issue consistency."""
        if type(self.run_id) is not str or not self.run_id:
            raise CoreValidationError("run_id must be non-empty")
        if type(self.config_snapshot_id) is not str or not self.config_snapshot_id:
            raise CoreValidationError("config_snapshot_id must be non-empty")
        if not isinstance(self.checks, tuple) or not all(
            type(check) is str and check for check in self.checks
        ):
            raise CoreValidationError("checks must be a tuple of non-empty strings")
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, PreflightIssue) for issue in self.issues
        ):
            raise CoreValidationError("issues must contain PreflightIssue values")
        if type(self.ready) is not bool or self.ready != (not self.issues):
            raise CoreValidationError("ready must equal whether issues is empty")
        if self.schema_version != "run-preflight/1.0":
            raise CoreValidationError("unsupported preflight schema version")

    def summary_json(self) -> str:
        """Serialize the report without secrets or full configuration values."""
        return to_json(self)


def run_preflight(
    config: ExperimentConfig,
    *,
    run_id: str,
    resume: bool = False,
) -> PreflightReport:
    """Validate a formal run before creating repositories or calling services.

    The function is deliberately side-effect free: it does not initialize a
    database, import Qlib, resolve an LLM endpoint, or create storage files.

    Args:
        config: Strict immutable experiment configuration.
        run_id: Stable CLI-supplied run identifier.
        resume: Whether an existing workflow database is required.

    Returns:
        A structured report suitable for CLI output and audit logs.
    """
    if not isinstance(config, ExperimentConfig):
        raise CoreValidationError("config must be ExperimentConfig")
    issues: list[PreflightIssue] = []
    checks: list[str] = []
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        issues.append(
            PreflightIssue("RUN_ID_INVALID", "run_id must be a safe CLI identifier")
        )
    else:
        checks.append("run_id")

    config_snapshot_id = _config_snapshot_id(config)
    _check_aliases(config, issues, checks)
    _check_versions(config, issues, checks)
    _check_prompts(config, issues, checks)
    _check_operator_backend(config, issues, checks)
    _check_storage(config, issues, checks, run_id=run_id, resume=resume)
    _check_llm(config, issues, checks)
    _check_qlib(config, issues, checks)
    return PreflightReport(
        run_id=run_id,
        config_snapshot_id=config_snapshot_id,
        ready=not issues,
        checks=tuple(checks),
        issues=tuple(issues),
    )


def _config_snapshot_id(config: ExperimentConfig) -> str:
    from cosmosalpha.core import create_config_snapshot

    return create_config_snapshot(config).snapshot_id


def _check_aliases(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    if config.data.provider not in _SUPPORTED_DATA_PROVIDERS:
        issues.append(
            PreflightIssue(
                "DATA_PROVIDER_UNSUPPORTED",
                f"unsupported data provider: {config.data.provider}",
            )
        )
    else:
        checks.append("data_provider_alias")
    if config.llm.provider not in _SUPPORTED_LLM_PROVIDERS:
        issues.append(
            PreflightIssue(
                "LLM_PROVIDER_UNSUPPORTED",
                f"unsupported LLM provider: {config.llm.provider}",
            )
        )
    else:
        checks.append("llm_provider_alias")
    if config.storage.backend not in _SUPPORTED_STORAGE_BACKENDS:
        issues.append(
            PreflightIssue(
                "STORAGE_BACKEND_UNSUPPORTED",
                f"unsupported storage backend: {config.storage.backend}",
            )
        )
    else:
        checks.append("storage_backend_alias")


def _check_versions(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    if not config.data.dataset_version or config.data.dataset_version.startswith(
        _PLACEHOLDER_PREFIX
    ):
        issues.append(
            PreflightIssue("DATA_VERSION_MISSING", "data.dataset_version is not fixed")
        )
    else:
        checks.append("data_version")
    if config.operators.operator_set != "OPERATOR_SET_BASE":
        issues.append(
            PreflightIssue(
                "OPERATOR_SET_UNSUPPORTED",
                f"unsupported operator set: {config.operators.operator_set}",
            )
        )
    else:
        checks.append("operator_set_version")
    if config.evaluation.protocol != "rankic_neutralized_v1":
        issues.append(
            PreflightIssue(
                "EVALUATION_PROTOCOL_UNSUPPORTED",
                f"unsupported evaluation protocol: {config.evaluation.protocol}",
            )
        )
    else:
        checks.append("evaluation_protocol")


def _check_prompts(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    manifest = resolve_prompt_manifest(config.prompts.manifest)
    if not manifest.is_file():
        issues.append(
            PreflightIssue(
                "PROMPT_MANIFEST_MISSING", f"prompt manifest not found: {manifest}"
            )
        )
        return
    try:
        registry = PromptTemplateRegistry.from_manifest(
            config.prompts.manifest,
        )
    except (PromptError, OSError, ValueError) as error:
        issues.append(PreflightIssue("PROMPT_MANIFEST_INVALID", str(error)))
        return
    if not registry.prompt_ids():
        issues.append(PreflightIssue("PROMPT_CATALOG_EMPTY", "prompt catalog is empty"))
    else:
        checks.append("prompt_manifest")


def _check_operator_backend(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    try:
        QlibExpressionCompiler()
    except QlibCompilationError as error:
        issues.append(PreflightIssue("OPERATOR_BACKEND_INCOMPLETE", str(error)))
    else:
        checks.append("qlib_operator_backend_coverage")


def _check_storage(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
    *,
    run_id: str,
    resume: bool,
) -> None:
    storage_root = Path(config.storage.root).expanduser()
    existing_parent = _nearest_existing_parent(storage_root)
    if existing_parent is None:
        issues.append(
            PreflightIssue(
                "STORAGE_PARENT_MISSING",
                f"storage parent does not exist: {storage_root.parent}",
            )
        )
    elif not os.access(existing_parent, os.W_OK):
        issues.append(
            PreflightIssue(
                "STORAGE_NOT_WRITABLE",
                f"storage parent is not writable: {existing_parent}",
            )
        )
    if resume:
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            issues.append(
                PreflightIssue(
                    "RESUME_STATE_MISSING",
                    "workflow database cannot be resolved for an invalid run_id",
                )
            )
        else:
            layout = RunStorageLayout.resolve(storage_root, run_id)
            _check_resume_state(layout.workflow_db_path, run_id, issues)
    if not any(issue.code.startswith("STORAGE_") for issue in issues):
        checks.append("storage_paths")


def _check_resume_state(
    workflow_db: Path,
    run_id: str,
    issues: list[PreflightIssue],
) -> None:
    """Confirm that a read-only SQLite workflow state contains ``run_id``."""
    if not workflow_db.is_file():
        issues.append(
            PreflightIssue(
                "RESUME_STATE_MISSING", f"workflow database not found: {workflow_db}"
            )
        )
        return
    try:
        database_uri = f"file:{workflow_db.resolve()}?mode=ro"
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
    except sqlite3.Error as error:
        issues.append(
            PreflightIssue(
                "RESUME_STATE_MISSING", f"workflow state is unreadable: {error}"
            )
        )
        return
    if exists is None:
        issues.append(
            PreflightIssue("RESUME_STATE_MISSING", f"workflow run not found: {run_id}")
        )


def _nearest_existing_parent(path: Path) -> Path | None:
    """Return the nearest existing directory where a run may initialize paths."""
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _check_llm(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    if config.llm.replay_only:
        checks.append("replay_llm_default")
        return
    if config.llm.provider == "mock":
        issues.append(
            PreflightIssue(
                "REAL_LLM_PROVIDER_INVALID", "real LLM cannot use mock provider"
            )
        )
        return
    key_configured = config.llm.api_key not in {"", "unused", "<redacted>"}
    key_env_configured = bool(config.llm.api_key_env.strip()) and bool(
        os.environ.get(config.llm.api_key_env.strip(), "").strip()
    )
    if not key_configured and not key_env_configured:
        issues.append(
            PreflightIssue(
                "LLM_API_KEY_NOT_CONFIGURED", "real LLM API key is not configured"
            )
        )
    else:
        checks.append("real_llm_explicitly_enabled")


def _check_qlib(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    if not config.qlib.enabled:
        if config.data.provider == "qlib":
            issues.append(
                PreflightIssue(
                    "QLIB_CONFIG_DISABLED",
                    "data.provider=qlib requires qlib.enabled=true",
                )
            )
        else:
            checks.append("qlib_not_required")
        return
    if config.data.provider != "qlib":
        issues.append(
            PreflightIssue(
                "QLIB_PROVIDER_MISMATCH", "qlib.enabled requires data.provider=qlib"
            )
        )
    if importlib.util.find_spec("qlib") is None:
        issues.append(
            PreflightIssue("QLIB_DEPENDENCY_MISSING", "qlib package is not installed")
        )
    if config.qlib.instruments:
        checks.append("qlib_explicit_instruments")
    else:
        _check_qlib_named_universe(config, issues, checks)
    market_fields = {item.field_name for item in config.qlib.market.fields}
    missing_market_fields = sorted(_REQUIRED_QLIB_MARKET_FIELDS - market_fields)
    if missing_market_fields:
        issues.append(
            PreflightIssue(
                "QLIB_MARKET_FIELDS_MISSING",
                "market source is missing configured fields: "
                + ", ".join(missing_market_fields),
            )
        )
    exposure_fields = {item.field_name for item in config.qlib.exposure.fields}
    missing_exposure_fields = sorted(
        {
            "industry_id",
            "log_mkt_cap",
        }
        - exposure_fields
    )
    if missing_exposure_fields:
        issues.append(
            PreflightIssue(
                "QLIB_EXPOSURE_FIELDS_MISSING",
                "exposure source is missing fields: "
                + ", ".join(missing_exposure_fields),
            )
        )
    mask_fields = {item.field_name for item in config.qlib.mask.fields}
    missing_mask_fields = sorted(
        {
            "universe_mask",
            "eligible_mask",
            "entry_tradable_mask",
            "exit_tradable_mask",
        }
        - mask_fields
    )
    if missing_mask_fields:
        issues.append(
            PreflightIssue(
                "QLIB_MASK_FIELDS_MISSING",
                "mask source is missing fields: " + ", ".join(missing_mask_fields),
            )
        )
    for name, source in (
        ("market", config.qlib.market),
        ("exposure", config.qlib.exposure),
        ("mask", config.qlib.mask),
    ):
        _check_qlib_source(name, source, issues)
    _check_qlib_new_stock_policy(config, issues, checks)
    if not any(issue.code.startswith("QLIB_") for issue in issues):
        checks.append("qlib_sources_and_dependency")


def _check_qlib_named_universe(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    universe = config.universe.name
    if _RUN_ID_PATTERN.fullmatch(universe) is None:
        issues.append(
            PreflightIssue(
                "QLIB_UNIVERSE_NAME_INVALID",
                "universe.name must be a safe Qlib instrument file stem",
            )
        )
        return
    local_paths: list[Path] = []
    for source_name, source in (
        ("market", config.qlib.market),
        ("exposure", config.qlib.exposure),
        ("mask", config.qlib.mask),
    ):
        if "://" in source.provider_uri or source.provider_uri.startswith(
            _PLACEHOLDER_PREFIX
        ):
            continue
        universe_path = (
            Path(source.provider_uri).expanduser() / "instruments" / f"{universe}.txt"
        )
        if not universe_path.is_file():
            issues.append(
                PreflightIssue(
                    "QLIB_UNIVERSE_FILE_MISSING",
                    f"{source_name} source is missing named universe: {universe_path}",
                )
            )
        else:
            local_paths.append(universe_path)
    if len(local_paths) > 1:
        try:
            digests = {
                hashlib.sha256(path.read_bytes()).hexdigest() for path in local_paths
            }
        except OSError as error:
            issues.append(PreflightIssue("QLIB_UNIVERSE_FILE_UNREADABLE", str(error)))
        else:
            if len(digests) != 1:
                issues.append(
                    PreflightIssue(
                        "QLIB_UNIVERSE_FILE_MISMATCH",
                        f"named universe files differ across Qlib sources: {universe}",
                    )
                )
    if not any(
        issue.code
        in {
            "QLIB_UNIVERSE_NAME_INVALID",
            "QLIB_UNIVERSE_FILE_MISSING",
            "QLIB_UNIVERSE_FILE_MISMATCH",
            "QLIB_UNIVERSE_FILE_UNREADABLE",
        }
        for issue in issues
    ):
        checks.append("qlib_named_universe")


def _check_qlib_new_stock_policy(
    config: ExperimentConfig,
    issues: list[PreflightIssue],
    checks: list[str],
) -> None:
    mask_uri = config.qlib.mask.provider_uri
    if "://" in mask_uri or mask_uri.startswith(_PLACEHOLDER_PREFIX):
        return
    try:
        actual = read_mask_new_stock_min_age_years(Path(mask_uri))
    except ValueError as error:
        issues.append(PreflightIssue("QLIB_NEW_STOCK_POLICY_INVALID", str(error)))
        return
    if actual is None:
        if config.universe.name == "cn_main_board":
            issues.append(
                PreflightIssue(
                    "QLIB_NEW_STOCK_POLICY_MISSING",
                    "cn_main_board requires a versioned Mask new-stock policy",
                )
            )
        return
    expected = config.universe.new_stock_min_age_years
    if actual != expected:
        issues.append(
            PreflightIssue(
                "QLIB_NEW_STOCK_POLICY_MISMATCH",
                f"Mask new-stock age is {actual}, configuration requires {expected}",
            )
        )
        return
    checks.append("qlib_new_stock_policy")


def _check_qlib_source(
    name: str,
    source: QlibSourceConfig,
    issues: list[PreflightIssue],
) -> None:
    if source.provider_uri.startswith(_PLACEHOLDER_PREFIX):
        issues.append(
            PreflightIssue(
                "QLIB_SOURCE_URI_MISSING", f"{name} source URI is not configured"
            )
        )
    if name == "exposure" and not source.provider_uri.startswith(_PLACEHOLDER_PREFIX):
        try:
            inspect_industry_provider(source.provider_uri)
        except ValueError as error:
            issues.append(PreflightIssue("QLIB_INDUSTRY_SOURCE_INVALID", str(error)))
