"""Strict prompt template loading and rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template

from cosmosalpha.core import CoreValidationError, JsonValue, stable_sha256, to_json
from cosmosalpha.llm.errors import PromptError

AGENT_PROMPT_CONTRACT_VERSION = "agent-prompt/2.0"
DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parent / "templates" / "v2"
BUILTIN_PROMPT_MANIFEST = "builtin:v2"
_PLACEHOLDER_PATTERN = re.compile(
    r"\$(?:{(?P<braced>[_a-zA-Z][_a-zA-Z0-9]*)}|(?P<named>[_a-zA-Z][_a-zA-Z0-9]*))"
)


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """One rendered chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validate message fields."""
        _require_string(self.role, "role")
        _require_string(self.content, "content")
        if self.role not in ("system", "user"):
            raise CoreValidationError("role must be system or user")


@dataclass(frozen=True, slots=True)
class PromptManifestEntry:
    """One prompt template entry declared by the manifest."""

    prompt_id: str
    version: str
    role: str
    template_file: str
    input_schema: str
    output_schema: str
    max_output_tokens: int

    def __post_init__(self) -> None:
        """Validate manifest entry fields."""
        _require_string(self.prompt_id, "prompt_id")
        _require_string(self.version, "version")
        _require_string(self.role, "role")
        _require_string(self.template_file, "template_file")
        _require_string(self.input_schema, "input_schema")
        _require_string(self.output_schema, "output_schema")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise CoreValidationError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Strict system/user prompt template."""

    prompt_id: str
    version: str
    role: str
    input_schema: str
    output_schema_version: str
    max_output_tokens: int
    system: str
    user: str
    output_schema: Mapping[str, JsonValue]
    checksum: str

    def __post_init__(self) -> None:
        """Validate prompt template fields."""
        _require_string(self.prompt_id, "prompt_id")
        _require_string(self.version, "version")
        _require_string(self.role, "role")
        _require_string(self.input_schema, "input_schema")
        _require_string(self.output_schema_version, "output_schema_version")
        _require_string(self.system, "system")
        _require_string(self.user, "user")
        if not isinstance(self.output_schema, Mapping):
            raise CoreValidationError("output_schema must be a mapping")
        _require_string(self.checksum, "checksum")

    @property
    def variables(self) -> tuple[str, ...]:
        """Return sorted placeholder names used by system and user templates."""
        names = {
            match.group("braced") or match.group("named")
            for text in (self.system, self.user)
            for match in _PLACEHOLDER_PATTERN.finditer(text)
        }
        return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class PromptRenderResult:
    """Rendered prompt passed to an LLM client."""

    prompt_render_id: str
    prompt_id: str
    template_version: str
    template_checksum: str
    input_schema: str
    output_schema: str
    messages: tuple[PromptMessage, ...]
    output_json_schema: Mapping[str, JsonValue]
    estimated_input_tokens: int
    token_estimation_method: str
    truncation_report: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate rendered prompt fields."""
        _require_string(self.prompt_render_id, "prompt_render_id")
        _require_string(self.prompt_id, "prompt_id")
        _require_string(self.template_version, "template_version")
        _require_string(self.template_checksum, "template_checksum")
        _require_string(self.input_schema, "input_schema")
        _require_string(self.output_schema, "output_schema")
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, PromptMessage) for message in self.messages
        ):
            raise CoreValidationError("messages must be a tuple of PromptMessage")
        if not isinstance(self.output_json_schema, Mapping):
            raise CoreValidationError("output_json_schema must be a mapping")
        if (
            type(self.estimated_input_tokens) is not int
            or self.estimated_input_tokens < 0
        ):
            raise CoreValidationError("estimated_input_tokens must be non-negative")
        _require_string(self.token_estimation_method, "token_estimation_method")
        _require_string(self.truncation_report, "truncation_report")
        if self.schema_version != "prompt-render-result/1.0":
            raise CoreValidationError("invalid prompt render schema_version")


def resolve_prompt_manifest(manifest: str | Path) -> Path:
    """Resolve the built-in v2 manifest or an explicit local manifest path."""
    if str(manifest) == BUILTIN_PROMPT_MANIFEST:
        return DEFAULT_PROMPT_ROOT / "manifest.yaml"
    return Path(manifest).expanduser()


class PromptTemplateRegistry:
    """Load prompt templates by stable alias and version."""

    def __init__(
        self,
        templates: tuple[PromptTemplate, ...],
    ) -> None:
        """Create a registry from validated templates."""
        self.contract_version = AGENT_PROMPT_CONTRACT_VERSION
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._latest: dict[str, PromptTemplate] = {}
        for template in templates:
            key = (template.prompt_id, template.version)
            if key in self._templates:
                raise PromptError(f"PROMPT_TEMPLATE_DUPLICATE: {key}")
            self._templates[key] = template
            self._latest[template.prompt_id] = template

    @classmethod
    def from_manifest(cls, manifest: str | Path) -> PromptTemplateRegistry:
        """Load a built-in or explicit manifest and its sibling templates."""
        manifest_path = resolve_prompt_manifest(manifest)
        root_path = manifest_path.parent
        manifest_values = _read_json_mapping(manifest_path)
        if manifest_values.get("contract_version") != AGENT_PROMPT_CONTRACT_VERSION:
            raise PromptError("PROMPT_VERSION_MISMATCH: unsupported contract version")
        raw_prompts = manifest_values.get("prompts")
        if not isinstance(raw_prompts, list):
            raise PromptError("PROMPT_SCHEMA_MISMATCH: prompts must be a list")
        templates: list[PromptTemplate] = []
        for raw_entry in raw_prompts:
            entry = _manifest_entry(raw_entry)
            templates.append(_load_template(root_path, entry))
        return cls(tuple(templates))

    @classmethod
    def from_directory(
        cls,
        root: str | Path = DEFAULT_PROMPT_ROOT,
    ) -> PromptTemplateRegistry:
        """Load manifest.yaml and templates from a prompt directory."""
        return cls.from_manifest(Path(root) / "manifest.yaml")

    def get(self, prompt_id: str, version: str | None = None) -> PromptTemplate:
        """Return a prompt template by id and optional version."""
        if version is None:
            try:
                return self._latest[prompt_id]
            except KeyError as error:
                raise PromptError(f"PROMPT_TEMPLATE_NOT_FOUND: {prompt_id}") from error
        try:
            return self._templates[(prompt_id, version)]
        except KeyError as error:
            raise PromptError(
                f"PROMPT_TEMPLATE_NOT_FOUND: {prompt_id}/{version}"
            ) from error

    def prompt_ids(self) -> tuple[str, ...]:
        """Return prompt ids in stable order."""
        return tuple(sorted(self._latest))


class PromptRenderer:
    """Render strict templates with exact variable sets and token budget checks."""

    def __init__(self, registry: PromptTemplateRegistry) -> None:
        """Create a renderer bound to a registry."""
        self._registry = registry

    def render(
        self,
        prompt_id: str,
        variables: Mapping[str, object],
        *,
        version: str | None = None,
        max_input_tokens: int = 16000,
    ) -> PromptRenderResult:
        """Render one prompt template."""
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise PromptError("PROMPT_CONTEXT_TOO_LARGE: invalid token budget")
        template = self._registry.get(prompt_id, version)
        rendered_variables = {
            key: value if type(value) is str else to_json(value)
            for key, value in variables.items()
        }
        _validate_variables(template, rendered_variables)
        system = Template(template.system).substitute(rendered_variables)
        user = Template(template.user).substitute(rendered_variables)
        user = _append_output_contract(user, template.output_schema)
        messages = (
            PromptMessage("system", system),
            PromptMessage("user", user),
        )
        estimated_tokens = _estimate_tokens(messages)
        if estimated_tokens > max_input_tokens:
            raise PromptError(
                "PROMPT_CONTEXT_TOO_LARGE: rendered prompt exceeds budget"
            )
        prompt_render_id = stable_sha256(
            {
                "messages": messages,
                "prompt_id": template.prompt_id,
                "template_checksum": template.checksum,
                "template_version": template.version,
            }
        )
        return PromptRenderResult(
            prompt_render_id=f"prompt-render-{prompt_render_id}",
            prompt_id=template.prompt_id,
            template_version=template.version,
            template_checksum=template.checksum,
            input_schema=template.input_schema,
            output_schema=template.output_schema_version,
            messages=messages,
            output_json_schema=template.output_schema,
            estimated_input_tokens=estimated_tokens,
            token_estimation_method="chars_div_4_ceiling",
            truncation_report="none",
            schema_version="prompt-render-result/1.0",
        )


def _load_template(root: Path, entry: PromptManifestEntry) -> PromptTemplate:
    payload = _read_json_mapping(root / entry.template_file)
    if (
        payload.get("prompt_id") != entry.prompt_id
        or payload.get("version") != entry.version
    ):
        raise PromptError("PROMPT_VERSION_MISMATCH: template does not match manifest")
    system = payload.get("system")
    user = payload.get("user")
    output_schema = payload.get("output_schema")
    if type(system) is not str or type(user) is not str:
        raise PromptError("PROMPT_SCHEMA_MISMATCH: system and user must be strings")
    if not isinstance(output_schema, Mapping):
        raise PromptError("PROMPT_SCHEMA_MISMATCH: output_schema must be an object")
    checksum = stable_sha256(
        {
            "input_schema": entry.input_schema,
            "max_output_tokens": entry.max_output_tokens,
            "output_schema": output_schema,
            "output_schema_version": entry.output_schema,
            "prompt_id": entry.prompt_id,
            "role": entry.role,
            "system": system,
            "user": user,
            "version": entry.version,
        }
    )
    return PromptTemplate(
        prompt_id=entry.prompt_id,
        version=entry.version,
        role=entry.role,
        input_schema=entry.input_schema,
        output_schema_version=entry.output_schema,
        max_output_tokens=entry.max_output_tokens,
        system=system,
        user=user,
        output_schema=output_schema,
        checksum=f"sha256:{checksum}",
    )


def _manifest_entry(value: object) -> PromptManifestEntry:
    if not isinstance(value, Mapping):
        raise PromptError("PROMPT_SCHEMA_MISMATCH: manifest entry must be object")
    try:
        return PromptManifestEntry(
            prompt_id=_string_field(value, "prompt_id"),
            version=_string_field(value, "version"),
            role=_string_field(value, "role"),
            template_file=_string_field(value, "template_file"),
            input_schema=_string_field(value, "input_schema"),
            output_schema=_string_field(value, "output_schema"),
            max_output_tokens=_int_field(value, "max_output_tokens"),
        )
    except CoreValidationError as error:
        raise PromptError("PROMPT_SCHEMA_MISMATCH: invalid manifest entry") from error


def _read_json_mapping(path: Path) -> Mapping[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PromptError(f"PROMPT_TEMPLATE_NOT_FOUND: {path}") from error
    except json.JSONDecodeError as error:
        raise PromptError(
            f"PROMPT_SCHEMA_MISMATCH: invalid JSON template {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise PromptError("PROMPT_SCHEMA_MISMATCH: template root must be object")
    return payload


def _validate_variables(
    template: PromptTemplate,
    variables: Mapping[str, str],
) -> None:
    expected = set(template.variables)
    actual = set(variables)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise PromptError("PROMPT_VARIABLE_MISSING: " + ", ".join(sorted(missing)))
    if unknown:
        raise PromptError("PROMPT_VARIABLE_UNKNOWN: " + ", ".join(sorted(unknown)))
    for key, value in variables.items():
        if type(value) is not str or not value:
            raise PromptError(f"PROMPT_VARIABLE_INVALID: {key}")


def _estimate_tokens(messages: tuple[PromptMessage, ...]) -> int:
    total_chars = sum(len(message.role) + len(message.content) for message in messages)
    return max(1, (total_chars + 3) // 4)


def _append_output_contract(
    user_message: str,
    output_schema: Mapping[str, JsonValue],
) -> str:
    """Make the deterministic output contract visible to the model.

    The provider request currently uses JSON mode rather than provider-specific JSON
    Schema mode.  The schema must therefore be present in the user-visible prompt;
    keeping it only in ``PromptRenderResult`` is insufficient guidance for a model.
    """
    return (
        f"{user_message}\n\n"
        "Authoritative output contract (JSON Schema):\n"
        f"{to_json(output_schema)}\n"
        "Return exactly one JSON object satisfying this contract. "
        "Do not add fields, change field types, or use enum values outside the contract."
    )


def _string_field(values: Mapping[str, object], field_name: str) -> str:
    value = values.get(field_name)
    if type(value) is not str or not value:
        raise PromptError(f"PROMPT_SCHEMA_MISMATCH: {field_name}")
    return value


def _int_field(values: Mapping[str, object], field_name: str) -> int:
    value = values.get(field_name)
    if type(value) is not int:
        raise PromptError(f"PROMPT_SCHEMA_MISMATCH: {field_name}")
    return value


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CoreValidationError(f"{field_name} must be a non-empty string")
    return value
