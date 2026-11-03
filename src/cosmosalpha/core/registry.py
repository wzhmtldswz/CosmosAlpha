"""Controlled component registry keyed by stable aliases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from cosmosalpha.core.errors import CoreValidationError

ComponentT = TypeVar("ComponentT")


def _require_string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CoreValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise CoreValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    return value


@dataclass(frozen=True, slots=True)
class RegisteredComponent[ComponentT]:
    """A component factory registered under a controlled alias."""

    alias: str
    version: str
    factory: Callable[..., ComponentT]
    component_type: type[ComponentT] | None

    def __post_init__(self) -> None:
        """Validate registered component metadata."""
        _require_string(self.alias, "alias")
        _require_string(self.version, "version")
        if type(self.factory) is str or not callable(self.factory):
            raise CoreValidationError("factory must be a callable, not a class path")
        if self.component_type is not None and not isinstance(
            self.component_type, type
        ):
            raise CoreValidationError("component_type must be a type or None")


class ComponentRegistry[ComponentT]:
    """Register and create components through stable aliases only."""

    def __init__(self, component_type: type[ComponentT] | None = None) -> None:
        """Create an empty registry.

        Args:
            component_type: Optional runtime type required for created components.
        """
        if component_type is not None and not isinstance(component_type, type):
            raise CoreValidationError("component_type must be a type or None")
        self._component_type = component_type
        self._components: dict[str, RegisteredComponent[ComponentT]] = {}

    def register(
        self,
        alias: str,
        version: str,
        factory: Callable[..., ComponentT],
    ) -> RegisteredComponent[ComponentT]:
        """Register a component factory.

        Args:
            alias: Stable registry alias.
            version: Component implementation version.
            factory: Callable that creates or returns the component.

        Returns:
            Registered component metadata.
        """
        normalized_alias = _require_string(alias, "alias")
        if normalized_alias in self._components:
            raise CoreValidationError(f"component alias already registered: {alias!r}")
        component = RegisteredComponent(
            alias=normalized_alias,
            version=version,
            factory=factory,
            component_type=self._component_type,
        )
        self._components[normalized_alias] = component
        return component

    def get(self, alias: str) -> RegisteredComponent[ComponentT]:
        """Return registered component metadata for an alias."""
        normalized_alias = _require_string(alias, "alias")
        try:
            return self._components[normalized_alias]
        except KeyError as error:
            raise CoreValidationError(
                f"unknown component alias: {normalized_alias!r}"
            ) from error

    def create(self, alias: str, *args: Any, **kwargs: Any) -> ComponentT:
        """Create a component by alias using its registered factory."""
        component = self.get(alias)
        instance = component.factory(*args, **kwargs)
        if self._component_type is not None and not isinstance(
            instance,
            self._component_type,
        ):
            raise CoreValidationError(
                f"factory for {component.alias!r} did not return "
                f"{self._component_type.__name__}"
            )
        return cast(ComponentT, instance)

    def contains(self, alias: str) -> bool:
        """Return whether an alias is registered."""
        normalized_alias = _require_string(alias, "alias")
        return normalized_alias in self._components

    def aliases(self) -> tuple[str, ...]:
        """Return registered aliases in stable sorted order."""
        return tuple(sorted(self._components))
