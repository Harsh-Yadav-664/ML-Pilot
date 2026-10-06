"""Provider registry — register and retrieve provider instances."""

from __future__ import annotations

from typing import Any


class ProviderRegistry:
    """Central registry for all MLPilot providers."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._provider_classes: dict[str, type] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register a provider instance under *name*."""
        self._providers[name] = instance

    def register_class(self, name: str, cls: type) -> None:
        """Register a provider class under *name* (instantiated on demand)."""
        self._provider_classes[name] = cls

    def get(self, name: str) -> Any:
        """Return the provider registered as *name*."""
        if name in self._providers:
            return self._providers[name]
        if name in self._provider_classes:
            instance = self._provider_classes[name]()
            self._providers[name] = instance
            return instance
        raise KeyError(f"Provider not found: {name!r}")

    def list_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(set(list(self._providers.keys()) + list(self._provider_classes.keys())))

    def has(self, name: str) -> bool:
        """Return True if *name* is registered."""
        return name in self._providers or name in self._provider_classes


# Global singleton registry
registry = ProviderRegistry()
