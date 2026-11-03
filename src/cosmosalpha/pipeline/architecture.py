"""Controlled top-level architecture aliases."""

from __future__ import annotations

from dataclasses import dataclass

from cosmosalpha.core import ComponentRegistry


@dataclass(frozen=True, slots=True)
class ChainArchitecture:
    """Select the existing single serial chain."""

    name: str = "chain"


@dataclass(frozen=True, slots=True)
class MCTSArchitecture:
    """Select Monte Carlo Tree Search as the top-level architecture."""

    name: str = "mcts"


@dataclass(frozen=True, slots=True)
class ParallelBranchesArchitecture:
    """Select configured branches whose internal cycles remain serial."""

    name: str = "parallel_branches"


@dataclass(frozen=True, slots=True)
class GraphArchitecture:
    """Select Graph Search v1 as the top-level architecture."""

    name: str = "graph"


ARCHITECTURE_REGISTRY: ComponentRegistry[object] = ComponentRegistry()
ARCHITECTURE_REGISTRY.register("chain", "1.0", ChainArchitecture)
ARCHITECTURE_REGISTRY.register("mcts", "1.0", MCTSArchitecture)
ARCHITECTURE_REGISTRY.register("graph", "1.0", GraphArchitecture)
ARCHITECTURE_REGISTRY.register(
    "parallel_branches",
    "1.0",
    ParallelBranchesArchitecture,
)


__all__ = [
    "ARCHITECTURE_REGISTRY",
    "ChainArchitecture",
    "GraphArchitecture",
    "MCTSArchitecture",
    "ParallelBranchesArchitecture",
]
