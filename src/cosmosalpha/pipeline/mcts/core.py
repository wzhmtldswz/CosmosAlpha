"""MCTS core data structures: MCTSNode and MCTSTree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cosmosalpha.core import ObjectRef


@dataclass
class MCTSNode:
    """A single node in the MCTS search tree.

    Each node represents one evaluated factor expression and carries
    MCTS statistics (visit count, best reward) for UCT selection.
    """

    # --- Identity ---
    factor_id: str
    expression: str
    experiment_ref: ObjectRef | None = None
    parent: MCTSNode | None = None

    # --- MCTS state ---
    children: list[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    best_reward: float = float("-inf")

    # --- Metadata ---
    depth: int = 0
    action_label: str = ""
    action_dimension: str = (
        ""  # 论文 §3.2: 本节点的改进维度 (effectiveness/stability/diversity/turnover)
    )

    # --- Evaluation ---
    rank_ic_mean: float | None = None

    # --- Multi-dimensional scores (论文 §3.3, Step 1) ---
    stability: float | None = None  # RankIC positive-day ratio [0,1]
    turnover: float | None = None  # Factor auto-correlation [0,1]
    diversity: float | None = None  # Max abs corr with alpha_zoo [0,1]
    overall_score: float | None = None  # Weighted composite score S(f)

    # --- Quality-gated branching control ---
    cooling: int = 0  # 累计差子节点次数，UCT 冷却因子 = 1/(1+cooling)
    # improvement6 方案2：分支剪枝标记 —— 所属分支已被剪枝，UCT 跳过
    pruned: bool = False

    # --- Alpha Portrait (v4: 论文两阶段生成) ---
    portrait: str = ""  # LLM 生成的因子文字描述（名称+逻辑+伪代码）

    # --- P1-5: factor value series & alignment keys (memory only, not JSON-serialized) ---
    factor_values: np.ndarray | None = field(default=None, repr=False, compare=False)
    factor_keys: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to a JSON-safe dict, using factor_id for parent/children refs."""
        return {
            "factor_id": self.factor_id,
            "expression": self.expression,
            "experiment_ref": _ref_to_dict(self.experiment_ref),
            "parent_factor_id": self.parent.factor_id if self.parent else None,
            "children_factor_ids": [c.factor_id for c in self.children],
            "visit_count": self.visit_count,
            "best_reward": self.best_reward,
            "depth": self.depth,
            "action_label": self.action_label,
            "action_dimension": self.action_dimension,
            "rank_ic_mean": self.rank_ic_mean,
            "stability": self.stability,
            "turnover": self.turnover,
            "diversity": self.diversity,
            "overall_score": self.overall_score,
            "cooling": self.cooling,
            "pruned": self.pruned,
            "portrait": self.portrait,
        }


@dataclass
class MCTSTree:
    """Manages the entire MCTS search tree and the alpha zoo.

    The alpha zoo collects all nodes whose evaluation score exceeds
    the acceptance threshold.
    """

    root: MCTSNode
    alpha_zoo: dict[str, MCTSNode] = field(default_factory=dict)
    all_nodes: list[MCTSNode] = field(default_factory=list)

    def add_node(self, parent: MCTSNode, child: MCTSNode) -> None:
        """Attach a child node into the tree."""
        parent.children.append(child)
        child.parent = parent
        child.depth = parent.depth + 1
        self.all_nodes.append(child)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tree to a JSON-safe dict."""
        return {
            "schema_version": "mcts-tree/1.0",
            "root_factor_id": self.root.factor_id,
            "alpha_zoo_factor_ids": list(self.alpha_zoo.keys()),
            "nodes": [n.to_dict() for n in self.all_nodes],
        }

    def snapshot(self) -> MCTSTreeSnapshot:
        """Return a persistable snapshot of the tree."""
        data = self.to_dict()
        data["nodes"] = [MCTSNodeSnapshot(**node) for node in data["nodes"]]
        return MCTSTreeSnapshot(**data)


@dataclass(frozen=True, slots=True)
class MCTSNodeSnapshot:
    """Strictly-typed JSON snapshot of one MCTS node.

    The storage decoder requires concrete leaf types — ``dict[str, Any]``
    cannot be decoded — so node snapshots mirror ``MCTSNode.to_dict()``
    with explicit fields (P0-2: historical FZoo/tree loading).
    """

    factor_id: str
    expression: str
    experiment_ref: dict[str, str] | None = None
    parent_factor_id: str | None = None
    children_factor_ids: list[str] = field(default_factory=list)
    visit_count: int = 0
    best_reward: float = float("-inf")
    depth: int = 0
    action_label: str = ""
    action_dimension: str = ""
    rank_ic_mean: float | None = None
    stability: float | None = None
    turnover: float | None = None
    diversity: float | None = None
    overall_score: float | None = None
    cooling: int = 0
    pruned: bool = False
    portrait: str = ""


@dataclass
class MCTSTreeSnapshot:
    """Persistable snapshot of an MCTS tree (schema_version required by storage)."""

    schema_version: str = "mcts-tree/1.0"
    root_factor_id: str = ""
    alpha_zoo_factor_ids: list[str] = field(default_factory=list)
    nodes: list[MCTSNodeSnapshot] = field(default_factory=list)


def _ref_to_dict(ref: ObjectRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "object_id": ref.object_id,
        "object_type": ref.object_type,
        "schema_version": ref.schema_version,
    }
