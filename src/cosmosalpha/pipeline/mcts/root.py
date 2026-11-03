"""Root generation interface for MCTS multi-tree mode.

需求：取消种子准备机制。树根节点由一次**完整链式挖掘**产出
（HypothesisAgent→FactorDesign→FactorExpression→校验修复→因子计算→
评价→FeedbackAgent 生成 Lesson），无任何预定义种子约束。

本模块定义 :class:`RootGenerator` 协议作为稳定接口；当前默认实现为
:class:`ChainRootGenerator`。未来可替换为华泰种子池、LLM 生成、历史根复用
等其他实现，接口保持不变（多树模式在每棵树预算耗尽后调用 ``generate``
得到下一棵树的根）。
"""

from __future__ import annotations

from typing import Protocol

from cosmosalpha.agents import ResearchContext
from cosmosalpha.core import ArtifactRef
from cosmosalpha.data import DataSnapshot
from cosmosalpha.pipeline.mcts.core import MCTSNode


class RootGenerator(Protocol):
    """生成并评估一棵树的根节点.

    实现方负责：通过一次完整挖掘（默认链式 loop）产出根节点，
    并完成评估（RankIC 等分数）。调用方据此决定是否开树搜索。
    """

    def generate(
        self,
        *,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        run_id: str,
    ) -> MCTSNode:
        """Generate and evaluate one root through the configured factor workflow."""


class ChainRootGenerator:
    """链式挖掘根生成器：一次完整 loop 产出根节点（无 seed 约束）.

    持有 MCTSController（duck typing）以复用其链式根构建逻辑
    （``_build_root(seed_expression=None)``），避免重复实现 node 组装。
    """

    def __init__(self, controller: object) -> None:
        """Bind the controller that materializes each chain-generated root."""
        self._controller = controller

    def generate(
        self,
        *,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        run_id: str,
    ) -> MCTSNode:
        """Generate and evaluate one tree root node (chain-mining default)."""
        return self._controller._build_root(  # type: ignore[attr-defined]
            seed_expression=None,
            data_snapshot=data_snapshot,
            base_context=base_context,
            config_snapshot_ref=config_snapshot_ref,
            code_version=code_version,
            run_id=run_id,
        )
