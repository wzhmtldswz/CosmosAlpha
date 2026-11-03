"""Fzoo — effective alpha repository with quality-gated admission and few-shot sampling.

Replicates the paper's Fzoo mechanism (§3.4, §D.2):
- Multi-dimensional admission criteria (RankIC, RankIR, Diversity)
- Dimension-aware few-shot exemplar selection
- Cold-start support (empty Fzoo → no exemplars)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cosmosalpha.pipeline.mcts.core import MCTSNode

# ------------------------------------------------------------------
# Admission thresholds — 三阶段渐进（v6）
# ------------------------------------------------------------------

# Phase 1: 冷启动（Fzoo 为空）—— 低门槛，让因子流入
FZOO_COLD_START_RANK_IC: float = 0.010
FZOO_COLD_START_RANK_IR: float = 0.12

# Phase 2: 暖机（Fzoo 1-9 个因子）—— 恢复正常品质门槛
FZOO_MIN_RANK_IC: float = 0.018
FZOO_MIN_RANK_IR: float = 0.18

# Phase 3: 正常运行（Fzoo ≥ 10）—— 渐进门槛
FZOO_PROGRESSIVE_RANK_IC: float = 0.02  # Phase 3 IC 地板
FZOO_PROGRESSIVE_RANK_IR: float = 0.18  # Phase 3 IR 地板
FZOO_PROGRESSIVE_PERCENTILE: float = 10.0  # 超过 Fzoo 中此百分位才入库
FZOO_WARMUP_SIZE: int = 10  # 暖机期 Fzoo 大小阈值

FZOO_MIN_DIVERSITY: float = 0.2


@dataclass
class FzooEntry:
    """One entry in the effective alpha repository Fzoo.

    Wraps an MCTSNode with additional metadata required for
    quality-gated admission and few-shot sampling.
    """

    factor_id: str
    expression: str
    portrait: str  # Alpha Portrait 文字描述 (Phase 2 填充)
    rank_ic: float
    rank_ir: float
    diversity: float  # 1 - max_abs_corr_with_fzoo, [0,1]
    turnover: float  # lag-5 auto-correlation [0,1]
    overall_score: float
    created_at_cycle: int = 0  # 入库时的 cycle 编号

    @classmethod
    def from_node(
        cls,
        node: MCTSNode,
        portrait: str = "",
        created_at_cycle: int = 0,
    ) -> FzooEntry:
        """Create an FzooEntry from an MCTSNode."""
        return cls(
            factor_id=node.factor_id,
            expression=node.expression,
            portrait=portrait or getattr(node, "portrait", ""),
            rank_ic=node.rank_ic_mean or 0.0,
            rank_ir=node.stability or 0.0,
            diversity=node.diversity or 0.0,
            turnover=node.turnover or 0.0,
            overall_score=node.overall_score or 0.0,
            created_at_cycle=created_at_cycle,
        )


# ------------------------------------------------------------------
# Admission gate (v4: 渐进门槛)
# ------------------------------------------------------------------


def should_admit_to_fzoo(node: MCTSNode, fzoo: dict[str, FzooEntry]) -> bool:
    """三阶段渐进入库标准（v6）.

    使用 |IC|/|IR|（负因子取反后等价于正因子，方向无关）。

    Phase 1 (Fzoo 空):    冷启动低门槛 — |IC|≥0.010, |IR|≥0.12
    Phase 2 (Fzoo 1-9):   暖机正常门槛 — |IC|≥0.018, |IR|≥0.18
    Phase 3 (Fzoo ≥10):   渐进门槛 — |IC|≥max(0.02, 10%ile), |IR|≥max(0.18, 10%ile)
    """
    ic = node.rank_ic_mean
    if ic is None:
        return False
    ir = node.stability
    if ir is None:
        return False
    div = node.diversity
    if div is not None and div < FZOO_MIN_DIVERSITY:
        return False

    fzoo_size = len(fzoo)
    abs_ic = abs(ic)
    abs_ir = abs(ir)

    # ── Phase 1: 冷启动 ──
    if fzoo_size == 0:
        return abs_ic >= FZOO_COLD_START_RANK_IC and abs_ir >= FZOO_COLD_START_RANK_IR

    # ── Phase 2: 暖机 ──
    if fzoo_size < FZOO_WARMUP_SIZE:
        return abs_ic >= FZOO_MIN_RANK_IC and abs_ir >= FZOO_MIN_RANK_IR

    # ── Phase 3: 渐进门槛 ──
    fzoo_ics = [e.rank_ic for e in fzoo.values() if e.rank_ic > 0]
    fzoo_irs = [e.rank_ir for e in fzoo.values() if e.rank_ir > 0]
    ic_threshold = FZOO_PROGRESSIVE_RANK_IC
    ir_threshold = FZOO_PROGRESSIVE_RANK_IR
    if fzoo_ics:
        ic_threshold = max(
            ic_threshold, float(np.percentile(fzoo_ics, FZOO_PROGRESSIVE_PERCENTILE))
        )
    if fzoo_irs:
        ir_threshold = max(
            ir_threshold, float(np.percentile(fzoo_irs, FZOO_PROGRESSIVE_PERCENTILE))
        )
    return abs_ic >= ic_threshold and abs_ir >= ir_threshold


# ------------------------------------------------------------------
# Fzoo percentile scoring (论文 Eq.6 — percentile rank against Fzoo)
# ------------------------------------------------------------------


def compute_fzoo_percentile_score(
    node: MCTSNode,
    fzoo: dict[str, FzooEntry],
) -> dict[str, float]:
    """计算节点各维度相对 Fzoo 的百分位得分.

    论文 Eq.6: 每维得分 = 该维度值在 Fzoo 中的百分位排名 [0,1]。
    Fzoo 为空时返回绝对归一化值作为 fallback。

    Returns:
        Dict with keys: effectiveness, stability, diversity, turnover.
    """
    if not fzoo:
        # Fallback: absolute normalization
        return {
            "effectiveness": min(max((node.rank_ic_mean or 0.0) / 0.05, 0.0), 1.0),
            "stability": min(max((node.stability or 0.0) / 1.0, 0.0), 1.0),
            "diversity": node.diversity or 0.0,
            "turnover": node.turnover or 0.0,
        }

    entries = list(fzoo.values())

    def _pctile(value: float, fzoo_values: list[float]) -> float:
        """Percentile rank of value among fzoo_values. Returns [0, 1]."""
        if not fzoo_values:
            return 0.5
        below = sum(1 for v in fzoo_values if v <= value)
        return below / len(fzoo_values)

    fzoo_ics = [e.rank_ic for e in entries]
    fzoo_irs = [e.rank_ir for e in entries]
    fzoo_divs = [e.diversity for e in entries]
    fzoo_turns = [e.turnover for e in entries]

    return {
        "effectiveness": _pctile(node.rank_ic_mean or 0.0, fzoo_ics),
        "stability": _pctile(node.stability or 0.0, fzoo_irs),
        "diversity": _pctile(node.diversity or 0.0, fzoo_divs),
        "turnover": _pctile(node.turnover or 0.0, fzoo_turns),
    }
