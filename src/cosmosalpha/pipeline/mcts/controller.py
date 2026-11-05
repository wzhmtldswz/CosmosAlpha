"""MCTS controller — orchestrates Selection, Expansion, and Backpropagation."""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Protocol

import numpy as np

from cosmosalpha.agents import (
    EvaluationResult,
    ExperimentLesson,
    ExperimentStatus,
    FactorPreprocessingResult,
    FactorSpec,
    ResearchContext,
    ValidationReport,
)
from cosmosalpha.core import (
    ArtifactRef,
    ObjectRef,
    decode_record_artifact,
    records_to_aligned_values,
    stable_sha256,
)
from cosmosalpha.core.logging import LOGGER, contextual, error_code
from cosmosalpha.data import DataSnapshot
from cosmosalpha.evaluation._statistics import mean_daily_cross_sectional_pearson
from cosmosalpha.factors.expression import FieldCatalog, OperatorCatalog
from cosmosalpha.llm.client import LLMClient, LLMRequest
from cosmosalpha.llm.prompts import PromptRenderResult
from cosmosalpha.pipeline.loop import CosmosAlphaLoop, CosmosAlphaLoopResult
from cosmosalpha.pipeline.mcts.core import MCTSNode, MCTSTree
from cosmosalpha.pipeline.mcts.fzoo import (
    FzooEntry,
    should_admit_to_fzoo,
)
from cosmosalpha.pipeline.mcts.portrait import (
    AlphaPortrait,
    build_fzoo_portraits,
    build_portrait_prompt,
    generate_portrait,
)
from cosmosalpha.pipeline.mcts.root import ChainRootGenerator, RootGenerator

# When using overall_score [0,1], use 1.0
_OVERALL_SCALE = 1.0

# ── 改进：生产高 IC 正因子（需求 1/2/3） ──
_STRONG_IC = 0.04  # 优质因子阈值：强制 3 子节点 + 追加预算（需求3）
_DEEP_DIVE_MAX = 4  # 高分深挖队列上限（v15o 均衡档：8→4，压缩同族深挖，预算流向新分支）
_DEEP_DIVE_DIVERSITY = (
    0.7  # B1: 深挖需 diversity >= 0.7（v15o 均衡档：0.5→0.7，同族变体不深挖）
)
_DEEP_DIVE_IC_STEP = (
    0.008  # A1: 子 IC 需超父 +0.008 才继续深挖（v15o 均衡档：0.005→0.008）
)
_FZOO_MAX_CORR = 0.75  # B2: 与 Fzoo |r| >= 0.75 拒绝入库/入队（v11 平衡：0.7→0.75）
_FZOO_REPLACE_IC_STEP = 0.003  # 准入替换：新因子 |IC| 需超同质旧因子 +0.003 才替换
_TREE_MAX_ROUNDS = 10  # A3: 单树最多 10 轮（v15o 均衡档：15→10，单树更早收束 → 同预算多开树 → 跨树多样化）


class ExperimentRepository(Protocol):
    """Minimal object repository for resolving evaluation results."""

    def get_object[ObjectT](
        self, ref: ObjectRef, target_type: type[ObjectT]
    ) -> ObjectT:
        """Resolve one persisted domain object."""

    def put_object(self, object_type: str, object_id: str, value: object) -> ObjectRef:
        """Persist a versioned object and return its reference."""


class ArtifactRepo(Protocol):
    """Minimal artifact repository for reading factor-value NPZ artifacts."""

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Read the immutable factor-value artifact."""


# ------------------------------------------------------------------
# Multi-dimensional scoring (论文 §3.3)
# ------------------------------------------------------------------

_EFFECTIVENESS_WEIGHT = 0.5
_STABILITY_WEIGHT = 0.25
_DIVERSITY_WEIGHT = 0.25
_TURNOVER_WEIGHT = 0.0  # 用户决策：不考虑换手率（0.5+0.25+0.25=1.0）

_DIMENSIONS = ("effectiveness", "stability", "diversity", "turnover")

# 各维度综合得分权重（用户决策：不考虑换手率 → turnover 权重为 0）。
# _WEIGHTED_DIMENSIONS 只含权重 > 0 的维度：维度选择（_select_dimension）与
# UCT 维度覆盖多样性（_uct_select）都只在这些有效维度上运作，保证系统一致。
_DIMENSION_WEIGHTS: dict[str, float] = {
    "effectiveness": _EFFECTIVENESS_WEIGHT,
    "stability": _STABILITY_WEIGHT,
    "diversity": _DIVERSITY_WEIGHT,
    "turnover": _TURNOVER_WEIGHT,
}
_WEIGHTED_DIMENSIONS = tuple(
    d for d in _DIMENSIONS if _DIMENSION_WEIGHTS.get(d, 0.0) > 0.0
)


def _sigmoid_ic(ic: float) -> float:
    """Sigmoid: IC≤0→0,  IC>0 mapped through logistic centred at 0.025.

    Scaling factor -80 ensures:
      IC=0.01 → 0.23   IC=0.03 → 0.60   IC=0.05 → 0.88   IC=0.08 → 0.99
    """
    if ic <= 0.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-80.0 * (ic - 0.025))))


def _compute_composite_score(
    node: MCTSNode,
    fzoo_entries: dict[str, FzooEntry] | None = None,
) -> float:
    """论文 Eq.1：加权复合评分。有 Fzoo 时用百分位排名，否则用绝对值."""
    if fzoo_entries:
        from cosmosalpha.pipeline.mcts.fzoo import compute_fzoo_percentile_score

        pct = compute_fzoo_percentile_score(node, fzoo_entries)
        return (
            _EFFECTIVENESS_WEIGHT * pct["effectiveness"]
            + _STABILITY_WEIGHT * pct["stability"]
            + _DIVERSITY_WEIGHT * pct["diversity"]
            + _TURNOVER_WEIGHT * pct["turnover"]
        )
    # Fallback: absolute scores
    eff = _sigmoid_ic(node.rank_ic_mean or 0.0)
    sta = min(max((node.stability or 0.0) / 1.0, 0.0), 1.0)
    div = node.diversity or 0.0
    tur = _gaussian_score(node.turnover or 0.0)
    return (
        _EFFECTIVENESS_WEIGHT * eff
        + _STABILITY_WEIGHT * sta
        + _DIVERSITY_WEIGHT * div
        + _TURNOVER_WEIGHT * tur
    )


def _compute_turnover(
    factor_values: np.ndarray,
    date_groups: dict[int, np.ndarray],
    lag: int = 5,
) -> float:
    """Lag-L 日间截面秩自相关系数均值。lag=5 衡量周度稳定性."""
    sorted_keys = sorted(date_groups.keys())
    correlations: list[float] = []
    for i in range(len(sorted_keys) - lag):
        prev = _safe_rank(factor_values[date_groups[sorted_keys[i]]])
        cur = _safe_rank(factor_values[date_groups[sorted_keys[i + lag]]])
        if len(prev) >= 4 and len(cur) >= 4:
            overlap = np.intersect1d(
                np.arange(len(prev)),
                np.arange(len(cur)),
            )
            if len(overlap) >= 4:
                corr = np.corrcoef(prev[overlap], cur[overlap])[0, 1]
                if np.isfinite(corr):
                    correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else 0.0


def _gaussian_score(x: float, mu: float = 0.5, sigma: float = 0.25) -> float:
    """Gaussian scoring: peak at mu, penalties at extremes.

    Used for turnover: rewards moderate turnover (ρ≈0.6),
    penalises ultra-stable (ρ≈0.95) and ultra-noisy (ρ≈0.15) factors.
    """
    return float(math.exp(-((x - mu) ** 2) / (2.0 * sigma**2)))


def _safe_rank(values: np.ndarray) -> np.ndarray:
    """Rank values, NaN-safe."""
    valid = np.isfinite(values)
    result = np.full_like(values, np.nan, dtype=float)
    if np.count_nonzero(valid) >= 2:
        result[valid] = np.argsort(np.argsort(values[valid])).astype(float) / (
            np.count_nonzero(valid) - 1
        )
    return result


def _extract_eval_scores(
    result: CosmosAlphaLoopResult,
    objects: ExperimentRepository,
) -> dict[str, float | None]:
    """一次性提取 rank_ic、stability (IC_IR)、diversity (top3-mean).

    合并 _extract_rank_ic + _extract_multi_dim_scores，避免重复 get_object。
    """
    record = result.experiment_record
    if record is None:
        return {"rank_ic": None, "stability": 0.0, "diversity": 0.0}
    refs = record.evaluation_result_refs
    if not refs:
        return {"rank_ic": None, "stability": 0.0, "diversity": 0.0}
    eval_result = objects.get_object(refs[0], EvaluationResult)
    metrics = {m.name: m.value for m in eval_result.metrics}
    return {
        "rank_ic": metrics.get("rank_ic_mean"),
        "stability": float(metrics.get("rank_ic_ir", 0.0)),
        "diversity": float(1.0 - metrics.get("factor_max_abs_daily_correlation", 1.0)),
    }


@dataclass
class MCTSResult:
    """Output of one MCTS search run."""

    tree: MCTSTree
    iterations: int
    best_rank_ic: float | None


class MCTSController:
    """Monte Carlo Tree Search controller for factor expression discovery.

    Each iteration selects a node via UCT, expands it by running one full
    CosmosAlphaLoop, and backpropagates the resulting reward up the tree.
    """

    def __init__(
        self,
        loop: CosmosAlphaLoop,
        experiment_repository: ExperimentRepository,
        artifact_repository: ArtifactRepo,
        *,
        llm_client: LLMClient | None = None,
        field_catalog: FieldCatalog,
        operator_catalog: OperatorCatalog,
        exploration_weight: float = 1.0,
        ic_scale: float = _OVERALL_SCALE,
        alpha_threshold: float = 0.02,
        model_alias: str = "mcts",
        correlation_min_cross_section_samples: int = 20,
        correlation_min_daily_effective_stock_ratio: float = 0.8,
        correlation_min_pair_window_ratio: float = 0.8,
    ) -> None:
        """Bind the shared Loop, repositories, Catalogs, and search parameters."""
        self._loop = loop
        self._objects = experiment_repository
        self._artifacts = artifact_repository
        self._llm = llm_client
        self._field_catalog = field_catalog
        self._operator_catalog = operator_catalog
        self._exploration_weight = exploration_weight
        self._ic_scale = ic_scale
        self._alpha_threshold = alpha_threshold
        self._model_alias = model_alias
        if (
            type(correlation_min_cross_section_samples) is not int
            or correlation_min_cross_section_samples < 2
        ):
            raise ValueError("correlation_min_cross_section_samples must be >= 2")
        for name, value in (
            (
                "correlation_min_daily_effective_stock_ratio",
                correlation_min_daily_effective_stock_ratio,
            ),
            ("correlation_min_pair_window_ratio", correlation_min_pair_window_ratio),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        self._correlation_min_samples = correlation_min_cross_section_samples
        self._correlation_min_daily_ratio = float(
            correlation_min_daily_effective_stock_ratio
        )
        self._correlation_min_pair_window_ratio = float(
            correlation_min_pair_window_ratio
        )
        self._correlation_snapshot_id: str | None = None
        self._correlation_dates: tuple[str, ...] = ()
        self._correlation_eligible_counts: dict[str, int] = {}
        # P0-1: 跨树共享 FZoo（多树模式由 run_multi_tree 注入；单树为空）
        self._shared_zoo: dict[str, MCTSNode] = {}
        # 已评估因子计数（每次成功 build_node 即完成一次根/候选完整评估）
        self._evaluated_factor_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prepare_correlation_context(self, data_snapshot: DataSnapshot) -> None:
        """Load the run mask denominators used by Graph-style correlation."""
        if self._correlation_snapshot_id == data_snapshot.snapshot_id:
            return
        records = decode_record_artifact(
            self._artifacts.get_bytes(data_snapshot.mask_data.mask_data_ref)
        )
        dates: set[str] = set()
        eligible_counts: dict[str, int] = {}
        for record in records:
            date_value = record.get("datetime")
            eligible = record.get("eligible_mask")
            if type(date_value) is not str or not date_value:
                raise ValueError("mask_data contains an invalid datetime")
            if eligible is not None and type(eligible) is not bool:
                raise ValueError("mask_data contains an invalid eligible_mask")
            dates.add(date_value)
            if eligible is True:
                eligible_counts[date_value] = eligible_counts.get(date_value, 0) + 1
        if not dates:
            raise ValueError("mask_data contains no evaluation dates")
        self._correlation_snapshot_id = data_snapshot.snapshot_id
        self._correlation_dates = tuple(sorted(dates))
        self._correlation_eligible_counts = eligible_counts

    @contextual
    def run(
        self,
        *,
        seed_expression: str | None = None,
        root: MCTSNode | None = None,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        max_iterations: int = 50,
        initial_budget: int = 3,
        budget_increment: int = 3,
        breakthrough_relax: float = 0.95,
        run_id: str = "mcts-run",
        persist_id: str = "mcts-tree",
        shared_zoo: dict[str, MCTSNode] | None = None,
        min_root_ic: float = 0.0,
    ) -> MCTSResult:
        """Run MCTS search starting from a seed factor expression.

        Step 0 evaluates the seed expression to form the root node.
        Steps 1..max_iterations perform select → expand → backpropagate.

        Dynamic budget (论文 §D.4):
        - Initial budget = 3, each expansion consumes 1.
        - A "breakthrough" (overall_score >= previous_best × breakthrough_relax)
          adds budget_increment (default 3) rounds.
        - breakthrough_relax (default 0.95): 达到历史最佳 95% 即算突破,
          避免因微小波动错失探索机会。
        - Search ends when budget exhausted OR max_iterations reached.
        """
        root_started = perf_counter()
        LOGGER.info("MCTS root preparation started")
        if data_snapshot is not None:
            self._prepare_correlation_context(data_snapshot)
        # P0-1: 树内评分、few-shot 与 diversity 参照均合并共享 FZoo
        self._shared_zoo = shared_zoo if shared_zoo is not None else {}
        if root is None:
            # 未显式传入根：seed_expression 非 None → 单树 seed 根；
            # None → 完整链式挖掘根（RootGenerator 路径，需求2）。
            root = self._build_root(
                seed_expression=seed_expression,
                data_snapshot=data_snapshot,
                base_context=base_context,
                config_snapshot_ref=config_snapshot_ref,
                code_version=code_version,
                run_id=run_id,
            )
        # v15b：根不再做 IC 阈值筛选 —— 根只要生成成功即可开树，
        # 由树内 UCT + 评分自然筛选（min_root_ic 参数保留签名但不再拦截）。
        tree = MCTSTree(root=root)
        tree.all_nodes.append(root)
        self._tree = tree  # 供 _expand 查重使用
        self._admit_to_fzoo(root, tree)
        LOGGER.info(
            "MCTS root ready nodes=%d budget=%d elapsed=%.2fs",
            len(tree.all_nodes),
            initial_budget,
            perf_counter() - root_started,
        )

        # 动态预算（论文 §D.4）
        remaining_budget = initial_budget
        highest_score = root.overall_score or 0.0
        # improvement6 方案1：高分深挖优先队列（|IC|>=0.04 节点保证被继续深挖）
        deep_dive_queue: deque[MCTSNode] = deque()
        deep_dived: set[str] = set()
        # D1：方向枯竭标记（高分但深挖未显著提升 → 后续展开要求换结构）
        stale: set[str] = set()
        self._stale = stale

        completed_rounds = 0
        for iteration in range(1, max_iterations + 1):
            if remaining_budget <= 0:
                break
            round_started = perf_counter()
            LOGGER.info(
                "MCTS round started index=%d/%d nodes=%d budget=%d",
                iteration,
                max_iterations,
                len(tree.all_nodes),
                remaining_budget,
            )

            # ── 方案1：高分深挖优先（队列非空跳过 UCT）──
            if deep_dive_queue:
                selected = deep_dive_queue.popleft()
                num_candidates = 3  # 深挖强制 3 子节点
                budget_free = False  # A2: 深挖也消耗预算（每候选 -1，树自然收束）
                deep_dived.add(selected.factor_id)
            else:
                selected = self._uct_select(root, self._exploration_weight)
                # 需求3：优质分支（父 |RankIC| > 0.04）强制分出 3 个子节点（不计入预算），
                # 普通分支 2 个子节点（各计 1 预算）
                budget_free = abs(selected.rank_ic_mean or 0.0) > _STRONG_IC
                num_candidates = 3 if budget_free else 2

            for candidate_idx in range(num_candidates):
                LOGGER.info(
                    "MCTS expansion started round=%d candidate=%d/%d parent=%s",
                    iteration,
                    candidate_idx + 1,
                    num_candidates,
                    selected.factor_id[:12],
                )
                child = self._expand(
                    parent=selected,
                    data_snapshot=data_snapshot,
                    base_context=base_context,
                    config_snapshot_ref=config_snapshot_ref,
                    code_version=code_version,
                    run_id=run_id,
                    iteration=iteration,
                    candidate_index=candidate_idx,
                )
                if child is None:
                    LOGGER.info(
                        "MCTS expansion skipped round=%d candidate=%d reason=NO_NEW_NODE",
                        iteration,
                        candidate_idx + 1,
                    )
                    continue
                tree.add_node(selected, child)
                self._backpropagate(child)
                self._admit_to_fzoo(child, tree)

                # 优质分支/深挖轮次的子节点不计入预算
                if not budget_free:
                    remaining_budget -= 1
                child_score = child.overall_score or 0.0
                parent_score = selected.overall_score or 0.0

                # 父节点冷却：差子累冷却，好子清零恢复
                if child_score < parent_score * 0.7:
                    selected.cooling += 1
                elif child_score > parent_score * 1.05:
                    selected.cooling = 0

                # 全局突破奖励（保留预算激励，突破新高 +budget_increment）
                if child_score >= highest_score * breakthrough_relax:
                    highest_score = max(highest_score, child_score)
                    remaining_budget += budget_increment

                # 需求3：优质因子（父或子 |IC|>0.04）→ 该树预算强制 +3。
                if budget_free or abs(child.rank_ic_mean or 0.0) > _STRONG_IC:
                    remaining_budget += 3

                # ── 方案1 + 同质化治理（B1/B2/A1/D1）──
                child_ic = abs(child.rank_ic_mean or 0.0)
                parent_ic = abs(selected.rank_ic_mean or 0.0)
                ic_improved = child_ic >= parent_ic + _DEEP_DIVE_IC_STEP  # A1
                max_corr = self._max_fzoo_corr(child)  # B2 与 Fzoo 最大 |r|
                if (
                    child_ic >= _STRONG_IC  # 高分
                    and (child.diversity or 0.0) >= _DEEP_DIVE_DIVERSITY  # B1 不冗余
                    and max_corr < _FZOO_MAX_CORR  # B2 不同质
                    and ic_improved  # A1 真改进（深挖沿持续改进路径）
                    and child.factor_id not in deep_dived
                    and len(deep_dive_queue) < _DEEP_DIVE_MAX
                ):
                    deep_dive_queue.append(child)
                # D1：高分但 IC 未显著提升 → 方向枯竭，后续展开要求换结构
                if child_ic >= _STRONG_IC and not ic_improved:
                    stale.add(child.factor_id)

                if remaining_budget <= 0:
                    break
            completed_rounds += 1
            LOGGER.info(
                "MCTS round completed index=%d/%d nodes=%d budget=%d elapsed=%.2fs",
                iteration,
                max_iterations,
                len(tree.all_nodes),
                remaining_budget,
                perf_counter() - round_started,
            )

        self._persist_tree(tree, persist_id)
        LOGGER.info(
            "MCTS stopped reason=%s rounds=%d nodes=%d budget=%d",
            "BUDGET_EXHAUSTED" if remaining_budget <= 0 else "ITERATION_LIMIT_REACHED",
            completed_rounds,
            len(tree.all_nodes),
            remaining_budget,
        )
        actual_iterations = iteration  # 捕获循环终止时的实际值
        return MCTSResult(
            tree=tree,
            iterations=actual_iterations,
            best_rank_ic=(
                tree.root.best_reward if tree.root.best_reward > float("-inf") else None
            ),
        )

    def run_multi_tree(
        self,
        *,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        max_total_rounds: int = 300,
        max_evaluated_factors: int | None = None,
        initial_budget: int = 4,
        budget_increment: int = 3,
        breakthrough_relax: float = 0.95,
        run_id: str = "mcts-multi",
        root_generator: RootGenerator | None = None,
    ) -> MCTSResult:
        """Run successive chain-rooted trees with a shared factor zoo.

        根节点 = 一次完整链式 loop（HypothesisAgent→FactorDesign→
        FactorExpression→校验修复→因子计算→评价→FeedbackAgent 生成 Lesson），
        无任何预定义种子约束（种子机制已取消，见 RootGenerator 接口）。
        每个生成的根都会开树搜索，由树内 UCT + Fzoo 准入自然筛选。
        """
        shared_zoo: dict[str, MCTSNode] = {}
        # ── v4: 加载历史 Fzoo（跨 session 持久化） ──
        shared_zoo = self._load_history_fzoo(shared_zoo)
        tree_summaries: list[dict[str, object]] = []
        best_overall = 0.0
        total_rounds = 0
        tree_index = 0
        generator = root_generator or ChainRootGenerator(self)

        while total_rounds < max_total_rounds:
            tree_index += 1
            tree_run_id = f"{run_id}-tree{tree_index:03d}"
            try:
                root = generator.generate(
                    data_snapshot=data_snapshot,
                    base_context=base_context,
                    config_snapshot_ref=config_snapshot_ref,
                    code_version=code_version,
                    run_id=tree_run_id,
                )
                result = self.run(
                    root=root,
                    data_snapshot=data_snapshot,
                    base_context=base_context,
                    config_snapshot_ref=config_snapshot_ref,
                    code_version=code_version,
                    max_iterations=min(
                        max_total_rounds - total_rounds,
                        _TREE_MAX_ROUNDS,  # A3: 单树最多 12 轮，保证多树
                    ),
                    initial_budget=initial_budget,
                    budget_increment=budget_increment,
                    breakthrough_relax=breakthrough_relax,
                    run_id=tree_run_id,
                    persist_id=f"mcts-tree-{tree_index:03d}",
                    shared_zoo=shared_zoo,
                )
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    f"Tree {tree_index} failed, skipping to next chain root"
                )
                # 链式根生成或树搜索失败也计 1 轮，避免无限重试
                total_rounds += 1
                if total_rounds >= max_total_rounds:
                    break
                continue
            # 根被拒（iterations=0，未开树）也计 1 轮：根评估本身是一次完整
            # loop 成本，且防止连续生成不合格根时无限重试烧 API。
            total_rounds += result.iterations if result.iterations > 0 else 1

            # 记录每棵树的摘要
            tree_summaries.append(
                {
                    "tree_index": tree_index,
                    "iterations": result.iterations,
                    "nodes": len(result.tree.all_nodes),
                    "zoo_size": len(result.tree.alpha_zoo),
                    "best_rank_ic": result.best_rank_ic,
                }
            )

            # 合并到共享 Zoo
            for fid, node in result.tree.alpha_zoo.items():
                score = node.overall_score or 0.0
                if score > best_overall:
                    best_overall = score
                if fid not in shared_zoo or score > (
                    shared_zoo[fid].overall_score or 0.0
                ):
                    shared_zoo[fid] = node

            # ── 达到评估因子数上限则整体停止（与 max_total_rounds 谁先到谁停）──
            if (
                max_evaluated_factors is not None
                and self._evaluated_factor_count >= max_evaluated_factors
            ):
                import logging

                logging.getLogger(__name__).info(
                    "multi-tree stop: evaluated %d factors >= %d (rounds %d)",
                    self._evaluated_factor_count,
                    max_evaluated_factors,
                    total_rounds,
                )
                break

            if total_rounds >= max_total_rounds:
                break

        # 持久化合并后的跨树 Zoo
        merged_tree = MCTSTree(
            root=MCTSNode(factor_id="multi-root", expression="multi-tree"),
        )
        merged_tree.alpha_zoo = shared_zoo
        self._persist_tree(merged_tree, "mcts-multi-zoo")
        return MCTSResult(
            tree=merged_tree,
            iterations=total_rounds,
            best_rank_ic=best_overall,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # v4: Fzoo 跨 session 持久化
    # ------------------------------------------------------------------

    def _load_history_fzoo(
        self, shared_zoo: dict[str, MCTSNode]
    ) -> dict[str, MCTSNode]:
        """从持久化的历史 tree JSON 加载因子到 shared_zoo.

        读取 objects/mcts_tree/ 下的 mcts-tree-*.json 和 mcts-multi-zoo.json，
        将其中 overall_score > 0 的因子加载为 MCTSNode 加入 Fzoo。
        """
        try:
            from cosmosalpha.pipeline.mcts.core import MCTSTreeSnapshot

            # P0-2: get_object 需要 ObjectRef；先加载合并 Zoo，再加载单树
            # （单树默认 persist_id="mcts-tree"，多树为 mcts-tree-001..019）
            for obj_id in ["mcts-multi-zoo", "mcts-tree"] + [
                f"mcts-tree-{i:03d}" for i in range(1, 20)
            ]:
                try:
                    ref = ObjectRef("mcts_tree", obj_id, "mcts-tree/1.0")
                    snapshot = self._objects.get_object(ref, MCTSTreeSnapshot)
                except Exception:
                    if obj_id == "mcts-multi-zoo":
                        continue  # merged zoo not found → try individual trees
                    break  # no more trees
                if isinstance(snapshot, MCTSTreeSnapshot):
                    for node_data in snapshot.nodes:
                        self._add_to_zoo_from_data(
                            asdict(node_data),
                            shared_zoo,
                        )
                    if obj_id == "mcts-multi-zoo":
                        break  # merged zoo has everything
        except Exception:
            pass
        if shared_zoo:
            import logging

            logging.getLogger(__name__).info(
                "Loaded %d factors from history Fzoo", len(shared_zoo)
            )
        return shared_zoo

    @staticmethod
    def _add_to_zoo_from_data(
        node_data: dict[str, object], zoo: dict[str, MCTSNode]
    ) -> None:
        """从持久化 node dict 构建 MCTSNode 加入 zoo（已存在则跳过）."""
        fid = str(node_data.get("factor_id", ""))
        if not fid or fid in zoo:
            return
        ov = node_data.get("overall_score")
        if ov is not None and float(ov) <= 0:
            return
        node = MCTSNode(
            factor_id=fid,
            expression=str(node_data.get("expression", "")),
            rank_ic_mean=float(node_data.get("rank_ic_mean", 0) or 0),
            stability=float(node_data.get("stability", 0) or 0),
            diversity=float(node_data.get("diversity", 0) or 0),
            turnover=float(node_data.get("turnover", 0) or 0),
            overall_score=float(ov or 0),
            depth=int(node_data.get("depth", 0)),
            portrait=str(node_data.get("portrait", "")),
        )
        zoo[fid] = node

    # ------------------------------------------------------------------
    # v4: Fzoo 入库 — 四维标准替换旧 overall_score > 0.02
    # ------------------------------------------------------------------

    def _admit_to_fzoo(self, node: MCTSNode, tree: MCTSTree) -> bool:
        """v4 four-dimension admission criteria (paper §D.2): progressive threshold.

        On pass, add to tree.alpha_zoo. The score is fixed once at node creation
        (P0-2); admission never rewrites overall_score/best_reward, keeping the
        scoring basis consistent along the backpropagation chain.

        准入替换（upsert）：新因子与 Fzoo 已有因子同质（|r| >= _FZOO_MAX_CORR）
        时，若新因子 |IC| 显著更高（> _FZOO_REPLACE_IC_STEP），替换较差的旧因子
        （当前树 + 共享 Zoo 一并移除）；否则拒绝 —— 避免次优因子占位 + 丢失更优
        因子，Fzoo 始终保留每机制最优代表。
        """
        # P0-1: 准入门槛参照全局 FZoo（当前树 ∪ 共享 Zoo），写入仍落在当前树
        fzoo_dict = self._fzoo_union()
        # B2 + 准入替换：同质 peer 存在时，显著更优则替换，否则拒绝
        if fzoo_dict:
            max_corr, peer = self._find_max_corr_peer(node, tree)
            if max_corr >= _FZOO_MAX_CORR:
                if peer is None:
                    # 无 factor_values 无法定位 peer → 保守拒绝（旧行为）
                    return False
                node_ic = abs(node.rank_ic_mean or 0.0)
                peer_ic = abs(peer.rank_ic_mean or 0.0)
                if node_ic > peer_ic + _FZOO_REPLACE_IC_STEP:
                    # 替换：移除同质的次优旧因子，继续准入新因子
                    self._evict_from_fzoo(peer.factor_id, tree)
                    fzoo_dict.pop(peer.factor_id, None)
                    import logging

                    logging.getLogger(__name__).info(
                        "Fzoo upsert: replace %s (ic=%.4f) with %s (ic=%.4f)",
                        peer.factor_id[:10],
                        peer_ic,
                        node.factor_id[:10],
                        node_ic,
                    )
                else:
                    # 同质且不显著更优 → 拒绝（保留旧因子）
                    return False
        if should_admit_to_fzoo(node, fzoo_dict):
            tree.alpha_zoo[node.factor_id] = node
            return True
        return False

    def _find_max_corr_peer(
        self, node: MCTSNode, tree: MCTSTree
    ) -> tuple[float, MCTSNode | None]:
        """节点与 Fzoo（当前树 ∪ 共享 Zoo）最大 |r| 及对应 peer 节点.

        有 factor_values 时精确计算并返回 peer（供准入替换定位同质因子）；
        无 factor_values 时用创建时 diversity 镜像近似（无法定位 peer，返回 None）。
        """
        values = getattr(node, "factor_values", None)
        keys = getattr(node, "factor_keys", ())
        if values is None or len(values) < 2:
            return max(1.0 - (node.diversity or 1.0), 0.0), None
        max_corr = 0.0
        peer: MCTSNode | None = None
        zoo_nodes: dict[str, MCTSNode] = dict(self._shared_zoo)
        if tree is not None:
            zoo_nodes.update(tree.alpha_zoo)
        for zoo_node in zoo_nodes.values():
            zoo_values = getattr(zoo_node, "factor_values", None)
            zoo_keys = getattr(zoo_node, "factor_keys", ())
            if zoo_values is None or len(zoo_values) < 2:
                continue
            corr = self._max_abs_daily_correlation(values, keys, zoo_values, zoo_keys)
            if not np.isfinite(corr):
                continue
            if corr > max_corr:
                max_corr = corr
                peer = zoo_node
        return max_corr, peer

    def _evict_from_fzoo(self, factor_id: str, tree: MCTSTree) -> None:
        """从当前树与共享 Zoo 移除因子（准入替换用，跨树一致）."""
        tree.alpha_zoo.pop(factor_id, None)
        self._shared_zoo.pop(factor_id, None)

    def _max_fzoo_corr(self, node: MCTSNode) -> float:
        """B2：节点与 Fzoo 已有因子的最大 |r|（0~1）.

        有 factor_values 时直接精确计算；否则用创建时的 diversity 镜像
        （1 - diversity，近似）。
        """
        if (
            getattr(node, "factor_values", None) is not None
            and len(node.factor_values) >= 2
        ):
            return 1.0 - self._compute_zoo_diversity(
                node.factor_values, node.factor_keys
            )
        return max(1.0 - (node.diversity or 1.0), 0.0)

    def _persist_tree(self, tree: MCTSTree, persist_id: str = "mcts-tree") -> ObjectRef:
        """Serialize and store the MCTS tree for post-run analysis."""
        return self._objects.put_object(
            "mcts_tree",
            persist_id,
            tree.snapshot(),
        )

    # ------------------------------------------------------------------
    # Step 0 — seed root
    # ------------------------------------------------------------------

    def _build_root(
        self,
        *,
        seed_expression: str | None,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        run_id: str,
    ) -> MCTSNode:
        """通过一次完整链式 loop 构建根节点.

        - seed_expression 为 None：**自由链式挖掘**（HypothesisAgent 自由提出
          假设 → FactorDesign → FactorExpression → 校验修复 → 因子计算 →
          评价 → FeedbackAgent 生成 Lesson），不加任何 seed 约束 —— 多树模式
          的默认根生成（RootGenerator 路径）。
        - seed_expression 非 None：单树模式（run_mcts_live），约束
          HypothesisAgent 使用给定表达式作为因子。
        """
        if seed_expression:
            ctx = replace(
                base_context,
                constraints=(
                    "Use the provided seed expression as the factor.",
                    f"Seed expression: {seed_expression}",
                ),
            )
        else:
            # C1：多树模式——避免与 Fzoo 已有机制重复（跨树方向多样化）
            fzoo_exprs = [
                e.expression for e in self._fzoo_union().values() if e.expression
            ]
            extra_constraints: list[str] = []
            if fzoo_exprs:
                extra_constraints.append(
                    "Avoid reusing the financial mechanisms already explored:"
                )
                extra_constraints.extend(f"- {e[:120]}" for e in fzoo_exprs[:5])
            if extra_constraints:
                ctx = replace(
                    base_context,
                    constraints=base_context.constraints + tuple(extra_constraints),
                )
            else:
                ctx = base_context
        loop_result = self._loop.run(
            run_id=run_id,
            cycle_id="mcts-0000",
            context=ctx,
            data_snapshot=data_snapshot,
            config_snapshot_ref=config_snapshot_ref,
            code_version=code_version,
            parent_ids=(),
            existing_factor_refs=(),
            run_context=base_context,
            complete_run=False,
        )
        scores = _extract_eval_scores(loop_result, self._objects)
        expression = _extract_expression(loop_result, self._objects)
        factor_id = expression and stable_sha256(
            {"expression": expression, "frequency": base_context.frequency}
        )
        return self._build_node(
            factor_id=factor_id or "root",
            expression=expression or seed_expression or "chain-root",
            experiment_ref=loop_result.experiment_ref,
            rank_ic_mean=scores["rank_ic"],
            action_label="original",
            loop_result=loop_result,
            eval_scores=scores,
            frequency=base_context.frequency,
        )

    # ------------------------------------------------------------------
    # UCT Selection
    # ------------------------------------------------------------------

    _DIVERSITY_UCT_WEIGHT = 0.5  # 维度多样性奖励在 UCT 中的权重（v6: 加强维度引导）

    def _uct_select(self, node: MCTSNode, c: float) -> MCTSNode:
        """Recursively select a node using UCT + virtual expansion + dimension diversity.

        - 方案 C：虚拟扩展动作使用父节点 best_reward 作为利用项
        - 方案 D：维度多样性奖励 → 优先选维度覆盖不全的节点展开
        - improvement6 方案2：已剪枝分支（pruned）完全跳过，不消耗预算
        """
        while True:
            if not node.children:
                return node

            # 方案2：只考虑未剪枝的子节点；全被剪枝则就地返回（预算自然耗尽终止）
            live = [n for n in node.children if not n.pruned]
            if not live:
                return node
            unvisited = [n for n in live if n.visit_count == 0]
            if unvisited:
                return random.choice(unvisited)

            log_n_parent = math.log(node.visit_count)
            best_child = max(
                live,
                key=lambda n: n.best_reward / self._ic_scale
                + c * math.sqrt(log_n_parent / n.visit_count),
            )
            best_uct = best_child.best_reward / self._ic_scale + c * math.sqrt(
                log_n_parent / best_child.visit_count
            )

            # 方案 C: 虚拟动作 = 父节点利用项 + 探索项
            virtual_uct = node.best_reward / self._ic_scale + c * math.sqrt(
                log_n_parent / (1 + len(node.children))
            )
            # 父节点冷却：差子多 → virtual_UCT 降低 → 少生新分支
            cooling_factor = 1.0 / (1.0 + node.cooling)
            virtual_uct *= cooling_factor

            # 方案 D: 维度多样性奖励 — 维度覆盖越少的节点越应被展开
            # 深度衰减：深层节点降低多样性奖励，鼓励深入而非广度分叉
            used_dims = {
                child.action_dimension
                for child in node.children
                if child.action_dimension
            }
            uncovered = max(len(_WEIGHTED_DIMENSIONS) - len(used_dims), 0)
            diversity_bonus = (
                (uncovered / len(_WEIGHTED_DIMENSIONS)) * c * self._DIVERSITY_UCT_WEIGHT
            )
            depth_penalty = 1.0 / (1.0 + node.depth * 0.3)
            virtual_uct += diversity_bonus * depth_penalty

            if virtual_uct > best_uct:
                return node
            node = best_child

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _select_dimension(self, node: MCTSNode, temperature: float = 1.0) -> str:
        """论文 §3.2 Eq.3：全程 Softmax 选择目标改进维度（对齐论文，去掉 IC 门控）.

        只从权重>0 的维度（effectiveness / stability / diversity）中按
        P(i|s)=Softmax((e_max·1q − Es)/T) 加权随机选择 —— 低分（弱）维度被优先
        选中；不再有 "IC<0.04 只追 effectiveness" 的两段门控。
        """
        ic = node.rank_ic_mean or 0.0
        scores = {
            "effectiveness": _sigmoid_ic(ic),
            "stability": min(max((node.stability or 0.0) / 1.0, 0.0), 1.0),
            "diversity": node.diversity or 0.0,
        }
        dims = list(_WEIGHTED_DIMENSIONS)
        dim_values = [scores.get(d, 0.0) for d in dims]
        e_max = max(dim_values)
        exp_scores = [math.exp((e_max - s) / temperature) for s in dim_values]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        return random.choices(dims, weights=probs, k=1)[0]

    def _generate_alpha_portrait(
        self,
        parent: MCTSNode,
        dimension: str,
    ) -> AlphaPortrait | None:
        """v4: 论文两阶段生成 — Alpha Portrait → Formula.

        1. 从 Fzoo 采样 few-shot 示例
        2. 构建 Portrait Prompt
        3. 调用 LLM 生成 portrait + expression
        4. 解析并返回 AlphaPortrait，失败返回 None（回退到 suggestion）
        """
        if self._llm is None:
            return None

        # 1. Fzoo few-shot 采样（论文 §D.1 维度导向 + §G k=1）
        fzoo_entries = self._sample_few_shot_paper(parent, dimension)

        # 2. 构建 parent scores
        parent_scores = {}
        if parent.rank_ic_mean is not None:
            parent_scores["RankIC"] = parent.rank_ic_mean
        if parent.stability is not None:
            parent_scores["RankIR"] = parent.stability
        if parent.diversity is not None:
            parent_scores["Diversity"] = parent.diversity
        if parent.turnover is not None:
            parent_scores["Turnover"] = parent.turnover

        # 3. 构建 Prompt
        few_shot_portraits = build_fzoo_portraits(fzoo_entries)
        prompt_text = build_portrait_prompt(
            dimension=dimension,
            field_catalog=self._field_catalog,
            operator_catalog=self._operator_catalog,
            parent_expression=parent.expression,
            parent_scores=parent_scores if parent_scores else None,
            few_shot_examples=few_shot_portraits,
        )

        # 4. LLM 生成
        started = perf_counter()
        LOGGER.info("MCTS factor portrait started parent=%s", parent.factor_id[:12])
        portrait = generate_portrait(
            self._llm,
            prompt_text,
            model_alias=self._model_alias,
        )
        if portrait is None:
            LOGGER.warning(
                "MCTS factor portrait unavailable; using suggestion fallback elapsed=%.2fs",
                perf_counter() - started,
            )
        else:
            LOGGER.info(
                "MCTS factor portrait completed elapsed=%.2fs", perf_counter() - started
            )
        return portrait

    def _sample_few_shot_paper(
        self,
        parent: MCTSNode,
        dimension: str,
        k: int = 1,
    ) -> list[FzooEntry]:
        """论文 §D.1 维度导向 few-shot 采样（k=1 per §G Other Settings）.

        - effectiveness / stability：剔除与当前 alpha 数值相关最高的 top 50%
          （η=50%，论文 §D.1），再从剩余低相关候选按该维度分
          （rank_ic / rank_ir）取 top-k —— 兼顾多样性（结构别太像）与高分。
        - diversity：选与当前 alpha 数值相关最低的 top-k（鼓励新颖结构）。
        - turnover：zero-shot（无示例），依赖 LLM 内在理解。
        """
        fzoo_dict = self._fzoo_union()
        if not fzoo_dict:
            return []  # 冷启动：无示例
        if dimension == "turnover":
            return []  # zero-shot

        parent_values = getattr(parent, "factor_values", None)
        parent_keys = getattr(parent, "factor_keys", ())
        zoo_nodes: dict[str, MCTSNode] = dict(self._shared_zoo)
        tree = getattr(self, "_tree", None)
        if tree is not None:
            zoo_nodes.update(tree.alpha_zoo)

        # 计算每个 Fzoo 因子与当前父因子的真实数值相关（无法对齐 → 保守 1.0）
        scored: list[tuple[float, FzooEntry]] = []
        for fid, entry in fzoo_dict.items():
            corr = 1.0
            node = zoo_nodes.get(fid)
            if (
                parent_values is not None
                and len(parent_values) >= 2
                and node is not None
            ):
                node_values = getattr(node, "factor_values", None)
                node_keys = getattr(node, "factor_keys", ())
                if node_values is not None and len(node_values) >= 2:
                    c = self._max_abs_daily_correlation(
                        parent_values, parent_keys, node_values, node_keys
                    )
                    if np.isfinite(c):
                        corr = c
            scored.append((corr, entry))

        if dimension == "diversity":
            # 相关最低的 top-k
            picked = sorted(scored, key=lambda t: t[0])[:k]
            return [e for _, e in picked]

        # effectiveness / stability：剔除最相关 top 50%（η=50%），再按该维分取 top-k
        scored_sorted = sorted(scored, key=lambda t: t[0], reverse=True)
        keep = scored_sorted[len(scored_sorted) // 2 :]
        metric = "rank_ic" if dimension == "effectiveness" else "rank_ir"
        picked = sorted(
            keep,
            key=lambda t: abs(getattr(t[1], metric) or 0.0),
            reverse=True,
        )[:k]
        return [e for _, e in picked]

    def _step_a_suggest(self, parent: MCTSNode) -> list[str]:
        """论文 §3.2: ask LLM for targeted improvement suggestions."""
        if self._llm is None:
            return []
        started = perf_counter()
        LOGGER.info("MCTS suggestion started parent=%s", parent.factor_id[:12])
        dim_scores = ""
        if parent.stability is not None:
            dim_scores = (
                f"Effectiveness: {_sigmoid_ic(parent.rank_ic_mean or 0.0):.2f}  "
                f"Stability: {parent.stability:.2f}  "
                f"Diversity: {1.0 - (parent.diversity or 1.0):.2f}  "
                f"Turnover: {parent.turnover or 0.0:.2f}"
            )
        dimension = self._select_dimension(parent)
        prompt_text = (
            f"Analyze this factor and suggest 2-3 concrete improvements "
            f"focused on: {dimension}.\n\n"
            f"Factor: {parent.expression}\n"
            f"{dim_scores}\n\n"
            f"Output exactly a JSON array of strings."
        )
        try:
            response = self._llm.complete(
                LLMRequest(
                    prompt=PromptRenderResult(
                        prompt_render_id="mcts-suggestion",
                        prompt_id="suggestion",
                        template_version="1.0",
                        template_checksum="mcts",
                        input_schema="suggestion/1.0",
                        output_schema="suggestion/1.0",
                        messages=(),
                        output_json_schema={
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        estimated_input_tokens=len(prompt_text) // 4,
                        token_estimation_method="char_div_4",
                        truncation_report="none",
                        schema_version="prompt-render-result/1.0",
                    ),
                    model_alias=self._model_alias,
                    max_output_tokens=300,
                    response_format="json_object",
                )
            )
            suggestions = _parse_suggestions(response.content)
            LOGGER.info(
                "MCTS suggestion completed count=%d elapsed=%.2fs",
                len(suggestions),
                perf_counter() - started,
            )
            return suggestions
        except Exception as error:
            LOGGER.warning(
                "MCTS suggestion failed error=%s; continuing without suggestions elapsed=%.2fs",
                error_code(error),
                perf_counter() - started,
            )
            return []

    def _expand(
        self,
        *,
        parent: MCTSNode,
        data_snapshot: DataSnapshot,
        base_context: ResearchContext,
        config_snapshot_ref: ArtifactRef,
        code_version: str,
        run_id: str,
        iteration: int,
        candidate_index: int = 0,
    ) -> MCTSNode | None:
        """Build a refinement ResearchContext, run one loop, return new node.

        Step A (if llm_client is set): ask LLM for improvement suggestions.
        Step B: inject suggestions into constraints → loop.run().
        """
        # Step A: generate refinement suggestions (论文 §3.2 Eq.4-5)
        dimension = self._select_dimension(parent)
        parent_ic = parent.rank_ic_mean if parent.rank_ic_mean is not None else 0.0
        suffix_parts: list[str] = [
            f"Focus dimension: {dimension}",
            f"Parent factor: {parent.expression}",
        ]
        if abs(parent_ic) >= _STRONG_IC:
            suffix_parts.append(
                f"\nNOTE: Parent RankIC = {parent_ic:.4f} is STRONG (>= {_STRONG_IC}).\n"
                "Keep its core mechanism intact and make a SMALL, targeted "
                "improvement to push it even higher. Preserve the winning signal."
            )
        if candidate_index > 0:
            # 注意：不要引用 "previous candidate"（模型看不到同批前一个兄弟
            # 的具体内容，会陷入 "previous candidate 是谁" 的推理死循环，
            # 把 max_tokens 全烧在 reasoning 上返回空 content）。
            # 只以 prompt 中可见的 parent 为参照，并明示不要推测其它候选。
            suffix_parts.append(
                f"\nCandidate #{candidate_index + 1}: design a structurally "
                f"distinct refinement relative to the parent factor above "
                f"(different operator topology or different input fields). "
                f"Reference only the parent factor; do not speculate about "
                f"other candidates in this batch."
            )
        if parent.rank_ic_mean is not None:
            suffix_parts.append(f"RankIC={parent.rank_ic_mean:.4f}")
        if parent.stability is not None:
            diversity = (
                f"{parent.diversity:.2f}" if parent.diversity is not None else "N/A"
            )
            turnover = (
                f"{parent.turnover:.2f}" if parent.turnover is not None else "N/A"
            )
            suffix_parts.append(
                f"Scores: stability={parent.stability:.2f} "
                f"diversity={diversity} turnover={turnover}"
            )

        # 退化检测：若连续 2 步 IC 下降，写入 warning
        if _detect_degradation(parent):
            suffix_parts.append(
                "\nWARNING: This branch's RankIC has been declining for "
                "2+ consecutive refinement steps. The current direction may "
                "be exhausted. Consider a FUNDAMENTALLY DIFFERENT approach "
                "rather than incremental tweaking."
            )

        # D1：父节点方向枯竭（高分深挖未显著提升）→ 换结构提示
        stale_set = getattr(self, "_stale", None)
        if stale_set and parent.factor_id in stale_set:
            suffix_parts.append(
                "\nCRITICAL: This direction has been FULLY EXPLORED — no new "
                "factor can be generated from it.\n"
                "Generate a COMPLETELY DIFFERENT factor with DIFFERENT operator "
                "topology, DIFFERENT input fields, and a DIFFERENT financial "
                "mechanism. Do NOT tweak the current one incrementally."
            )

        # Inject full operator catalog and window rules to prevent LLM mistakes
        op_names = tuple(op.name for op in self._operator_catalog.operators)
        window_vals = sorted(
            set(
                w
                for op in self._operator_catalog.operators
                for w in op.supported_windows
            )
        )
        window_str = ", ".join(str(w) for w in window_vals)
        suffix_parts.append(
            "\nHARD CONSTRAINTS — violate these and your output will be REJECTED:\n"
            f"  - Valid operators: {', '.join(op_names)}\n"
            f"  - Valid window sizes: {window_str} (ONLY these exact integers)\n"
            f"  - Window argument MUST be one of: {window_str}\n"
            "  - No other window values, no invented operators, no infix operators"
        )

        portrait: AlphaPortrait | None = None
        if self._llm is not None:
            # ── v4: Alpha Portrait → Formula 两阶段生成 ──
            portrait = self._generate_alpha_portrait(parent, dimension)
            if portrait is not None:
                suffix_parts.insert(
                    0, f"Pre-generated Alpha Portrait:\n{portrait.to_constraint_text()}"
                )
                parent.portrait = portrait.to_constraint_text()
            # ── 回退：传统 suggestion 机制 ──
            else:
                suggestions = self._step_a_suggest(parent)
                if suggestions:
                    suffix_parts.append("\nRefinement suggestions:")
                    for s in suggestions[:3]:
                        suffix_parts.append(f"  - {s}")

        parent_summary = _format_node_summary(parent)
        refinement_history = _build_refinement_history(parent)
        suffix = "\n".join(suffix_parts)
        # 过滤空/空白字符串，避免 _require_string_tuple 校验失败
        history_items: list[str] = []
        if parent_summary.strip():
            history_items.append(parent_summary.strip())
        if refinement_history and refinement_history.strip():
            history_items.append(refinement_history.strip())

        # ── expand 执行完整链式（含真实 HypothesisAgent）：对齐 Graph 的
        #    研究链语义，让假设为 design 提供方向聚焦，而非合成假设跳过 ──
        # 历史 Lesson 注入：精简到最近 2 条父/祖 lesson（v15i 全砍导致产出质量
        # 塌陷：无方向指引；v15h 全量 6 条导致 hypothesis 失败率升。取 2 条
        # 折中：保留方向价值、控制 hypothesis 推理负担）。
        parent_experiment_id = (
            parent.experiment_ref.object_id if parent.experiment_ref else None
        )
        _lesson_text, lesson_refs = _build_mcts_history_lesson_text(
            self._objects,
            parent_experiment_id,
            max_items=2,
        )

        # ── 结构化父因子上下文注入（对齐 Graph GRAPH_GUIDANCE_JSON）──
        # FactorDesign 直接拿到 parent expression / focus dimension / scores，
        # 无需从假设散文推断优化目标（此前信息断裂导致推理穷举烧满）。
        parent_scores_guidance: dict[str, float] = {}
        if parent.rank_ic_mean is not None:
            parent_scores_guidance["RankIC"] = float(parent.rank_ic_mean)
        if parent.stability is not None:
            parent_scores_guidance["Stability"] = float(parent.stability)
        if parent.diversity is not None:
            parent_scores_guidance["Diversity"] = float(parent.diversity)
        if parent.turnover is not None:
            parent_scores_guidance["Turnover"] = float(parent.turnover)
        parent_guidance_json = json.dumps(
            {
                "parent_expression": parent.expression,
                "focus_dimension": dimension,
                "parent_scores": parent_scores_guidance,
            },
            ensure_ascii=False,
        )
        ctx = replace(
            base_context,
            constraints=base_context.constraints
            + (suffix, f"MCTS_PARENT_GUIDANCE_JSON={parent_guidance_json}"),
            history_summaries=tuple(history_items),
        )
        try:
            loop_result = self._loop.run(
                run_id=run_id,
                cycle_id=(
                    f"mcts-{iteration:04d}"
                    if candidate_index == 0
                    else f"mcts-{iteration:04d}-c{candidate_index + 1}"
                ),
                context=ctx,
                data_snapshot=data_snapshot,
                config_snapshot_ref=config_snapshot_ref,
                code_version=code_version,
                parent_ids=(
                    (parent.experiment_ref.object_id,) if parent.experiment_ref else ()
                ),
                existing_factor_refs=(),
                # 精简 lessons（最近 2 条父/祖）给 hypothesis/design 方向参考
                history_lesson_refs=lesson_refs,
                run_context=base_context,
                complete_run=False,
                # 真实 HypothesisAgent 执行（默认 skip_hypothesis=False）：
                # expand 每候选先由 LLM 产出研究假设，再进入 FactorDesign。
                skip_feedback=False,
            )
        except Exception as error:
            LOGGER.warning(
                "MCTS candidate failed error=%s; continuing search", error_code(error)
            )
            return None
        scores = _extract_eval_scores(loop_result, self._objects)
        expression = _extract_expression(loop_result, self._objects)
        if expression is None:
            return None
        # 论文：所有 IsValid(f) 的因子都成为树节点，低分节点被 UCT 自然避开
        # 即使 IC 无法计算（数据不足），也创建节点，overall_score 默认为 0
        factor_id = stable_sha256(
            {"expression": expression, "frequency": base_context.frequency}
        )
        return self._build_node(
            factor_id=factor_id,
            expression=expression,
            experiment_ref=loop_result.experiment_ref,
            rank_ic_mean=scores["rank_ic"],
            action_label=f"refine:{parent.factor_id}",
            action_dimension=dimension,
            loop_result=loop_result,
            parent=parent,
            eval_scores=scores,
            portrait_text=portrait.to_constraint_text() if portrait else "",
            frequency=base_context.frequency,
        )

    # ------------------------------------------------------------------
    # Node construction with multi-dimensional scoring
    # ------------------------------------------------------------------

    def _build_node(
        self,
        *,
        factor_id: str,
        expression: str,
        experiment_ref: ObjectRef | None,
        rank_ic_mean: float | None,
        action_label: str,
        action_dimension: str = "",
        loop_result: CosmosAlphaLoopResult,
        parent: MCTSNode | None = None,
        eval_scores: dict[str, float | None] | None = None,
        portrait_text: str = "",
        frequency: str = "1d",
    ) -> MCTSNode:
        """Build a MCTSNode with multi-dimensional scores from loop result."""
        if eval_scores is None:
            eval_scores = _extract_eval_scores(loop_result, self._objects)
        turnover = self._compute_turnover_from_result(loop_result)
        # P1-5: diversity = 1 - max_abs_corr(against all current FZoo factors),
        # overriding the evaluator's top-3 mean approximation.
        factor_values, factor_keys = self._extract_factor_values_from_result(
            loop_result
        )
        diversity = self._compute_zoo_diversity(factor_values, factor_keys)
        node = MCTSNode(
            factor_id=factor_id,
            expression=expression,
            experiment_ref=experiment_ref,
            parent=parent,
            rank_ic_mean=rank_ic_mean,
            stability=eval_scores.get("stability"),
            diversity=diversity,
            turnover=turnover,
            action_label=action_label,
            action_dimension=action_dimension,
            portrait=portrait_text,
        )
        # Factor values are kept only on in-memory nodes so later nodes can
        # compute max correlation against the FZoo.
        node.factor_values = factor_values
        node.factor_keys = factor_keys
        # P0-2: score once at creation (percentile when FZoo is non-empty,
        # absolute otherwise) and never rewrite afterwards, so parents and
        # children share the same scoring basis within the tree.
        node.overall_score = _compute_composite_score(
            node, self._current_fzoo_entries()
        )
        node.best_reward = node.overall_score
        # 每次成功评估（根或展开候选）计数，供 run_multi_tree 按评估因子数停止
        self._evaluated_factor_count += 1
        return node

    def _compute_turnover_from_result(
        self, result: CosmosAlphaLoopResult
    ) -> float | None:
        """Read factor values from preprocessing artifact, compute turnover."""
        for candidate in result.candidate_results:
            if candidate.preprocessing_result_ref is not None:
                try:
                    pre_result = self._objects.get_object(
                        candidate.preprocessing_result_ref,
                        FactorPreprocessingResult,
                    )
                    ref = getattr(
                        pre_result, "input_factor_values_ref", None
                    ) or getattr(pre_result, "evaluated_factor_values_ref", None)
                    if ref is not None:
                        data = self._artifacts.get_bytes(ref)
                        records = decode_record_artifact(data)
                        return _compute_turnover_from_records(records)
                except Exception:
                    pass
        return None

    def _fzoo_union(self) -> dict[str, FzooEntry]:
        """Return the FzooEntry view over current tree alpha_zoo ∪ shared cross-tree zoo.

        P0-1: tree-internal percentile scoring, few-shot sampling and diversity
        references all consume the union, so multi-tree runs share the global
        FZoo instead of each tree starting from an empty zoo.
        """
        union: dict[str, FzooEntry] = {
            fid: FzooEntry.from_node(n) for fid, n in self._shared_zoo.items()
        }
        tree = getattr(self, "_tree", None)
        if tree is not None:
            for fid, n in tree.alpha_zoo.items():
                union[fid] = FzooEntry.from_node(n)
        return union

    def _current_fzoo_entries(self) -> dict[str, FzooEntry] | None:
        """Return the FzooEntry dict for scoring (current tree ∪ shared zoo).

        Used for percentile scoring at node creation; returns None when empty,
        falling back to absolute scoring.
        """
        union = self._fzoo_union()
        return union if union else None

    def _extract_factor_values_from_result(
        self, result: CosmosAlphaLoopResult
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Read factor values and alignment keys from preprocessing artifacts.

        Alignment keys are instrument@datetime (falling back to row{idx}@datetime
        without instrument) so max correlation against FZoo factors can be
        computed by aligning on stock and date.
        """
        for candidate in result.candidate_results:
            if candidate.preprocessing_result_ref is None:
                continue
            try:
                pre_result = self._objects.get_object(
                    candidate.preprocessing_result_ref,
                    FactorPreprocessingResult,
                )
                ref = getattr(
                    pre_result, "evaluated_factor_values_ref", None
                ) or getattr(pre_result, "input_factor_values_ref", None)
                if ref is None:
                    continue
                data = self._artifacts.get_bytes(ref)
                records = decode_record_artifact(data)
                values, keys = records_to_aligned_values(
                    records,
                    preserve_missing=True,
                )
                if values is not None:
                    return values, keys
            except Exception:
                continue
        return None, ()

    def _compute_zoo_diversity(
        self,
        values: np.ndarray | None,
        keys: tuple[str, ...],
    ) -> float:
        """P1-5: diversity = 1 - max |correlation| against all FZoo factors.

        Paper definition: diversity is based on the max correlation with FZoo
        factors; returns 1.0 (no redundancy) when the FZoo is empty or no
        alignable factor values exist.
        """
        if values is None or len(values) < 2:
            return 1.0
        max_corr = 0.0
        # P0-1: diversity 参照当前树 ∪ 共享 Zoo 的全部因子（跨树防冗余）
        tree = getattr(self, "_tree", None)
        zoo_nodes: dict[str, MCTSNode] = dict(self._shared_zoo)
        if tree is not None:
            zoo_nodes.update(tree.alpha_zoo)
        for zoo_node in zoo_nodes.values():
            zoo_values = getattr(zoo_node, "factor_values", None)
            zoo_keys = getattr(zoo_node, "factor_keys", ())
            if zoo_values is None or len(zoo_values) < 2:
                continue
            corr = self._max_abs_daily_correlation(values, keys, zoo_values, zoo_keys)
            if not np.isfinite(corr):
                continue
            max_corr = max(max_corr, corr)
        return 1.0 - max_corr

    def _max_abs_daily_correlation(
        self,
        values: np.ndarray,
        keys: tuple[str, ...],
        other_values: np.ndarray,
        other_keys: tuple[str, ...],
    ) -> float:
        """Apply the shared Graph daily-correlation semantics to two factors."""
        return _max_abs_daily_corr_aligned(
            values,
            keys,
            other_values,
            other_keys,
            eligible_counts=self._correlation_eligible_counts,
            evaluation_dates=self._correlation_dates,
            min_cross_section_samples=self._correlation_min_samples,
            min_daily_effective_stock_ratio=self._correlation_min_daily_ratio,
            min_pair_window_ratio=self._correlation_min_pair_window_ratio,
        )

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode) -> None:
        """Walk up the tree, updating best_reward and visit_count."""
        score = node.best_reward
        while node is not None:
            node.visit_count += 1
            if score > node.best_reward:
                node.best_reward = score
            node = node.parent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_expression(
    result: CosmosAlphaLoopResult,
    objects: ExperimentRepository,
) -> str | None:
    """Get the first validated factor's normalized expression."""
    # Priority 1: accepted factors from experiment record
    for candidate in result.candidate_results:
        if candidate.status == ExperimentStatus.ACCEPTED and candidate.factor_ref:
            spec = objects.get_object(candidate.factor_ref, FactorSpec)
            return spec.expression
    # Priority 2: validated factors from validation reports
    if result.experiment_record:
        for vref in result.experiment_record.validation_report_refs:
            vr = objects.get_object(vref, ValidationReport)
            if vr.is_valid:
                expr = vr.normalized_expression
                if expr:
                    return expr
    # Priority 3: any factor_ref from candidates
    for candidate in result.candidate_results:
        if candidate.factor_ref:
            spec = objects.get_object(candidate.factor_ref, FactorSpec)
            return spec.expression
    return None


def _compute_turnover_from_records(
    records: tuple[dict[str, object], ...],
) -> float:
    """Compute factor turnover from decoded artifact records."""
    if not records:
        return 0.0
    factor_values = np.array(
        [float(r.get("factor_value", np.nan) or np.nan) for r in records]
    )
    if not np.any(np.isfinite(factor_values)):
        return 0.0
    dates = [str(r.get("datetime", "")) for r in records]
    unique_dates = sorted(set(dates))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(dates):
        groups.setdefault(date_to_idx[d], []).append(i)

    group_indices = {k: np.array(v) for k, v in groups.items()}
    return _compute_turnover(factor_values, group_indices)


def _max_abs_daily_corr_aligned(
    values: np.ndarray,
    keys: tuple[str, ...],
    other_values: np.ndarray,
    other_keys: tuple[str, ...],
    *,
    eligible_counts: dict[str, int],
    evaluation_dates: tuple[str, ...],
    min_cross_section_samples: int,
    min_daily_effective_stock_ratio: float,
    min_pair_window_ratio: float,
) -> float:
    """Compute abs(mean daily cross-sectional Pearson) after key alignment."""
    if not evaluation_dates:
        return 0.0
    other_index = {k: i for i, k in enumerate(other_keys)}
    left_dates = {_alignment_key_date(key) for key in keys}
    right_dates = {_alignment_key_date(key) for key in other_keys}
    overlap_dates = tuple(sorted(set(evaluation_dates) & left_dates & right_dates))
    if (
        not overlap_dates
        or len(overlap_dates) / len(evaluation_dates) < min_pair_window_ratio
    ):
        return 0.0
    overlap_date_set = set(overlap_dates)
    pairs_by_date: dict[str, tuple[list[float], list[float]]] = {}
    for i, key in enumerate(keys):
        j = other_index.get(key)
        if j is not None:
            date_value = _alignment_key_date(key)
            if date_value not in overlap_date_set:
                continue
            left, right = pairs_by_date.setdefault(date_value, ([], []))
            left.append(float(values[i]))
            right.append(float(other_values[j]))
    daily_pairs = (
        (
            np.asarray(pairs_by_date.get(date_value, ([], []))[0], dtype=float),
            np.asarray(pairs_by_date.get(date_value, ([], []))[1], dtype=float),
            eligible_counts.get(date_value, 0),
        )
        for date_value in overlap_dates
    )
    correlation, _ = mean_daily_cross_sectional_pearson(
        daily_pairs,
        min_cross_section_samples=min_cross_section_samples,
        min_daily_effective_stock_ratio=min_daily_effective_stock_ratio,
    )
    return float(abs(correlation)) if math.isfinite(correlation) else 0.0


def _alignment_key_date(key: str) -> str:
    """Extract the date suffix from an ``instrument@datetime`` key."""
    if type(key) is not str or "@" not in key:
        raise ValueError("factor alignment keys must use instrument@datetime")
    return key.rsplit("@", maxsplit=1)[1]


def _detect_degradation(node: MCTSNode) -> bool:
    """Check if the branch has had 2+ consecutive IC declines AND current IC < 0.

    Only flags degradation when the current node already has negative IC —
    preserves the ability to traverse temporarily-declining-but-positive paths.
    """
    if (node.rank_ic_mean or 0.0) >= 0:
        return False
    current = node
    declining = 0
    while current.parent is not None:
        parent = current.parent
        if (
            current.rank_ic_mean is not None
            and parent.rank_ic_mean is not None
            and current.rank_ic_mean < parent.rank_ic_mean
        ):
            declining += 1
            if declining >= 2:
                return True
        else:
            break
        current = parent
    return False


def _build_refinement_history(node: MCTSNode) -> str:
    """Build a refinement history string from root to the given node.

    Walks up to root, reverses, and formats each step with expression
    and IC change. Returns empty string if the chain is too short.
    """
    chain: list[MCTSNode] = []
    current: MCTSNode | None = node
    while current is not None:
        chain.append(current)
        current = current.parent
    if len(chain) < 2:
        return ""
    chain.reverse()
    lines = ["Refinement history:"]
    for i in range(1, len(chain)):
        prev = chain[i - 1]
        cur = chain[i]
        prev_ic = f"{prev.rank_ic_mean:.4f}" if prev.rank_ic_mean is not None else "N/A"
        cur_ic = f"{cur.rank_ic_mean:.4f}" if cur.rank_ic_mean is not None else "N/A"
        if prev.rank_ic_mean is not None and cur.rank_ic_mean is not None:
            delta = cur.rank_ic_mean - prev.rank_ic_mean
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            change = f"{direction} {delta:+.4f}"
        else:
            change = ""
        action = cur.action_dimension or cur.action_label or "refine"
        lines.append(f"  Step {i} ({action}): {cur.expression}")
        lines.append(f"    IC: {prev_ic} → {cur_ic}  {change}")
    return "\n".join(lines).strip()


def _format_node_summary(node: MCTSNode) -> str:
    ric = f"{node.rank_ic_mean:.4f}" if node.rank_ic_mean is not None else "N/A"
    stability = f"{node.stability:.4f}" if node.stability is not None else "N/A"
    diversity = f"{node.diversity:.4f}" if node.diversity is not None else "N/A"
    turnover = f"{node.turnover:.4f}" if node.turnover is not None else "N/A"
    return (
        f"Factor: {node.expression}  "
        f"RankIC={ric}  stability={stability}  diversity={diversity}  "
        f"turnover={turnover}  visits={node.visit_count}"
    )


def _parse_suggestions(content: str) -> list[str]:
    """Parse LLM output into a list of suggestion strings.

    Handles three formats: JSON array, newline-separated list, single string.
    """
    import json as _json

    try:
        parsed = _json.loads(content)
        if isinstance(parsed, list):
            return [str(s) for s in parsed if s]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass
    # Fallback: split by newlines or bullet points
    lines = [line.strip(" -•*#0123456789.)") for line in content.split("\n")]
    return [line for line in lines if len(line) > 10]


# ------------------------------------------------------------------
# 历史 Lesson 注入（对齐链式挖掘：父/祖/兄弟优先，最多 6 条完整 Lesson）
# ------------------------------------------------------------------

_MCTS_HISTORY_LESSON_LIMIT = 6


def _format_lesson_for_mcts(
    lesson: ExperimentLesson,
    *,
    tag: str = "",
) -> str:
    """把一个结构化 Lesson 格式化为 MCTS 可读文本.

    注入内容（与链式挖掘一致）：
      - 假设机制 (hypothesis_mechanism)
      - 稳定失败码 (stable_failure_codes)
      - 关键指标 IC (key_metrics)
      - 经验教训 (effective_lessons) 与下一步建议 (next_actions)
    """
    prefix = f"[{tag}] " if tag else ""
    lines = [f"{prefix}Lesson {lesson.cycle_id} ({lesson.run_id}):"]
    if lesson.hypothesis_mechanism:
        lines.append(f"  机制: {lesson.hypothesis_mechanism}")
    if lesson.stable_failure_codes:
        lines.append("  失败码: " + ", ".join(lesson.stable_failure_codes))
    metrics = tuple(
        m for m in lesson.key_metrics if m.name in {"rank_ic_mean", "rank_ic_ir"}
    )
    for m in metrics[:4]:
        lines.append(f"  {m.name}={m.value:.4f} (n={m.sample_count})")
    if lesson.effective_lessons:
        for el in lesson.effective_lessons[:3]:
            lines.append(f"  经验: {el}")
    if lesson.next_actions:
        for na in lesson.next_actions[:2]:
            lines.append(f"  建议: {na}")
    return "\n".join(lines)


def _build_mcts_history_lesson_text(
    repository: ExperimentRepository,
    parent_experiment_id: str | None,
    max_items: int = _MCTS_HISTORY_LESSON_LIMIT,
) -> tuple[str, tuple[ObjectRef, ...]]:
    """选择历史 Lesson（父/祖/兄弟优先）并格式化为文本.

    返回 (lesson_text, lesson_refs)。lesson_refs 供 loop.run(history_lesson_refs=...)
    使用，与链式挖掘的 ContextBuilder 注入方式保持一致。
    """
    try:
        from cosmosalpha.pipeline.context import (
            _lineage_distances,
        )
        from cosmosalpha.pipeline.lesson import lesson_ref_for_experiment
    except Exception:
        return "", ()

    try:
        records = repository.list_experiments()
        completed = tuple(r for r in records if r.completed_at is not None)
    except Exception:
        return "", ()
    if not completed:
        return "", ()

    parent_ids: tuple[str, ...] = (
        (parent_experiment_id,) if parent_experiment_id else ()
    )
    lineage = _lineage_distances(completed, parent_ids)

    # 父/祖/兄弟优先：血缘距离近的排前，其次按完成时间新→旧
    def sort_key(record):
        dist = lineage.get(record.experiment_id, 1_000_000)
        ts = 0.0
        if record.completed_at is not None:
            ts = record.completed_at.timestamp()
        return (dist, -ts)

    ordered = sorted(completed, key=sort_key)
    selected_lessons: list[ExperimentLesson] = []
    selected_refs: list[ObjectRef] = []
    for record in ordered:
        if len(selected_lessons) >= max_items:
            break
        if parent_experiment_id and record.experiment_id == parent_experiment_id:
            continue  # 父节点本身在 MCTS 中已作为 Parent factor 提供
        try:
            ref = lesson_ref_for_experiment(record.experiment_id)
            lesson = repository.get_object(ref, ExperimentLesson)
        except Exception:
            continue
        if lesson is None:
            continue
        selected_lessons.append(lesson)
        selected_refs.append(ref)

    if not selected_lessons:
        return "", ()

    lines: list[str] = []
    for idx, lesson in enumerate(selected_lessons, start=1):
        # 血缘距离标注：0=父，1=祖/兄弟
        dist = lineage.get(lesson.experiment_id)
        tag = (
            "parent"
            if dist == 0
            else ("ancestor/sibling" if dist is not None else "history")
        )
        lines.append(_format_lesson_for_mcts(lesson, tag=f"hist{idx}:{tag}"))
    return "\n\n".join(lines), tuple(selected_refs)
