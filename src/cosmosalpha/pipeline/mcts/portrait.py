"""Alpha Portrait generation — two-stage factor design (论文 §3.1-3.2, §D.1).

Stage 1: Generate Alpha Portrait (name + logic + pseudo-code)
Stage 2: Generate Alpha Formula from portrait

With Fzoo few-shot exemplar support and temperature control.
"""

from __future__ import annotations

import json as _json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cosmosalpha.factors.expression import FieldCatalog, OperatorCatalog
from cosmosalpha.llm.client import LLMClient, LLMRequest
from cosmosalpha.llm.prompts import PromptRenderResult

# ------------------------------------------------------------------
# Temperature settings (论文 §D.3)
# ------------------------------------------------------------------

PORTRAIT_TEMPERATURE: float = 1.0  # 生成阶段：鼓励多样性

# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlphaPortrait:
    """Parsed Alpha Portrait from LLM generation."""

    name: str
    logic: str
    pseudo_code: str
    expression: str

    def to_constraint_text(self) -> str:
        """Format portrait as guidance for structured factor generation."""
        return (
            f"Alpha Portrait:\n"
            f"  Name: {self.name}\n"
            f"  Logic: {self.logic}\n"
            f"  Pseudo-code: {self.pseudo_code}\n"
            f"  Suggested Expression: {self.expression}"
        )

    def to_few_shot_text(self) -> str:
        """Format portrait as a few-shot example for the LLM prompt."""
        return (
            f"Example — {self.name}:\n"
            f"  Logic: {self.logic}\n"
            f"  Expression: {self.expression}"
        )


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------


def build_portrait_prompt(
    *,
    dimension: str,
    field_catalog: FieldCatalog,
    operator_catalog: OperatorCatalog,
    parent_expression: str = "",
    parent_scores: dict[str, float] | None = None,
    few_shot_examples: Sequence[AlphaPortrait] = (),
) -> str:
    """Build the Alpha Portrait generation prompt with Fzoo few-shot.

    Args:
        dimension: Target refinement dimension (effectiveness/stability/diversity/turnover).
        field_catalog: Available data fields.
        operator_catalog: Available operators.
        parent_expression: The parent node's expression (for refinement context).
        parent_scores: Parent's multi-dimensional scores.
        few_shot_examples: Fzoo few-shot exemplars (empty for cold start).

    Returns:
        Complete prompt text for the LLM.
    """
    field_names = ", ".join(
        f.name for f in field_catalog.fields if f.available and not f.context_only
    )
    op_names = ", ".join(op.name for op in operator_catalog.operators)
    windows = sorted(
        set(w for op in operator_catalog.operators for w in op.supported_windows)
    )
    windows_str = ", ".join(str(w) for w in windows)

    # Few-shot section
    few_shot_text = ""
    if few_shot_examples:
        examples_text = "\n".join(
            f"  {i+1}. {ex.to_few_shot_text()}"
            for i, ex in enumerate(few_shot_examples)
        )
        few_shot_text = (
            f"\nReference Examples (high-quality alphas from the repository):\n"
            f"{examples_text}\n"
            f"\nStudy these examples for structure and logic patterns, "
            f"but design a NOVEL factor.\n"
        )

    # Parent context
    parent_text = ""
    if parent_expression:
        parent_text = f"\nParent Factor to Improve: {parent_expression}\n"
        if parent_scores:
            score_lines = "\n".join(f"  {k}: {v:.3f}" for k, v in parent_scores.items())
            parent_text += f"Current Scores:\n{score_lines}\n"

    return (
        f"\n"
        f"You are an expert quantitative researcher designing alpha factors "
        f"for the Chinese A-share market.\n"
        f"\n"
        f"Available data fields (daily frequency): {field_names}\n"
        f"Available operators: {op_names}\n"
        f"Valid time windows: {windows_str} (ONLY these exact integers)\n"
        f"{few_shot_text}"
        f"{parent_text}"
        f"Target Dimension: {dimension}\n"
        f"\n"
        f"Step 1 — Alpha Portrait:\n"
        f"  - Factor Name: A concise, descriptive name\n"
        f"  - Investment Logic: Explain WHY this factor should predict future "
        f"stock returns. Reference specific market behaviors or anomalies.\n"
        f"  - Pseudo-code: Describe the computation in structured steps.\n"
        f"\n"
        f"Step 2 — Alpha Formula:\n"
        f"  Translate the portrait into a valid expression using ONLY the "
        f"operators listed above.\n"
        f"  Format: func_name(arg1, arg2, ...)\n"
        f"  Window arguments MUST be one of: {windows_str}\n"
        f"  Do NOT use infix operators (+, -, *, /). Use add, sub, mul, div.\n"
        f"\n"
        f"Output ONLY a JSON object:\n"
        f'{{"portrait_name": "...", "portrait_logic": "...", '
        f'"portrait_pseudocode": "...", "expression": "..."}}'
    )


# ------------------------------------------------------------------
# LLM interaction
# ------------------------------------------------------------------


def generate_portrait(
    llm: LLMClient,
    prompt_text: str,
    *,
    model_alias: str = "mcts",
    temperature: float = PORTRAIT_TEMPERATURE,
) -> AlphaPortrait | None:
    """Call LLM to generate an Alpha Portrait and Formula.

    Args:
        llm: LLM client.
        prompt_text: Complete prompt from build_portrait_prompt.
        model_alias: Model alias for the LLM request.
        temperature: LLM temperature (1.0 for generation, 0.8 for correction).

    Returns:
        Parsed AlphaPortrait, or None on failure.
    """
    try:
        response = llm.complete(
            LLMRequest(
                prompt=PromptRenderResult(
                    prompt_render_id="mcts-portrait",
                    prompt_id="portrait",
                    template_version="1.0",
                    template_checksum="mcts",
                    input_schema="portrait/1.0",
                    output_schema="portrait/1.0",
                    messages=(),
                    output_json_schema={
                        "type": "object",
                        "properties": {
                            "portrait_name": {"type": "string"},
                            "portrait_logic": {"type": "string"},
                            "portrait_pseudocode": {"type": "string"},
                            "expression": {"type": "string"},
                        },
                        "required": [
                            "portrait_name",
                            "portrait_logic",
                            "portrait_pseudocode",
                            "expression",
                        ],
                    },
                    estimated_input_tokens=len(prompt_text) // 4,
                    token_estimation_method="char_div_4",
                    truncation_report="none",
                    schema_version="prompt-render-result/1.0",
                ),
                model_alias=model_alias,
                max_output_tokens=800,
                response_format="json_object",
                temperature=temperature,
            )
        )
        return parse_portrait_response(response.content)
    except Exception:
        return None


def parse_portrait_response(content: str) -> AlphaPortrait | None:
    """Parse LLM JSON response into an AlphaPortrait.

    Handles common formatting issues: markdown code fences, trailing commas,
    extra whitespace.
    """
    text = content.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        # Try extracting JSON from mixed text
        import re

        match = re.search(r'\{[^{}]*"portrait_name"[^{}]*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            return None

    name = data.get("portrait_name", "")
    logic = data.get("portrait_logic", "")
    pseudo = data.get("portrait_pseudocode", "")
    expression = data.get("expression", "")

    if not expression or not name:
        return None

    # Clean expression: remove backticks, extra spaces
    expression = expression.strip().strip("`").strip()

    return AlphaPortrait(
        name=name.strip(),
        logic=logic.strip(),
        pseudo_code=pseudo.strip(),
        expression=expression,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def build_fzoo_portraits(
    fzoo_entries: Sequence[Any],
    max_examples: int = 3,
) -> list[AlphaPortrait]:
    """Convert FzooEntry list to AlphaPortrait list for few-shot prompts.

    Args:
        fzoo_entries: FzooEntry objects with .expression and .portrait fields.
        max_examples: Maximum number of examples to include.

    Returns:
        List of AlphaPortrait objects for use in build_portrait_prompt().
    """
    portraits: list[AlphaPortrait] = []
    for entry in fzoo_entries[:max_examples]:
        # Extract name from portrait text (first line or first sentence)
        portrait_text = getattr(entry, "portrait", "") or ""
        name = getattr(entry, "factor_id", "Alpha")[:20]
        logic = portrait_text[:200] if portrait_text else "Effective alpha factor"
        portraits.append(
            AlphaPortrait(
                name=name,
                logic=logic,
                pseudo_code="",
                expression=getattr(entry, "expression", ""),
            )
        )
    return portraits
