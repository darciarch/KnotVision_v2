"""Prompt template, class additions and metric-driven feedback rules for edit mode.

Changing this file changes every agent's hash (job.agent_hash) and therefore invalidates the QC
calibration of every edit configuration.
"""
from __future__ import annotations

BASE_PROMPT = (
    "Redraw this carpet photo as its flat production design: a top-down, perfectly rectangular "
    "drawing of the same pattern. Hard, clean edges. No pile texture, no shading, no shadows, "
    "no highlights, no fringe, no perspective. Use ONLY the colors in the second image "
    "(hex: {hex_list}). Keep every motif at exactly the same position, size and "
    "shape as in the photo. Do not add, remove, complete, or reinterpret any motif."
)

CLASS_ADDITIONS = {
    "gradient": "Where the photo shows a smooth color transition, keep it smooth; do not invent steps.",
    "distressed": "Keep the worn, speckled areas as they are; do not clean them up.",
    "geometric": "Lines must be perfectly straight; repeated motifs must be identical.",
    "minimal": "Large flat regions; no texture inside them.",
    "classic": "",
}

NEGATIVE_PROMPT = "photo, texture, fabric, shadow, blur, noise, watermark, text"

ROUND_PROMPT = "Reference photo is image 3; correct the drawing (image 1) to match it."

FEEDBACK_RULES = {
    "align_px": "Do not move, shift, rotate or rescale anything; every motif stays exactly where it is.",
    "added_pct": "Remove any element that is not in the reference photo. Do not complete faint or partial motifs.",
    "missing_pct": "Some motifs from the photo are missing; draw every motif visible in the photo, including faint ones.",
    "palette_deltaE_p95": "Use only the exact colors from the palette image; no intermediate tones, no gradients unless present in the photo.",
    "noise_1knot_pct": "Fill each region with one flat color; no speckle, grain or texture.",
    "line_jaggedness": "Straight lines must be perfectly straight and parallel to the edges.",
}


def base_prompt(hex_list: list[str], design_class: str) -> str:
    p = BASE_PROMPT.format(hex_list=", ".join(hex_list))
    add = CLASS_ADDITIONS.get(design_class, "")
    return p + (" " + add if add else "")


def feedback(violations: list[str], design_class: str) -> list[str]:
    """One sentence per violated feature. line_jaggedness only for geometric / classic (border)."""
    out = []
    for v in violations:
        if v == "line_jaggedness" and design_class not in ("geometric", "classic"):
            continue
        if v in FEEDBACK_RULES:
            out.append(FEEDBACK_RULES[v])
    return out


def round_prompt(prev_prompt: str, feedback_lines: list[str]) -> str:
    return prev_prompt + " " + ROUND_PROMPT + (" " + " ".join(feedback_lines) if feedback_lines else "")
