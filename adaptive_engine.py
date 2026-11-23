"""
VivaMate - Adaptive Difficulty Engine
Dynamically adjusts difficulty based on rolling student performance.
"""

from typing import List, Tuple

DIFFICULTY_LEVELS = ["Beginner 🟢", "Intermediate 🟡", "Advanced 🔴"]


def compute_next_difficulty(
    current_difficulty: str,
    scores: List[int],
    auto_adapt: bool = True,
) -> Tuple[str, str]:
    """
    Returns (next_difficulty, explanation_reason).
    - Score > 7 (8-10): step up one difficulty tier.
    - Score < 5 (0-4): step down one difficulty tier.
    - Otherwise (5-7): maintain current tier.
    """
    if not auto_adapt or not scores:
        return current_difficulty, "Initial or manual difficulty active."

    idx = (
        DIFFICULTY_LEVELS.index(current_difficulty)
        if current_difficulty in DIFFICULTY_LEVELS
        else 0
    )
    last_score = scores[-1]

    if last_score > 7 and idx < len(DIFFICULTY_LEVELS) - 1:
        new_diff = DIFFICULTY_LEVELS[idx + 1]
        return (
            new_diff,
            f"Stepped up to {new_diff} (last score: {last_score}/10 > 7).",
        )

    if last_score < 5 and idx > 0:
        new_diff = DIFFICULTY_LEVELS[idx - 1]
        return (
            new_diff,
            f"Stepped down to {new_diff} to reinforce fundamentals (last score: {last_score}/10 < 5).",
        )

    return (
        current_difficulty,
        f"Maintained {current_difficulty} (last score: {last_score}/10).",
    )