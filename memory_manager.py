"""
VivaMate - 2-Tier Memory Layer
Tier 1: Short-Term Episodic Window (last k=2 Q&A turns for conversational follow-ups).
Tier 2: Long-Term Semantic State (O(1) compact summary of covered topics & weak areas).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TwoTierVivaMemory:
    window_size: int = 2
    episodic_window: List[Dict] = field(default_factory=list)
    covered_questions: List[str] = field(default_factory=list)
    strong_topics: List[str] = field(default_factory=list)
    weak_topics: List[str] = field(default_factory=list)

    def record_turn(
        self,
        question: str,
        answer: str,
        score: int,
        difficulty: str,
        weakness_summary: str = "",
    ) -> None:
        self.episodic_window.append(
            {
                "question": question,
                "answer": answer[:300],
                "score": score,
                "difficulty": difficulty,
            }
        )
        if len(self.episodic_window) > self.window_size:
            self.episodic_window = self.episodic_window[-self.window_size :]

        short_q = question.strip().split("\n")[0][:90]
        self.covered_questions.append(short_q)

        if score > 7:
            self.strong_topics.append(short_q)
        elif score < 5:
            note = weakness_summary.strip()[:80] if weakness_summary else short_q
            self.weak_topics.append(note)

    def format_for_prompt(self) -> str:
        """Returns a token-efficient representation of both memory tiers."""
        tier1_lines = []
        for idx, turn in enumerate(self.episodic_window, start=1):
            tier1_lines.append(
                f"Turn-{idx} [{turn['difficulty']}, Score {turn['score']}/10]: "
                f"Q: {turn['question']} | A: {turn['answer']}"
            )
        tier1_str = "\n".join(tier1_lines) if tier1_lines else "None yet."

        covered_str = ", ".join(self.covered_questions[-8:]) or "None yet"
        weak_str = "; ".join(self.weak_topics[-5:]) or "None identified yet"

        return (
            f"[Tier 1 - Recent Context Window (last {self.window_size} turns)]\n"
            f"{tier1_str}\n\n"
            f"[Tier 2 - Compact Semantic State]\n"
            f"- Already Covered (DO NOT repeat): {covered_str}\n"
            f"- Weak Areas to Probe/Reinforce: {weak_str}"
        )

    def to_dict(self) -> dict:
        return {
            "short_term_window_count": len(self.episodic_window),
            "covered_count": len(self.covered_questions),
            "strong_topics": self.strong_topics[-5:],
            "weak_topics": self.weak_topics[-5:],
        }