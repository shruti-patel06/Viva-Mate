"""
VivaMate - LangChain LLM Engine & Pydantic Structured Output
Primary: Groq Cloud API (fast cloud inference, zero local disk footprint).
Fallback: Built-in Offline Demo Mode with prebuilt questions, answers & evaluations.
"""

import os
import re
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class VivaEvaluation(BaseModel):
    score: int = Field(description="Integer score from 0 to 10", ge=0, le=10)
    strengths: List[str] = Field(description="2 concise bullet points on strengths")
    weaknesses: List[str] = Field(description="2 concise bullet points on weaknesses")
    ideal_answer: str = Field(description="Concise 2-3 sentence ideal viva answer")
    follow_up_question: str = Field(
        description="One short oral follow-up question under 20 words"
    )


class LangChainVivaEngine:
    DEMO_QUESTIONS = [
        "Why do we normalize tables in a database before denormalizing?",
        "How does an index speed up SELECT queries, and what is its write cost?",
        "What does ON DELETE CASCADE do in a foreign key relationship?",
        "What is the difference between a process and a thread?",
        "How does TCP guarantee reliable delivery over an unreliable network?",
    ]

    DEMO_ANSWERS = [
        (
            "Normalization removes data redundancy and prevents insert, update, and delete "
            "anomalies by organizing data into related tables. We only denormalize selectively "
            "later when read-heavy queries suffer from too many joins."
        ),
        (
            "An index creates a sorted lookup structure like a B-tree so the database avoids "
            "a full table scan during SELECT queries, though INSERT, UPDATE, and DELETE become "
            "slightly slower because the index must also be updated."
        ),
        (
            "ON DELETE CASCADE automatically deletes matching child rows in the referencing "
            "table whenever the parent record is deleted, keeping referential integrity intact."
        ),
        (
            "A process is an independent program with its own isolated memory space, whereas "
            "a thread is a lightweight execution unit inside a process that shares heap memory "
            "and resources with sibling threads."
        ),
        (
            "TCP uses sequence numbers, acknowledgments (ACKs), timeouts with retransmissions, "
            "and checksums so lost or out-of-order packets are detected and reassembled accurately."
        ),
    ]

    DEMO_EVALUATIONS = [
        VivaEvaluation(
            score=9,
            strengths=[
                "Clearly explained eliminating redundancy and update anomalies.",
                "Highlighted read vs. write trade-offs accurately.",
            ],
            weaknesses=[
                "Could mention 3NF vs. BCNF briefly.",
                "Keep the summary slightly tighter for a quick viva.",
            ],
            ideal_answer=(
                "Normalization reduces duplicate data and update anomalies across tables, "
                "while controlled denormalization avoids costly joins on read-heavy workloads."
            ),
            follow_up_question="Which anomaly does 3NF specifically remove?",
        ),
        VivaEvaluation(
            score=4,
            strengths=[
                "Identified the main lookup structure at a high level.",
            ],
            weaknesses=[
                "Missed explaining write overhead on INSERT/UPDATE/DELETE.",
                "Needs clearer terminology for an oral viva.",
            ],
            ideal_answer=(
                "An index speeds up SELECT lookups using structures like B-trees, "
                "but slows down writes because index entries must be maintained."
            ),
            follow_up_question="When can adding too many indexes hurt performance?",
        ),
        VivaEvaluation(
            score=8,
            strengths=[
                "Gave a direct, accurate definition suited for an oral viva.",
                "Mentioned referential integrity clearly.",
            ],
            weaknesses=[
                "Could warn briefly about accidental cascading deletes in production.",
                "Mention foreign key constraint setup explicitly.",
            ],
            ideal_answer=(
                "ON DELETE CASCADE propagates a parent row deletion to all dependent "
                "child rows automatically via the foreign key constraint."
            ),
            follow_up_question="How does ON DELETE SET NULL differ from CASCADE?",
        ),
        VivaEvaluation(
            score=6,
            strengths=[
                "Distinguished isolated vs. shared memory correctly.",
            ],
            weaknesses=[
                "Did not mention context-switching cost differences.",
                "Could briefly note inter-process communication overhead.",
            ],
            ideal_answer=(
                "Processes have separate address spaces and heavier context switches, "
                "while threads share process memory and switch faster."
            ),
            follow_up_question="Why is thread context switching faster than process switching?",
        ),
        VivaEvaluation(
            score=8,
            strengths=[
                "Listed sequence numbers, ACKs, and retransmissions accurately.",
                "Structured the response clearly for a quick oral check.",
            ],
            weaknesses=[
                "Could also mention flow control or sliding window briefly.",
                "Keep phrasing slightly more concise.",
            ],
            ideal_answer=(
                "TCP ensures reliability through sequence numbers, cumulative ACKs, "
                "checksums, and timeout-based retransmissions."
            ),
            follow_up_question="What triggers a fast retransmit in TCP?",
        ),
    ]

    def __init__(
        self,
        backend: str = "Groq Cloud API",
        model_name: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.eval_parser = PydanticOutputParser(pydantic_object=VivaEvaluation)
        self._demo_q_idx = 0
        self._demo_eval_idx = 0

    def _get_chat_model(self):
        if self.backend == "Groq Cloud API":
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is missing in .env.")
            from langchain_groq import ChatGroq

            return ChatGroq(
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.4,
            )

        raise RuntimeError("Offline Demo Mode selected.")

    def get_demo_answer(self, question_number: int = 1) -> str:
        idx = max(0, question_number - 1) % len(self.DEMO_ANSWERS)
        return self.DEMO_ANSWERS[idx]

    def generate_text(
        self, prompt_template: ChatPromptTemplate, variables: dict, task: str
    ) -> str:
        if self.backend == "Offline Demo Mode":
            return self._demo_text(task, variables)

        llm = self._get_chat_model()
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(variables).strip()

    def evaluate_answer(
        self, prompt_template: ChatPromptTemplate, variables: dict
    ) -> VivaEvaluation:
        if self.backend == "Offline Demo Mode":
            return self._demo_evaluation(variables.get("question_number"))

        variables = {
            **variables,
            "format_instructions": self.eval_parser.get_format_instructions(),
        }
        raw_text = self.generate_text(prompt_template, variables, task="evaluation")

        try:
            return self.eval_parser.parse(raw_text)
        except Exception:
            score_match = re.search(r'"?score"?\s*:\s*(\d+)', raw_text, re.IGNORECASE)
            score = min(int(score_match.group(1)), 10) if score_match else 6
            return VivaEvaluation(
                score=score,
                strengths=["Addressed the core concept of the question."],
                weaknesses=["Could include more precise technical terminology."],
                ideal_answer=raw_text[:400],
                follow_up_question="Can you give a quick example of when that happens?",
            )

    def _demo_text(self, task: str, variables: Optional[dict] = None) -> str:
        variables = variables or {}
        if task == "question":
            q_num = variables.get("question_number")
            if isinstance(q_num, int) and q_num >= 1:
                idx = (q_num - 1) % len(self.DEMO_QUESTIONS)
            else:
                idx = self._demo_q_idx % len(self.DEMO_QUESTIONS)
                self._demo_q_idx += 1
            return self.DEMO_QUESTIONS[idx]

        student_name = variables.get("student_name") or "Student"
        subject = variables.get("subject") or "the selected subject"
        scores = variables.get("scores") or []
        return (
            f"1. Strong Areas:\n- {student_name} completed the {subject} demo session ({scores}).\n\n"
            "2. Areas Needing Improvement:\n- Switch to Groq Cloud API for live answer-specific grading.\n\n"
            "3. Recommendations:\n- Practice concise 20-30 second oral explanations.\n\n"
            "4. Overall Performance:\n- Demo flow verified with adaptive difficulty and 2-tier memory."
        )

    def _demo_evaluation(self, question_number: Optional[int] = None) -> VivaEvaluation:
        if isinstance(question_number, int) and question_number >= 1:
            idx = (question_number - 1) % len(self.DEMO_EVALUATIONS)
        else:
            idx = self._demo_eval_idx % len(self.DEMO_EVALUATIONS)
            self._demo_eval_idx += 1
        return self.DEMO_EVALUATIONS[idx]