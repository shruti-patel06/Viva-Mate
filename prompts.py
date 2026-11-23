"""
VivaMate - LangChain Prompt Templates
Uses ChatPromptTemplate with RAG context injection, 2-tier memory,
and Pydantic format instructions.
"""

from langchain_core.prompts import ChatPromptTemplate

QUESTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an academic viva examiner conducting a {difficulty} level oral viva.

Subject: {subject}
{project_context}
{rag_context}

Memory State:
{memory_state}

{follow_up_instruction}

Rules:
- Ask ONE brief, spoken viva question (1-2 short sentences, ideally under 25 words).
- Sound like a live oral examiner testing quick conceptual or practical understanding—never write a 5-mark or 10-mark exam question or multi-part prompt.
- Match the {difficulty} level strictly.
- If Lab Manual Context is provided above, ground your question directly in those experiment details.
- Never repeat or rephrase any question listed in "Already Covered".
- Return ONLY the single viva question text, with no preamble."""
)

EVALUATION_PROMPT = ChatPromptTemplate.from_template(
    """You are an academic viva examiner evaluating a student's spoken-style viva answer.

Difficulty Level: {difficulty}
Question: {question}
Student Answer: {student_answer}

Evaluate fairly, concisely, and constructively for a quick viva response.
{format_instructions}"""
)

REPORT_PROMPT = ChatPromptTemplate.from_template(
    """A student completed an AI mock-viva assessment.

Student Name: {student_name}
Roll Number: {roll_number}
Department: {department} | Year: {year} | Section: {section}
Subject: {subject}
Scores: {scores}
Questions Answered: {answered_questions} | Skipped: {skipped_questions}

Session Summary:
{session_breakdown}

Generate a concise, actionable academic readiness report with:
1. Strong Areas
2. Areas Needing Improvement
3. Recommendations
4. Overall Performance"""
)