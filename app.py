import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from adaptive_engine import DIFFICULTY_LEVELS, compute_next_difficulty
from llm_engine import LangChainVivaEngine, VivaEvaluation
from memory_manager import TwoTierVivaMemory
from prompts import EVALUATION_PROMPT, QUESTION_PROMPT, REPORT_PROMPT
from rag_pipeline import LabManualRAG
from report_exporter import generate_csv_transcript, generate_markdown_report

MAX_QUESTIONS = 5

VIVA_STYLE_CONSTRAINT = (
    "ORAL VIVA FORMAT RULES (STRICT): Ask ONE crisp, conversational viva question "
    "in 1-2 short sentences (under 25 words). Sound like a real examiner asking a "
    "quick spoken check—never ask a 5-mark or 10-mark essay question, never ask "
    "multi-part (a, b, c) questions, and do not ask for long step-by-step derivations."
)

st.set_page_config(page_title="VivaMate", page_icon="🎓", layout="centered")
st.title("🎓 VivaMate")
st.subheader("AI-Powered Adaptive Viva Examination System")


def _levels_list():
    if DIFFICULTY_LEVELS:
        return list(DIFFICULTY_LEVELS)
    return ["Beginner 🟢", "Intermediate 🟡", "Advanced 🔴"]


def _resolve_level_index(diff_str: str, levels: list) -> int:
    if diff_str in levels:
        return levels.index(diff_str)
    low = (diff_str or "").lower()
    for i, lvl in enumerate(levels):
        lvl_low = lvl.lower()
        if "begin" in low and "begin" in lvl_low:
            return i
        if "inter" in low and "inter" in lvl_low:
            return i
        if "adv" in low and "adv" in lvl_low:
            return i
    return 0


def _get_next_difficulty(current_diff: str, scores: list, auto_adapt: bool):
    levels = _levels_list()
    cur_idx = _resolve_level_index(current_diff, levels)
    normalized_current = levels[cur_idx]

    if not auto_adapt:
        return normalized_current, f"Auto-adapt OFF — staying at {normalized_current}."
    if not scores:
        return normalized_current, f"Starting at {normalized_current}."

    last_score = scores[-1]
    expected_idx = cur_idx
    if last_score > 7 and cur_idx < len(levels) - 1:
        expected_idx = cur_idx + 1
    elif last_score < 5 and cur_idx > 0:
        expected_idx = cur_idx - 1

    try:
        next_d, reason = compute_next_difficulty(normalized_current, scores, auto_adapt=True)
        resolved_idx = _resolve_level_index(next_d, levels)
        if expected_idx != cur_idx and resolved_idx == cur_idx:
            resolved_idx = expected_idx
            direction = "UP" if resolved_idx > cur_idx else "DOWN"
            reason = (
                f"Auto-adapted {direction} to {levels[resolved_idx]} "
                f"(last score: {last_score}/10)."
            )
        final_diff = levels[resolved_idx]
        return final_diff, reason or f"Difficulty set to {final_diff} (last score: {last_score}/10)."
    except Exception:
        final_diff = levels[expected_idx]
        if expected_idx > cur_idx:
            return final_diff, f"Auto-adapted UP to {final_diff} (last score: {last_score}/10)."
        if expected_idx < cur_idx:
            return final_diff, f"Auto-adapted DOWN to {final_diff} (last score: {last_score}/10)."
        return final_diff, f"Maintained {final_diff} (last score: {last_score}/10)."


DEFAULT_DIFF = _levels_list()[0]

# -----------------------------
# Session State Initialization
# -----------------------------
DEFAULTS = {
    "student_name": "",
    "roll_number": "",
    "department": "IT",
    "year": "4th Year",
    "section": "",
    "question_number": 0,
    "scores": [],
    "skipped_questions": 0,
    "answered_questions": 0,
    "answer_evaluated": False,
    "last_evaluation": None,
    "transcript": [],
    "active_difficulty": DEFAULT_DIFF,
    "difficulty_selector": DEFAULT_DIFF,
    "pending_difficulty_selector": None,
    "adapt_reason": "",
    "follow_up_hint": None,
    "final_report": "",
    "last_rag_chunks": [],
    "viva_completed": False,
    "pending_answer_text": None,
    "answer_text": "",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = list(val) if isinstance(val, list) else val

if "memory" not in st.session_state:
    st.session_state.memory = TwoTierVivaMemory(window_size=2)

if "rag" not in st.session_state or not hasattr(st.session_state.rag, "ingest_pdf"):
    st.session_state.rag = LabManualRAG()

# -----------------------------
# Sidebar: AI Runtime & Lab Manual RAG
# -----------------------------
with st.sidebar:
    st.header("⚙️ AI & RAG Configuration")
    backend = st.selectbox(
        "AI Backend",
        [
            "Groq Cloud API",
            "Offline Demo Mode",
        ],
    )

    api_key = ""
    model_name = "openai/gpt-oss-20b"

    if backend == "Groq Cloud API":
        model_choice = st.selectbox(
            "Groq Model",
            ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "Custom Model ID"],
        )
        model_name = (
            st.text_input("Enter Groq Model ID", value="openai/gpt-oss-20b")
            if model_choice == "Custom Model ID"
            else model_choice
        )
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            st.caption("✅ GROQ_API_KEY loaded from .env")
        else:
            st.warning("⚠️ GROQ_API_KEY not found in .env")
    else:
        st.info("🧪 Offline Demo Mode: Questions and answers are pre-filled automatically.")

    st.divider()
    auto_adapt = st.toggle(
        "🧠 Auto-Adapt Difficulty",
        value=True,
        help="Steps difficulty up (> 7/10) or down (< 5/10) automatically.",
    )
    if auto_adapt:
        st.caption("📈 > 7/10 steps UP • 📉 < 5/10 steps DOWN")

    st.divider()
    st.subheader("📚 Lab Manual RAG (Optional)")
    uploaded_pdf = st.file_uploader(
        "Upload Lab Manual (.pdf)",
        type=["pdf"],
        help="Indexed in-memory per session using 0 LLM tokens.",
    )
    rag_topic_focus = st.text_input(
        "Experiment / Topic Focus",
        placeholder="e.g., Experiment 4 or leave blank for all",
    )

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.read()
        same_doc_fn = getattr(st.session_state.rag, "is_same_document", None)
        already_indexed = (
            same_doc_fn(pdf_bytes, uploaded_pdf.name)
            if callable(same_doc_fn)
            else (
                bool(getattr(st.session_state.rag, "chunks", []))
                and getattr(st.session_state.rag, "manual_name", "") == uploaded_pdf.name
            )
        )
        if not already_indexed:
            with st.spinner("Indexing lab manual in memory..."):
                try:
                    num_chunks, mode = st.session_state.rag.ingest_pdf(
                        pdf_bytes, uploaded_pdf.name
                    )
                    st.success(f"Indexed {num_chunks} chunks ({mode})")
                except Exception as exc:
                    st.error(f"PDF ingestion failed: {exc}")
        else:
            st.caption(
                f"✅ {st.session_state.rag.manual_name} "
                f"({len(st.session_state.rag.chunks)} chunks • "
                f"{len(st.session_state.rag.used_chunk_indices)} visited)"
            )

    st.divider()
    if st.button("🔄 Start New Viva", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = list(val) if isinstance(val, list) else val
        st.session_state.active_difficulty = DEFAULT_DIFF
        st.session_state.difficulty_selector = DEFAULT_DIFF
        st.session_state.pending_difficulty_selector = None
        st.session_state.pending_answer_text = ""
        st.session_state.answer_text = ""
        st.session_state.memory = TwoTierVivaMemory(window_size=2)
        st.session_state.rag.reset_history()
        st.session_state.pop("question", None)
        st.rerun()

llm = LangChainVivaEngine(
    backend=backend,
    model_name=model_name,
    api_key=api_key,
)

# -----------------------------
# Clean Setup Section (Collapses Once Viva Starts)
# -----------------------------
viva_started = "question" in st.session_state
levels_available = _levels_list()


def _on_manual_difficulty_change():
    st.session_state.active_difficulty = st.session_state.difficulty_selector
    st.session_state.pending_difficulty_selector = None
    st.session_state.adapt_reason = (
        f"Manually set to {st.session_state.active_difficulty}."
    )


if st.session_state.pending_difficulty_selector in levels_available:
    st.session_state.difficulty_selector = st.session_state.pending_difficulty_selector
    st.session_state.pending_difficulty_selector = None
elif st.session_state.difficulty_selector not in levels_available:
    st.session_state.difficulty_selector = DEFAULT_DIFF

with st.expander("👨🎓 Student & Viva Setup", expanded=not viva_started):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.text_input("Student Name", key="student_name")
        st.selectbox(
            "Department",
            [
                "CSE", "IT", "CSE-AIML", "CSE-DS", "CSE-CSC",
                "ECE", "EEE", "Mechanical", "Civil", "Other",
            ],
            key="department",
        )
        st.text_input("Section", key="section")
    with col_s2:
        st.text_input("Roll Number", key="roll_number")
        st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"], key="year")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        subject = st.selectbox(
            "Choose Subject",
            [
                "Operating Systems", "DBMS", "Software Engineering",
                "Computer Networks", "Java", "Python", "AI/ML",
                "Cloud Computing", "Cyber Security", "Project Viva", "HR Interview",
            ],
        )
    with col_v2:
        st.selectbox(
            "Difficulty",
            levels_available,
            key="difficulty_selector",
            on_change=_on_manual_difficulty_change,
        )
        if not auto_adapt:
            st.session_state.active_difficulty = st.session_state.difficulty_selector

    project_title = ""
    project_domain = ""
    tech_stack = ""
    question_type = "Mixed"
    if subject == "Project Viva":
        project_title = st.text_input("Project Title")
        project_domain = st.text_input("Project Domain")
        tech_stack = st.text_input("Technology Stack")
        question_type = st.selectbox(
            "Question Type",
            ["Conceptual", "Implementation", "Code-Level", "Technology-Specific", "Mixed"],
        )


def _trim_to_viva_question(raw_text: str) -> str:
    cleaned = " ".join((raw_text or "").strip().split())
    if not cleaned:
        return "What is the core idea behind this topic in practice?"
    sentences = []
    current = []
    for token in cleaned.split(" "):
        current.append(token)
        if token.endswith(("?", ".", "!")) and len(token) > 1:
            sentences.append(" ".join(current))
            current = []
            if len(sentences) == 2:
                break
    if current and len(sentences) < 2:
        sentences.append(" ".join(current))
    short_q = " ".join(sentences).strip()
    return short_q[:200]


# ---------------------------
# Question Generator Function
# ---------------------------
def generate_question(use_followup: bool = False):
    next_diff, reason = _get_next_difficulty(
        st.session_state.active_difficulty,
        st.session_state.scores,
        auto_adapt=auto_adapt,
    )
    st.session_state.active_difficulty = next_diff
    st.session_state.pending_difficulty_selector = next_diff
    st.session_state.adapt_reason = reason

    project_ctx = ""
    if subject == "Project Viva":
        project_ctx = (
            f"Project Title: {project_title or 'N/A'}\n"
            f"Project Domain: {project_domain or 'N/A'}\n"
            f"Technology Stack: {tech_stack or 'N/A'}\n"
            f"Question Type: {question_type}"
        )

    rag_ctx = ""
    if st.session_state.rag.chunks:
        if use_followup and st.session_state.last_rag_chunks:
            chunks = st.session_state.last_rag_chunks
        else:
            weak_hint = (
                st.session_state.memory.weak_topics[-1]
                if st.session_state.memory.weak_topics
                else ""
            )
            query = (
                rag_topic_focus.strip()
                or f"{subject} {next_diff} {weak_hint} question {st.session_state.question_number}".strip()
            )
            chunks = st.session_state.rag.retrieve(
                query, k=3, question_number=st.session_state.question_number
            )
            st.session_state.last_rag_chunks = chunks
        rag_ctx = "Lab Manual Context (Retrieved Excerpts):\n" + "\n---\n".join(chunks)
    else:
        st.session_state.last_rag_chunks = []

    difficulty_directive = (
        f"Target Difficulty Level: {next_diff}. "
        f"Calibrate depth and precision strictly to {next_diff} (higher difficulty = deeper edge cases, "
        f"trade-offs, or internal mechanics; lower difficulty = direct core concept)."
    )
    instructions = [VIVA_STYLE_CONSTRAINT, difficulty_directive]
    if use_followup and st.session_state.follow_up_hint:
        instructions.append(
            f"Follow-up Focus at {next_diff} level: Ask a quick oral follow-up under 20 words about: "
            f"{st.session_state.follow_up_hint}"
        )
    follow_up_inst = "\n".join(instructions)

    try:
        raw_question_text = llm.generate_text(
            QUESTION_PROMPT,
            {
                "subject": subject,
                "difficulty": next_diff,
                "project_context": project_ctx,
                "rag_context": rag_ctx,
                "memory_state": st.session_state.memory.format_for_prompt(),
                "follow_up_instruction": follow_up_inst,
                "question_number": st.session_state.question_number,
            },
            task="question",
        )
        question_text = _trim_to_viva_question(raw_question_text)
    except Exception as exc:
        st.error(f"Question generation failed: {exc}")
        return

    st.session_state.question = question_text
    st.session_state.answer_evaluated = False
    st.session_state.last_evaluation = None
    st.session_state.follow_up_hint = None

    if backend == "Offline Demo Mode":
        st.session_state.pending_answer_text = llm.get_demo_answer(
            st.session_state.question_number
        )
    else:
        st.session_state.pending_answer_text = ""


if not viva_started:
    if st.button("▶️ Start 5-Question Viva", type="primary", use_container_width=True):
        st.session_state.question_number = 1
        generate_question()
        st.rerun()

# ---------------------------
# Active Viva Flow (Questions 1 to 5)
# ---------------------------
if viva_started and not st.session_state.viva_completed:
    if backend == "Offline Demo Mode":
        st.caption(
            "🧪 Offline Demo Mode — Prebuilt answer loaded below. Click Evaluate Answer to demo grading."
        )

    completed_count = len(st.session_state.scores)
    st.progress(
        min(completed_count / MAX_QUESTIONS, 1.0),
        text=f"Question {st.session_state.question_number} of {MAX_QUESTIONS} • {completed_count}/{MAX_QUESTIONS} evaluated",
    )

    st.info(
        f"**Question {st.session_state.question_number}/{MAX_QUESTIONS}** • "
        f"Difficulty: **{st.session_state.active_difficulty}**"
    )
    if st.session_state.adapt_reason:
        st.caption(f"🧠 {st.session_state.adapt_reason}")

    st.markdown(f"#### {st.session_state.question}")

    if st.session_state.last_rag_chunks:
        with st.expander("📚 Active Lab Manual Chunks (RAG)"):
            for i, chunk in enumerate(st.session_state.last_rag_chunks, start=1):
                st.markdown(f"**Chunk {i}:** {chunk[:320]}...")

    # Apply pre-filled demo answer (or clear for live mode) before rendering text_area
    if st.session_state.pending_answer_text is not None:
        st.session_state["answer_text"] = st.session_state.pending_answer_text
        st.session_state.pending_answer_text = None
    elif (
        backend == "Offline Demo Mode"
        and not st.session_state.answer_evaluated
        and not st.session_state["answer_text"].strip()
    ):
        # Also auto-fill immediately if user switches to Offline Demo Mode mid-question
        st.session_state["answer_text"] = llm.get_demo_answer(
            st.session_state.question_number
        )

    answer = st.text_area("Your Answer", key="answer_text", height=130)

    if not st.session_state.answer_evaluated:
        col_a1, col_a2 = st.columns([3, 1])
        with col_a1:
            if st.button("✅ Evaluate Answer", type="primary", use_container_width=True):
                if not answer.strip():
                    st.warning("Please enter an answer or click Skip.")
                else:
                    try:
                        eval_obj: VivaEvaluation = llm.evaluate_answer(
                            EVALUATION_PROMPT,
                            {
                                "difficulty": st.session_state.active_difficulty,
                                "question": st.session_state.question,
                                "student_answer": answer,
                                "question_number": st.session_state.question_number,
                            },
                        )
                        weakness_note = "; ".join(eval_obj.weaknesses)
                        st.session_state.memory.record_turn(
                            st.session_state.question,
                            answer,
                            eval_obj.score,
                            st.session_state.active_difficulty,
                            weakness_summary=weakness_note,
                        )
                        st.session_state.scores.append(eval_obj.score)
                        st.session_state.answered_questions += 1
                        st.session_state.follow_up_hint = eval_obj.follow_up_question
                        st.session_state.transcript.append(
                            {
                                "question_number": st.session_state.question_number,
                                "question": st.session_state.question,
                                "answer": answer,
                                "score": eval_obj.score,
                                "difficulty": st.session_state.active_difficulty,
                                "strengths": eval_obj.strengths,
                                "weaknesses": eval_obj.weaknesses,
                                "ideal_answer": eval_obj.ideal_answer,
                                "status": "Answered",
                            }
                        )
                        st.session_state.last_evaluation = eval_obj
                        st.session_state.answer_evaluated = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
        with col_a2:
            if st.button("⏭️ Skip", use_container_width=True):
                st.session_state.skipped_questions += 1
                st.session_state.scores.append(0)
                st.session_state.memory.record_turn(
                    st.session_state.question,
                    "[Skipped]",
                    0,
                    st.session_state.active_difficulty,
                    weakness_summary="Skipped question",
                )
                st.session_state.transcript.append(
                    {
                        "question_number": st.session_state.question_number,
                        "question": st.session_state.question,
                        "answer": "[Skipped]",
                        "score": 0,
                        "difficulty": st.session_state.active_difficulty,
                        "strengths": [],
                        "weaknesses": ["Question was skipped."],
                        "ideal_answer": "Review this topic in your lab manual.",
                        "status": "Skipped",
                    }
                )
                if len(st.session_state.scores) >= MAX_QUESTIONS:
                    st.session_state.viva_completed = True
                else:
                    st.session_state.question_number += 1
                    generate_question()
                st.rerun()

    if st.session_state.last_evaluation is not None:
        ev: VivaEvaluation = st.session_state.last_evaluation
        st.success(f"Score: {ev.score}/10")
        st.markdown("**Strengths:**\n" + "\n".join(f"- {s}" for s in ev.strengths))
        st.markdown("**Mistakes / Weaknesses:**\n" + "\n".join(f"- {w}" for w in ev.weaknesses))
        st.markdown(f"**How to Improve (Ideal Answer):**\n{ev.ideal_answer}")

    if st.session_state.answer_evaluated:
        if len(st.session_state.scores) >= MAX_QUESTIONS:
            if st.button("🏁 Finish Viva & View Results", type="primary", use_container_width=True):
                st.session_state.viva_completed = True
                st.rerun()
        else:
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                if st.button("➡️ Next Question", type="primary", use_container_width=True):
                    st.session_state.question_number += 1
                    generate_question(use_followup=False)
                    st.rerun()
            with col_n2:
                if st.session_state.follow_up_hint and st.button(
                    "🔍 Ask Follow-up Question", use_container_width=True
                ):
                    st.session_state.question_number += 1
                    generate_question(use_followup=True)
                    st.rerun()

# ---------------------------
# End-of-Viva Performance Dashboard (Shown Only After 5 Questions)
# ---------------------------
if st.session_state.viva_completed:
    st.divider()
    st.header("📊 Final Performance Dashboard")

    avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
    total_questions = len(st.session_state.transcript)
    completion_pct = (
        st.session_state.answered_questions / total_questions * 100
        if total_questions > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{avg_score:.2f}/10")
        st.metric("Answered", st.session_state.answered_questions)
    with col2:
        st.metric("Skipped", st.session_state.skipped_questions)
        st.metric("Total Questions", total_questions)
    with col3:
        st.metric("Completion Rate", f"{completion_pct:.0f}%")
        st.metric("Final Difficulty", st.session_state.active_difficulty)

    chart_df = pd.DataFrame(
        {
            "Question": [f"Q{t['question_number']}" for t in st.session_state.transcript],
            "Score": [t["score"] for t in st.session_state.transcript],
        }
    ).set_index("Question")
    st.line_chart(chart_df)

    st.subheader("📝 Question-by-Question Review: Mistakes & How to Improve")
    for t in st.session_state.transcript:
        with st.expander(
            f"Q{t['question_number']} ({t['difficulty']}) — Score: {t['score']}/10 [{t['status']}]",
            expanded=(t["score"] < 7),
        ):
            st.markdown(f"**Question:** {t['question']}")
            st.markdown(f"**Your Answer:** {t['answer']}")
            if t["strengths"]:
                st.markdown("**What You Did Well:**\n" + "\n".join(f"- {s}" for s in t["strengths"]))
            if t["weaknesses"]:
                st.markdown("**Mistakes / Gaps:**\n" + "\n".join(f"- {w}" for w in t["weaknesses"]))
            st.markdown(f"**How You Could Improve (Ideal Answer):**\n{t['ideal_answer']}")

    with st.expander("🧠 Inspect 2-Tier Memory State"):
        st.json(st.session_state.memory.to_dict())

    st.divider()
    st.subheader(f"📄 AI Performance Report — {st.session_state.student_name or 'Student'}")

    if st.button("Generate Performance Report", type="primary"):
        breakdown = "\n".join(
            f"Q{t['question_number']} [{t['difficulty']}] ({t['score']}/10): {t['question']}"
            for t in st.session_state.transcript
        )
        try:
            st.session_state.final_report = llm.generate_text(
                REPORT_PROMPT,
                {
                    "student_name": st.session_state.student_name or "N/A",
                    "roll_number": st.session_state.roll_number or "N/A",
                    "department": st.session_state.department,
                    "year": st.session_state.year,
                    "section": st.session_state.section or "N/A",
                    "subject": subject,
                    "scores": st.session_state.scores,
                    "answered_questions": st.session_state.answered_questions,
                    "skipped_questions": st.session_state.skipped_questions,
                    "session_breakdown": breakdown,
                },
                task="report",
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    if st.session_state.final_report:
        st.markdown(st.session_state.final_report)

    student_info = {
        "student_name": st.session_state.student_name,
        "roll_number": st.session_state.roll_number,
        "department": st.session_state.department,
        "year": st.session_state.year,
        "section": st.session_state.section,
    }

    md_report = generate_markdown_report(
        student_info=student_info,
        subject=subject,
        scores=st.session_state.scores,
        answered=st.session_state.answered_questions,
        skipped=st.session_state.skipped_questions,
        transcript=st.session_state.transcript,
        ai_report=st.session_state.final_report,
    )
    csv_data = generate_csv_transcript(
        student_info=student_info,
        subject=subject,
        transcript=st.session_state.transcript,
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "⬇️ Download Report (.md)",
            data=md_report,
            file_name=f"VivaMate_Report_{st.session_state.roll_number or 'student'}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_d2:
        st.download_button(
            "⬇️ Download Transcript (.csv)",
            data=csv_data,
            file_name=f"VivaMate_Transcript_{st.session_state.roll_number or 'student'}.csv",
            mime="text/csv",
            use_container_width=True,
        )