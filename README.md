# 🎓 VivaMate – AI-Powered Adaptive Viva Examination System

## 📌 Overview

VivaMate is an **AI-Powered Mock Viva Practice & Readiness Simulator** built with **LangChain**, **Streamlit**, **Groq Cloud API**, and a **Per-Session Lab Manual RAG Pipeline**. It helps students practice subject and project vivas, grounds questions in their uploaded lab manual PDFs without burning LLM tokens during ingestion, dynamically adapts difficulty, and produces downloadable assessment reports.

---

## 🏗️ Technical Architecture (Review Pitch Guide)

1. **LangChain Orchestration & Structured Output (`llm_engine.py`, `prompts.py`)**:
   - Uses `ChatPromptTemplate` + `PydanticOutputParser` (`VivaEvaluation` schema) for deterministic JSON evaluation (`score`, `strengths`, `weaknesses`, `ideal_answer`, `follow_up_question`).
   - Primary runtime: **Groq Cloud API** (`llama-3.1-8b-instant`) with zero local model download, automatic fallback to **Ollama** (if running locally), and an **Offline Demo Mode** safety net.
2. **Per-Session Lab Manual RAG Pipeline (`rag_pipeline.py`)**:
   - Ingests uploaded `.pdf` lab manuals via `PyPDF` + `RecursiveCharacterTextSplitter` (`chunk_size=700`, `chunk_overlap=100`).
   - Indexes in-memory using **0 LLM tokens** (supports FAISS + MiniLM embeddings or zero-download TF-IDF cosine similarity) and retrieves only the top-3 relevant chunks per question to minimize token usage.
3. **2-Tier Memory Layer (`memory_manager.py`)**:
   - **Tier 1 (Short-Term Episodic Window, `k=2`)**: Keeps only the last 2 Q&A turns for natural follow-up continuity.
   - **Tier 2 (Long-Term Semantic State)**: Maintains a compact `O(1)` summary of covered topics and weak areas so prompt token size never grows linearly.
4. **Adaptive Difficulty Controller (`adaptive_engine.py`)**:
   - Automatically steps up difficulty (`>= 8/10`) or steps down (`<= 4/10`) with transparent explanations.
5. **Analytics & Export (`report_exporter.py`)**:
   - Live score progression chart, 2-Tier Memory Inspector, and downloadable `.md` Report + `.csv` Transcript.

---

## ☁️ Quick Deployment (Streamlit Community Cloud)

1. Push the `VivaMate/` folder to GitHub.
2. Deploy `app.py` on [Streamlit Community Cloud](https://share.streamlit.io/).
3. Add `GROQ_API_KEY` in Streamlit Secrets (or paste it in the sidebar during your demo).
