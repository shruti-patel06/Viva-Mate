"""
VivaMate - Dynamic Lab Manual RAG Pipeline
Session-scoped PDF ingestion and retrieval using zero LLM tokens and zero model downloads.
Tracks previously retrieved chunks so each viva question pulls fresh,
non-repeating excerpts from different parts of the lab manual.
"""

import hashlib
import io
from typing import List, Set, Tuple

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LabManualRAG:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.manual_name: str = ""
        self.manual_hash: str = ""
        self.retriever_mode: str = "Not Loaded"
        self.used_chunk_indices: Set[int] = set()
        self._vectorizer = None
        self._tfidf_matrix = None

    def reset_history(self) -> None:
        self.used_chunk_indices.clear()

    def is_same_document(self, file_bytes: bytes, filename: str) -> bool:
        digest = hashlib.sha256(file_bytes).hexdigest()
        return (
            bool(self.chunks)
            and self.manual_name == filename
            and self.manual_hash == digest
        )

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> Tuple[int, str]:
        """Extracts, chunks, and indexes a PDF entirely in memory (0 LLM tokens, instant)."""
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[Page {idx}]\n{text.strip()}")

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            raise ValueError("Could not extract readable text from this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_text(full_text)
        self.manual_name = filename
        self.manual_hash = hashlib.sha256(file_bytes).hexdigest()
        self.used_chunk_indices.clear()

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(self.chunks)
        self.retriever_mode = "In-Memory TF-IDF Vectorizer (Instant)"

        return len(self.chunks), self.retriever_mode

    def retrieve(self, query: str, k: int = 3, question_number: int = 1) -> List[str]:
        """
        Dynamically retrieves top-k chunks while avoiding already-used chunks
        so each question covers fresh experiments/sections of the manual.
        """
        if not self.chunks:
            return []

        if len(self.used_chunk_indices) >= len(self.chunks):
            self.used_chunk_indices.clear()

        candidate_indices: List[int] = []

        if self._vectorizer is not None and self._tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
            ranked = sims.argsort()[::-1]
            for idx in ranked:
                idx = int(idx)
                if idx not in self.used_chunk_indices and sims[idx] > 0:
                    candidate_indices.append(idx)
                    if len(candidate_indices) == k:
                        break

        if len(candidate_indices) < k:
            total = len(self.chunks)
            start = ((question_number - 1) * k) % total
            for offset in range(total):
                idx = (start + offset) % total
                if (
                    idx not in self.used_chunk_indices
                    and idx not in candidate_indices
                ):
                    candidate_indices.append(idx)
                    if len(candidate_indices) == k:
                        break

        self.used_chunk_indices.update(candidate_indices)
        return [self.chunks[i] for i in candidate_indices]