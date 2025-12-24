import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: str
    text: str
    score: float

class RAGIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts: List[Tuple[str, str, str]] = []  # (doc_id, chunk_id, text)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]: 
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk.strip())
            i += max(1, chunk_size - overlap)
        return [c for c in chunks if c]

    def build_from_folder(self, docs_path: str = "./docs"):
        all_chunks = []
        for fn in os.listdir(docs_path):
            if not fn.lower().endswith((".txt", ".md")):
                continue
            doc_id = fn
            with open(os.path.join(docs_path, fn), "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self._chunk_text(content)
            for j, ch in enumerate(chunks):
                chunk_id = f"{os.path.splitext(fn)[0]}_{j}"
                self.texts.append((doc_id, chunk_id, ch))
                all_chunks.append(ch)

        if not all_chunks:
            raise RuntimeError(f"No .txt/.md documents found in {docs_path}")

        embs = self.embedder.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
        embs = embs.astype(np.float32)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = 4) -> List[DocChunk]:
        if self.index is None:
            raise RuntimeError("RAG index not built. Call build_from_folder first.")

        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, top_k)

        out: List[DocChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            doc_id, chunk_id, text = self.texts[int(idx)]
            out.append(DocChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, score=float(score)))
        return out
