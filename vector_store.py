from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import pickle

import faiss
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ResearchVectorStore:
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_size = 384

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []

        for text in texts:
            text = text or ""
            vec = np.zeros(self.embedding_size, dtype=np.float32)

            for i, ch in enumerate(text[: self.embedding_size]):
                vec[i] = (ord(ch) % 97) / 97.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def create_vector_store(self, documents: List[Document]) -> None:
        self.documents = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]

        embeddings = self._embed(self.documents)

        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(embeddings)

        self.save()

    def save(self) -> None:
        if self.index is None:
            raise ValueError("FAISS index is not initialized.")

        faiss.write_index(self.index, str(self.store_path / "faiss_index.bin"))

        with open(self.store_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        with open(self.store_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def Load(cls, store_path: str) -> "ResearchVectorStore":
        instance = cls(store_path)

        index_path = Path(store_path) / "faiss_index.bin"
        metadata_path = Path(store_path) / "metadata.pkl"
        documents_path = Path(store_path) / "documents.pkl"

        if not all(p.exists() for p in [index_path, metadata_path, documents_path]):
            raise FileNotFoundError("Vector store files are missing.")

        instance.index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            instance.metadata = pickle.load(f)

        with open(documents_path, "rb") as f:
            instance.documents = pickle.load(f)

        instance.embedding_size = instance.index.d
        return instance

    def query_similar(self, query: str, k: int = 5, use_recency: bool = False) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty query received.")
            return []

        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector store is empty.")
            return []

        query_vec = self._embed([query.strip()])
        scores, indices = self.index.search(query_vec, len(self.documents))

        current_year = datetime.now().year
        results = []

        for idx, sim in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            meta = self.metadata[idx]
            similarity = float(sim)

            if use_recency:
                year = meta.get("year", None)
                if isinstance(year, int) and year > 0:
                    recency_score = (year - (current_year - 100)) / 100
                    combined_score = 0.7 * similarity + 0.3 * recency_score
                else:
                    combined_score = similarity
            else:
                combined_score = similarity

            results.append({
                "content": self.documents[idx],
                "metadata": meta,
                "similarity": similarity,
                "combined_score": float(combined_score)
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        results = results[:k]

        if use_recency:
            results.sort(key=lambda x: x["metadata"].get("year", 0), reverse=True)

        return results