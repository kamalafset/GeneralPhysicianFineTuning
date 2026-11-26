import hashlib
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from src.embeddings import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        collection_name: str = "lab_reports",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        # cosine distance suits sentence-transformer embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"[INFO] Chroma collection '{collection_name}' at {persist_dir} "
              f"({self.collection.count()} existing vectors)")

    @staticmethod
    def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma only accepts str/int/float/bool metadata values."""
        return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

    @staticmethod
    def _make_id(chunk: Any, index: int) -> str:
        """Deterministic id so re-ingesting the same chunks overwrites instead of duplicating."""
        src = str(chunk.metadata.get("source", ""))
        page = str(chunk.metadata.get("page", ""))
        digest = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        return hashlib.md5(f"{src}|{page}|{index}|{digest}".encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Any], batch_size: int = 500) -> int:
        """Chunk, embed and upsert documents into Chroma. Returns number of chunks stored."""
        chunks = self.embedding_pipeline.chunk_documents(documents)
        if not chunks:
            print("[WARN] No chunks to store.")
            return 0
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        self.add_embeddings(embeddings, chunks, batch_size=batch_size)
        return len(chunks)

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Any], batch_size: int = 500) -> None:
        ids = [self._make_id(c, i) for i, c in enumerate(chunks)]
        texts = [c.page_content for c in chunks]
        metas = [self._clean_metadata(c.metadata) for c in chunks]
        embs = np.asarray(embeddings, dtype="float32").tolist()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embs[start:end],
                documents=texts[start:end],
                metadatas=metas[start:end],  # type: ignore[arg-type]
            )
        print(f"[INFO] Stored {len(ids)} vectors in Chroma. Collection size: {self.collection.count()}")

    def query(self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Semantic search. `where` filters on metadata, e.g. {"patient_id": "P0001"}."""
        q_emb = self.embedding_pipeline.model.encode([query_text]).astype("float32").tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=top_k, where=where)
        ids = res["ids"][0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    from src.data_cleaner import clean_documents

    docs = clean_documents(load_all_documents("data/reports"))
    store = ChromaVectorStore()
    store.add_documents(docs)
    for r in store.query("patients with high HbA1c", top_k=3):
        print(round(r["distance"], 3), r["metadata"].get("patient_id"), r["text"][:120].replace("\n", " | "))
