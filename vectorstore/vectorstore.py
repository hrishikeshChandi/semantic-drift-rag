import os
import json
from langchain_core import documents
import numpy as np
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from config.constants import EMBEDDINGS_MODEL
from filelock import FileLock
from core.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, index_path: str):
        self.embeddings = EMBEDDINGS_MODEL
        self.db: Optional[FAISS] = None
        self.retriever = None
        self.index_path = index_path
        self.documents_path = os.path.join(index_path, "documents.json")

    def get_embeddings_model(self):
        return self.embeddings

    def _save_documents(self, documents: List[Document]):
        logger.debug(f"Saving {len(documents)} documents to {self.documents_path}")
        os.makedirs(self.index_path, exist_ok=True)

        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(documents)} documents")

    def _load_documents(self) -> Optional[List[Document]]:
        logger.debug(f"Loading documents from {self.documents_path}")
        if not os.path.exists(self.documents_path):
            logger.warning(f"No documents found at {self.documents_path}")
            return None

        with open(self.documents_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in docs_data
        ]

        logger.info(f"Loaded {len(documents)} documents from {self.documents_path}")
        return documents

    def _save_stats(self, vectors, centroid):
        distances = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            dist = 1 - np.dot(vec, centroid)
            distances.append(dist)

        mu = np.mean(distances)
        sigma = np.std(distances)

        if sigma < 1e-6:
            sigma = 1e-6

        stats = np.array([mu, sigma])
        lock = FileLock(os.path.join(self.index_path, "corpus_stats.lock"))
        with lock:
            np.save(os.path.join(self.index_path, "corpus_stats.npy"), stats)
        logger.debug(
            f"Saved stats (mu: {mu:.4f}, sigma: {sigma:.4f}) to {self.index_path}/corpus_stats.npy"
        )

    def _save_centroid(self):
        index = self.db.index
        vectors = index.reconstruct_n(0, index.ntotal)
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        lock = FileLock(os.path.join(self.index_path, "centroid.lock"))
        with lock:
            np.save(os.path.join(self.index_path, "corpus_centroid.npy"), centroid)
        self._save_stats(vectors, centroid)

    def _create_or_load_db(self, documents: List[Document]):
        logger.debug(f"Checking for existing FAISS index at {self.index_path}")
        index_file = os.path.join(self.index_path, "index.faiss")

        if os.path.exists(index_file):
            self.db = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            stats_path = os.path.join(self.index_path, "corpus_stats.npy")
            centroid_path = os.path.join(self.index_path, "corpus_centroid.npy")
            if not os.path.exists(stats_path) or not os.path.exists(centroid_path):
                logger.warning("Missing centroid/stats. Recomputing...")
                self._save_centroid()
        else:
            logger.info(f"Creating new FAISS index at {self.index_path}")
            os.makedirs(self.index_path, exist_ok=True)
            self.db = FAISS.from_documents(documents, self.embeddings)
            self.db.save_local(self.index_path)
            self._save_centroid()
            self._save_documents(documents)

    def create_retriever(self, documents: List[Document], k: int = 4):
        logger.info("Creating retriever with provided documents")
        self._create_or_load_db(documents)

        dense_retriever = self.db.as_retriever(search_kwargs={"k": k})
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = k

        self.retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever], weights=[0.7, 0.3]
        )
        logger.info("Hybrid retriever created (dense + BM25)")

    def load_retriever(self, k: int = 4) -> bool:
        logger.debug(f"Loading retriever from {self.index_path}")
        index_file = os.path.join(self.index_path, "index.faiss")

        if not os.path.exists(index_file):
            logger.warning(f"No index found at {self.index_path}")
            return False

        try:
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.db = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            documents = self._load_documents()

            if documents is None:
                logger.warning("No documents found. Dense-only retriever created.")
                self.retriever = self.db.as_retriever(search_kwargs={"k": k})
                return True

            dense_retriever = self.db.as_retriever(search_kwargs={"k": k})
            sparse_retriever = BM25Retriever.from_documents(documents)
            sparse_retriever.k = k

            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever], weights=[0.7, 0.3]
            )
            logger.info("Hybrid retriever loaded (dense + BM25 rebuilt from JSON)")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_retriever(self):
        logger.debug("Retrieving retriever instance")
        if self.retriever is None:
            raise ValueError(
                "Retriever not created. Call create_retriever() or load_retriever() first."
            )
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        logger.debug(f"🔍 Retrieving documents for query: {query}")
        return self.get_retriever().invoke(query)
