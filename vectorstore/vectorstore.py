import os
import json
import numpy as np
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from config.constants import EMBEDDINGS_MODEL
from filelock import FileLock
from core.logging_config import get_logger
from sklearn.cluster import KMeans
from kneed import KneeLocator

logger = get_logger(__name__)

MIN_CHUNKS_FOR_CLUSTERING = 10


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

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _find_optimal_k(self, vectors: np.ndarray) -> int:
        n = vectors.shape[0]

        if n < MIN_CHUNKS_FOR_CLUSTERING:
            logger.info(f"Too few vectors ({n}) for clustering, using k=1")
            return 1

        k_max = max(2, int(np.sqrt(n / 2)))

        if k_max < 2:
            return 1

        k_values = list(range(1, k_max + 1))
        wcss = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(vectors)
            wcss.append(km.inertia_)

        if len(wcss) == 1:
            return 1

        knee = KneeLocator(
            x=k_values,
            y=wcss,
            curve="convex",
            direction="decreasing",
            interp_method="interp1d",
        )

        if knee.knee is not None:
            optimal_k = knee.knee
            logger.info(
                f"KneeLocator found elbow at k={optimal_k} "
                f"(tested k=1..{k_max}, wcss={[round(w, 2) for w in wcss]})"
            )
        else:
            # kneed couldn't find a clear elbow — fall back to k=2
            optimal_k = 2
            logger.warning(
                f"KneeLocator could not find a clear elbow "
                f"(tested k=1..{k_max}, wcss={[round(w, 2) for w in wcss]}), "
                f"falling back to k={optimal_k}"
            )

        return optimal_k

    def _save_clusters(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        optimal_k = self._find_optimal_k(normalized)

        if optimal_k == 1:
            centroid = np.mean(normalized, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            cluster_centroids = centroid[np.newaxis, :]
        else:
            logger.info(f"Running final KMeans with optimal k={optimal_k}")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
            kmeans.fit(normalized)
            cluster_centroids = kmeans.cluster_centers_

            cc_norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
            cc_norms = np.where(cc_norms == 0, 1, cc_norms)
            cluster_centroids = cluster_centroids / cc_norms

        lock = FileLock(os.path.join(self.index_path, "cluster_centroids.lock"))
        with lock:
            np.save(
                os.path.join(self.index_path, "cluster_centroids.npy"),
                cluster_centroids,
            )
        logger.info(f"Saved {optimal_k} cluster centroids")

    def _save_stats(self, vectors: np.ndarray, cluster_centroids: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        distances = []
        for vec in normalized:
            sims = cluster_centroids @ vec
            nearest_dist = 1 - float(np.max(sims))
            distances.append(nearest_dist)

        distances = np.array(distances)
        mu = float(np.mean(distances))
        sigma = float(np.std(distances))

        if sigma < 1e-6:
            sigma = 1e-6

        stats = np.array([mu, sigma])
        lock = FileLock(os.path.join(self.index_path, "corpus_stats.lock"))
        with lock:
            np.save(os.path.join(self.index_path, "corpus_stats.npy"), stats)
        logger.debug(f"Saved stats (mu: {mu:.4f}, sigma: {sigma:.4f})")

    def _save_centroid(self):
        index = self.db.index
        vectors = index.reconstruct_n(0, index.ntotal)

        centroid = np.mean(vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        lock = FileLock(os.path.join(self.index_path, "centroid.lock"))
        with lock:
            np.save(os.path.join(self.index_path, "corpus_centroid.npy"), centroid)

        self._save_clusters(vectors)

        cluster_centroids = np.load(
            os.path.join(self.index_path, "cluster_centroids.npy")
        )
        self._save_stats(vectors, cluster_centroids)

    def _create_or_load_db(self, documents: List[Document]):
        # logger.debug(f"Checking for existing FAISS index at {self.index_path}")
        # index_file = os.path.join(self.index_path, "index.faiss")

        # if os.path.exists(index_file):
        #     self.db = FAISS.load_local(
        #         self.index_path, self.embeddings, allow_dangerous_deserialization=True
        #     )
        #     stats_path = os.path.join(self.index_path, "corpus_stats.npy")
        #     centroid_path = os.path.join(self.index_path, "corpus_centroid.npy")
        #     if not os.path.exists(stats_path) or not os.path.exists(centroid_path):
        #         logger.warning("Missing centroid/stats. Recomputing...")
        #         self._save_centroid()
        # else:
        #     logger.info(f"Creating new FAISS index at {self.index_path}")
        #     os.makedirs(self.index_path, exist_ok=True)
        #     self.db = FAISS.from_documents(documents, self.embeddings)
        #     self.db.save_local(self.index_path)
        #     self._save_centroid()
        #     self._save_documents(documents)

        logger.info(f"Rebuilding FAISS index at {self.index_path}")
        # Remove old index completely
        if os.path.exists(self.index_path):
            import shutil

            shutil.rmtree(self.index_path)

        os.makedirs(self.index_path, exist_ok=True)

        # Build fresh index
        self.db = FAISS.from_documents(documents, self.embeddings)
        self.db.save_local(self.index_path)

        # Recompute centroid, clusters, stats
        self._save_centroid()

        # Save documents (for BM25)
        self._save_documents(documents)

        logger.info("Index rebuilt successfully")

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
        logger.debug(f"Retrieving documents for query: {query}")
        return self.get_retriever().invoke(query)
