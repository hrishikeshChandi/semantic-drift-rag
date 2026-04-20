import os
import json
import numpy as np
from filelock import FileLock
from config.constants import EMBEDDINGS_MODEL
from typing import Optional
from core.logging_config import get_logger

logger = get_logger(__name__)


class DriftDetector:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.embeddings_model = EMBEDDINGS_MODEL

        cluster_path = os.path.join(index_path, "cluster_centroids.npy")
        centroid_path = os.path.join(index_path, "corpus_centroid.npy")

        if os.path.exists(cluster_path):
            self.cluster_centroids = np.load(cluster_path)
            logger.info(f"Loaded {len(self.cluster_centroids)} cluster centroids")
        else:
            self.cluster_centroids = np.load(centroid_path)[np.newaxis, :]
            logger.warning(
                "No cluster centroids found, falling back to single centroid"
            )

        self.session_state = {}
        self.session_state_path = os.path.join(self.index_path, "session_state.json")
        self.load_or_create_session_memory()

        stats_path = os.path.join(index_path, "corpus_stats.npy")
        stats = np.load(stats_path)
        self.mu = stats[0]
        self.sigma = stats[1]

        self.drift_threshold = self.mu + 3.5 * self.sigma
        self.warning_threshold = self.mu + 2.5 * self.sigma

        logger.info(
            f"Thresholds — warning: {self.warning_threshold:.4f}, "
            f"drift: {self.drift_threshold:.4f}, "
            f"clusters: {len(self.cluster_centroids)}"
        )

    def load_or_create_session_memory(self) -> None:
        if os.path.exists(self.session_state_path):
            with open(self.session_state_path, "r") as f:
                self.session_state = json.load(f)
        else:
            self.session_state = {
                "query_embeddings": [],
                "query_history": [],
                "drift_scores": [],
            }
            with open(self.session_state_path, "w") as f:
                json.dump(self.session_state, f)

    def _embed_query(self, query: str) -> np.ndarray:
        query_vec = self.embeddings_model.embed_query(query)
        query_vec = np.array(query_vec)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            return query_vec / norm
        return query_vec

    def _nearest_cluster_distance(self, vec: np.ndarray) -> float:
        sims = self.cluster_centroids @ vec
        return float(1 - np.max(sims))

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1 - np.dot(a, b))

    def _update_session(self, query: str, score: float, embedding: np.ndarray):
        lock = FileLock(self.session_state_path + ".lock")
        with lock:
            self.session_state["query_embeddings"].append(embedding.tolist())
            self.session_state["query_history"].append(query)
            self.session_state["drift_scores"].append(score)

            max_history = 20
            if len(self.session_state["query_embeddings"]) > max_history:
                self.session_state["query_embeddings"] = self.session_state[
                    "query_embeddings"
                ][-max_history:]
                self.session_state["query_history"] = self.session_state[
                    "query_history"
                ][-max_history:]
                self.session_state["drift_scores"] = self.session_state["drift_scores"][
                    -max_history:
                ]

            with open(self.session_state_path, "w") as f:
                json.dump(self.session_state, f)

    def _session_centroid(self) -> Optional[np.ndarray]:
        if len(self.session_state["query_embeddings"]) < 2:
            return None
        embeddings = np.array(self.session_state["query_embeddings"])
        centroid = np.mean(embeddings, axis=0)
        return centroid / np.linalg.norm(centroid)

    def analyze(self, query: str) -> dict:
        logger.debug(f"Analyzing query: {query}")
        query_vector = self._embed_query(query)

        query_drift_score = self._nearest_cluster_distance(query_vector)

        session_centroid = self._session_centroid()
        if session_centroid is not None:
            trajectory_drift_score = self._nearest_cluster_distance(session_centroid)
        else:
            trajectory_drift_score = query_drift_score

        final_score = max(query_drift_score, trajectory_drift_score * 0.7)

        if final_score > self.drift_threshold:
            logger.warning(
                f"Query out of scope. Score: {final_score:.3f} > {self.drift_threshold:.3f}"
            )
            status = "out_of_scope"
            decision = "refuse"
            reason = "Query is far outside the semantic scope of uploaded documents"

        elif final_score > self.warning_threshold:
            logger.warning(
                f"Query near boundary. Score: {final_score:.3f} > {self.warning_threshold:.3f}"
            )
            status = "warning"
            decision = "ask_clarification"
            reason = "Query is near the boundary of document scope"

        else:
            logger.debug(f"Query within scope. Score: {final_score:.3f}")
            status = "ok"
            decision = "answer"
            reason = "Query is within document scope"

        confidence = max(0.0, min(1.0, 1 - final_score))

        if status != "out_of_scope":
            self._update_session(query=query, score=final_score, embedding=query_vector)

        return {
            "status": status,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "query_drift_score": round(query_drift_score, 4),
            "trajectory_drift_score": round(trajectory_drift_score, 4),
            "final_score": round(final_score, 4),
            "nearest_cluster_distance": round(query_drift_score, 4),
            "n_clusters": len(self.cluster_centroids),
            "query_history_length": len(self.session_state["query_history"]),
        }

    def reset_session(self) -> None:
        logger.info("Resetting session state")
        self.session_state = {
            "query_embeddings": [],
            "query_history": [],
            "drift_scores": [],
        }
        with open(self.session_state_path, "w") as f:
            json.dump(self.session_state, f)
