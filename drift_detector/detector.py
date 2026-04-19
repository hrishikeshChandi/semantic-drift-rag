import os
import json
import numpy as np
from filelock import FileLock
from config.constants import EMBEDDINGS_MODEL


class DriftDetector:
    def __init__(
        self,
        index_path: str,
    ):
        self.index_path = index_path
        self.embeddings_model = EMBEDDINGS_MODEL

        corpus_centroid_path = os.path.join(index_path, "corpus_centroid.npy")
        self.centroid = np.load(corpus_centroid_path)

        self.session_state = {}
        self.session_state_path = os.path.join(self.index_path, "session_state.json")
        self.load_or_create_session_memory()

        stats_path = os.path.join(index_path, "corpus_stats.npy")
        stats = np.load(stats_path)

        self.mu = stats[0]
        self.sigma = stats[1]

        self.drift_threshold = self.mu + 3.5 * self.sigma
        self.warning_threshold = self.mu + 2.5 * self.sigma

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

    def _cosine_distance(self, a, b) -> float:
        return 1 - np.dot(a, b)

    def _update_session(self, query: str, score: float, embedding: np.ndarray):
        lock = FileLock(self.session_state_path + ".lock")
        with lock:
            self.session_state["query_embeddings"].append(embedding.tolist())
            self.session_state["query_history"].append(query)
            self.session_state["drift_scores"].append(score)

            with open(self.session_state_path, "w") as f:
                json.dump(self.session_state, f)

    def _session_centroid(self):
        if len(self.session_state["query_embeddings"]) < 2:
            return None
        embeddings = np.array(self.session_state["query_embeddings"])
        centroid = np.mean(embeddings, axis=0)
        return centroid / np.linalg.norm(centroid)

    def analyze(self, query: str) -> dict:
        query_vector = self._embed_query(query)
        query_drift_score = self._cosine_distance(query_vector, self.centroid)
        session_centroid = self._session_centroid()

        if session_centroid is not None:
            trajectory_drift_score = self._cosine_distance(
                session_centroid, self.centroid
            )
        else:
            trajectory_drift_score = query_drift_score

        final_score = max(query_drift_score, trajectory_drift_score * 0.7)

        if final_score > self.drift_threshold:
            status = "out_of_scope"
            decision = "refuse"
            reason = "Query is far outside the semantic scope of uploaded documents"

        elif final_score > self.warning_threshold:
            status = "warning"
            decision = "ask_clarification"
            reason = "Query is near the boundary of document scope"

        else:
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
            "query_drift_score": query_drift_score,
            "trajectory_drift_score": trajectory_drift_score,
            "final_score": final_score,
            "query_history_length": len(self.session_state["query_history"]),
        }

    def reset_session(self) -> None:
        self.session_state = {
            "query_embeddings": [],
            "query_history": [],
            "drift_scores": [],
        }
        with open(self.session_state_path, "w") as f:
            json.dump(self.session_state, f)
