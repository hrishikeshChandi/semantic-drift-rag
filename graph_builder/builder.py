import time
from langgraph.graph import StateGraph, START, END
from state.rag_state import State
from nodes.nodes import Nodes

from core.logging_config import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    def __init__(self, retriever, llm, evaluator, user_id: str):
        logger.info("Initializing GraphBuilder with retriever, llm, and evaluator...")
        self.nodes = Nodes(retriever, llm, evaluator, user_id)
        self.graph = None

    def build(self):
        logger.debug("Building the graph...")
        builder = StateGraph(State)

        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("evaluator", self.nodes.evaluate_answer)
        builder.add_node("responder", self.nodes.generate_answer)

        builder.set_entry_point("retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", "evaluator")
        builder.add_conditional_edges(
            "evaluator",
            self.nodes.router,
            {
                "retriever": "retriever",
                "end": END,
            },
        )

        self.graph = builder.compile()
        logger.debug("Graph built successfully.")
        return self.graph

    def run(self, question: str) -> dict:
        start_time = time.time()
        if not self.graph:
            self.build()
        initial_state = State(question=question)
        logger.debug(f"Running the graph with initial state: {initial_state}")
        result = self.graph.invoke(initial_state)
        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f}s")
        return result
