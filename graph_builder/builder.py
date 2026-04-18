from langgraph.graph import StateGraph, START, END
from state.rag_state import State
from nodes.nodes import Nodes


class GraphBuilder:
    def __init__(self, retriever, llm, evaluator, user_id: str):
        print("Initializing GraphBuilder with retriever, llm, and evaluator...")
        self.nodes = Nodes(retriever, llm, evaluator, user_id)
        self.graph = None

    def build(self):
        print("Building the graph...")
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
        print("Graph built successfully.")
        return self.graph

    def run(self, question: str) -> dict:
        if not self.graph:
            self.build()

        initial_state = State(question=question)
        print(f"Running the graph with initial state: {initial_state}")
        return self.graph.invoke(initial_state)
