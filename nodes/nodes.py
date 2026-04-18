from state.rag_state import State
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class Nodes:
    def __init__(self, retriever, llm, evaluator, user_id: str):
        print("Initializing nodes...")
        self.retriever = retriever
        self.llm = self._wrap_llm_with_history(llm, user_id)
        self.evaluator = evaluator
        self.store = {}
        self.user_id = user_id

    def _wrap_llm_with_history(self, llm, user_id: str):
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(llm, get_session_history)

    def retrieve_docs(self, state: State) -> State:
        print(f"Retrieving documents for question: {state.question}")
        query = state.question
        if state.refined_query:
            query = f"{state.question} (Refined query: {state.refined_query})"
        state.docs = self.retriever.invoke(query)
        print(f"Retrieved {len(state.docs)} documents.")
        return state

    def generate_answer(self, state: State) -> State:
        print("Generating answer based on retrieved documents...")
        context = "\n\n".join(
            [
                f"""Page content: {doc.page_content}
                File name: {doc.metadata.get("source", "Unknown")}
                Page number: {doc.metadata.get("page", "Unknown")}"""
                for doc in state.docs
            ]
        )
        state.context = context
        prompt = f"""
You are a highly capable research assistant. Your task is to answer the user's question based **only** on the provided context.

Guidelines
1. Accuracy: Do not invent facts, assume, or use external knowledge. If the context lacks sufficient information, clearly state that and explain what is missing.
2. Conciseness: Provide a direct answer first, then briefly show your reasoning (if helpful). Avoid unnecessary repetition.
3. Source citation: After every factual claim, cite the source using this exact format:  
   `(Source: 'File Name', page X)`  
   If multiple sources support the same point, combine them: `(Sources: 'File A', page 2; 'File B', page 5)`.
4. Uncertainty handling: If the context is ambiguous or contradictory, acknowledge that and present the different possibilities with their respective citations.
5. No meta‑commentary: Do not write things like “Based on the context provided…” or “As an AI…”. Just answer directly.

Context
{context}

Question
{state.question}

"""
        if state.suggestion:
            prompt += f"\n\nNote: Previous evaluation suggested: {state.suggestion}\n"
        if state.refined_query:
            prompt += f"\n\nNote: Previous evaluation suggested refining the question to: {state.refined_query}\n"
        response = self.llm.invoke(
            prompt,
            config={
                "configurable": {
                    "session_id": self.user_id,
                },
            },
        )
        state.answer = response.content.strip()
        return state

    def evaluate_answer(self, state: State) -> State:
        print("Evaluating the generated answer...")
        prompt = f"""You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Your job is to evaluate the quality of an answer based ONLY on the provided context.

You must:
1. Check if the answer is grounded in the context (faithfulness)
2. Check if it fully answers the question (relevancy)
3. Check if important information is missing

IMPORTANT RULES:
- Do NOT use external knowledge
- Do NOT change the intent of the question
- Suggest improvements ONLY if needed
- Query refinement should ADD clarity, not change meaning

Score guidelines:
- 0.9–1.0: Perfect answer, fully grounded, complete
- 0.75–0.89: Mostly correct, minor gaps
- 0.5–0.74: Partial answer, missing key info
- 0.0–0.49: Incorrect or not grounded

Output Format (Strict):
    score: float (0 to 1)
    suggestion: string
    refined_query: string

Context:
{state.context}

Question:
{state.question}

Answer:
{state.answer}"""
        response = self.evaluator.invoke(prompt)
        state.retries += 1
        state.score = response.score
        state.suggestion = response.suggestion
        state.refined_query = response.refined_query
        state.is_good = state.score >= 0.75

        print(state)
        return state

    def router(self, state: State):

        print(
            f"Routing based on evaluation: is_good={state.is_good}, retries={state.retries}"
        )

        if not state.is_good and state.retries < 3:
            return "retriever"
        else:
            return "end"
