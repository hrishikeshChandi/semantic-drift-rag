import os, time
from state.rag_state import State
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from core.logging_config import get_logger

logger = get_logger(__name__)

GLOBAL_CHAT_STORE = {}


class Nodes:
    def __init__(self, retriever, llm, evaluator, user_id: str):
        logger.info("Initializing nodes...")
        self.retriever = retriever
        self.llm = self._wrap_llm_with_history(llm, user_id)
        self.evaluator = evaluator
        self.user_id = user_id

    def _wrap_llm_with_history(self, llm, user_id: str):
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in GLOBAL_CHAT_STORE:
                GLOBAL_CHAT_STORE[session_id] = ChatMessageHistory()
            return GLOBAL_CHAT_STORE[session_id]

        return RunnableWithMessageHistory(llm, get_session_history)

    def retrieve_docs(self, state: State) -> State:
        logger.info(f"Retrieving documents for question: {state.question}")
        query = state.question
        if state.refined_query:
            query = f"{state.question} (Refined query: {state.refined_query})"
        state.docs = self.retriever.invoke(query)
        logger.debug(f"Retrieved {len(state.docs)} documents.")
        return state

    def generate_answer(self, state: State) -> State:
        logger.info("Generating answer based on retrieved documents...")
        context = "\n\n".join(
            [
                f"""Page content: {doc.page_content}
                File name: {os.path.basename(doc.metadata.get("source", "Unknown"))}
                Page number: {doc.metadata.get("page", "Unknown")}"""
                for doc in state.docs
            ]
        )
        state.context = context
        prompt = f"""
You are a highly capable research assistant. Your task is to answer the user's question using ONLY the provided context.

====================
STRICT RULES (Violation = incorrect answer)
====================

1. Grounding (ZERO TOLERANCE)
- Use ONLY the given context.
- Do NOT add external knowledge, assumptions, or general facts.
- If the answer is not fully present, clearly state what is missing.

2. Completeness (CRITICAL)
- Extract ALL relevant details from the context.
- If multiple points exist, ALL must be included.
- Do NOT give short or minimal answers if more information is available.

3. Depth
- Reflect the richness of the context.
- Do NOT oversimplify or compress detailed information.

4. Citation (MANDATORY)
- After EVERY factual statement, include a citation in this format:
  (Source: 'File Name', page X)
- If multiple sources apply:
  (Sources: 'File A', page 2; 'File B', page 5)
- Citations MUST be inline (not at the end).
- NEVER say "no citation needed".

5. Uncertainty Handling
- If the context is incomplete, ambiguous, or contradictory:
  clearly state the limitation and explain what is missing.

6. Forbidden (AUTOMATIC FAILURE)
- No hallucination
- No guessing
- No meta-commentary (e.g., "Based on the context...")
- No placeholder text (e.g., "[citation needed]")

====================
CONTEXT
====================
{context}

====================
QUESTION
====================
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
        start_time = time.time()
        logger.info("Evaluating the generated answer...")
        prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the answer using ONLY the provided context.

====================
SCORING RUBRIC (Total = 1.0)
====================

Faithfulness (0.3)
- 0.3 → All claims are supported by context
- 0.15 → Minor unsupported claim
- 0.0 → Any hallucination → FAIL

Relevance (0.2)
- 0.2 → Fully answers question
- 0.1 → Partially answers
- 0.0 → Off-topic

Completeness (0.3) — CRITICAL
- 0.3 → ALL important details included
- 0.15 → Some missing details
- 0.0 → Shallow or incomplete

Depth & Precision (0.2)
- 0.2 → Detailed and precise
- 0.1 → Somewhat vague
- 0.0 → Too short or oversimplified

====================
STRICT RULES
====================

- DO NOT use external knowledge
- DO NOT reward shallow answers
- If key details are missing → score must be < 0.75
- If hallucination exists → score = 0.0

====================
PENALTY RULES
====================

- If answer is significantly shorter than the amount of information in context → reduce score
- If citations are missing → subtract 0.2
- If citations are NOT inline → treat as missing
- If meta-commentary is present → subtract 0.1
- If ANY entity or claim is not explicitly present in context → score = 0.0

====================
OUTPUT FORMAT (STRICT)
====================
    score: float (0 to 1)
    suggestion: string
    refined_query: string

====================
INPUT
====================

Context:
{state.context}

Question:
{state.question}

Answer:
{state.answer}
"""
        response = self.evaluator.invoke(prompt)
        state.retries += 1
        state.score = response.score
        state.suggestion = response.suggestion
        state.refined_query = response.refined_query
        state.is_good = state.score >= 0.8
        elapsed = time.time() - start_time
        logger.info(
            f"Evaluation completed in {elapsed:.2f}s - Score: {state.score:.3f}"
        )
        return state

    def router(self, state: State):

        logger.debug(
            f"Routing based on evaluation: is_good={state.is_good}, retries={state.retries}"
        )

        if not state.is_good and state.retries < 3:
            return "retriever"
        else:
            return "end"
