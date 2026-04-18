from typing import List
from pydantic import BaseModel
from langchain_classic.schema import Document


class State(BaseModel):
    question: str
    context: str = ""
    docs: List[Document] = []
    answer: str = ""
    retries: int = 0
    suggestion: str = ""
    refined_query: str = ""
    score: float = 0.0
    is_good: bool = False


print("RAG State model loaded.")
