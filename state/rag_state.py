from typing import List
from pydantic import BaseModel
from langchain_classic.schema import Document
from core.logging_config import get_logger

logger = get_logger(__name__)


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


logger.debug("RAG State model loaded.")
