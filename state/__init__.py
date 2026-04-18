from typing import List
from pydantic import BaseModel
from langchain_classic.schema import Document


class State(BaseModel):
    question: str
    docs: List[Document] = []
    answer: str = ""
