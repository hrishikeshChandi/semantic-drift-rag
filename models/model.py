from pydantic import BaseModel

print("Defining models...")


class EvaluatorResponse(BaseModel):
    score: float
    refined_query: str
    suggestion: str


print("Models defined.")
