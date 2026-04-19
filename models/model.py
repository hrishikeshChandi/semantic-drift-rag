from pydantic import BaseModel
from core.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Defining models...")


class EvaluatorResponse(BaseModel):
    score: float
    refined_query: str
    suggestion: str


logger.debug("Models defined.")
