from langchain.chat_models import init_chat_model

from models.model import EvaluatorResponse

print("Initializing LLMs...")
llm = init_chat_model(
    "openrouter:qwen/qwen3-next-80b-a3b-instruct:free",
    temperature=0.6,
)

evaluator = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    temperature=0,
).with_structured_output(EvaluatorResponse)

print("LLMs initialized.")
