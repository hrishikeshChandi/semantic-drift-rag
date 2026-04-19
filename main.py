import uvicorn, os, time
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from core.limiter import limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from vectorstore.vectorstore import VectorStore
from config.constants import UPLOAD_ROOT, HOST, PORT, MODULE
from graph_builder.builder import GraphBuilder
from llm.llm import llm, evaluator
from drift_detector.detector import DriftDetector
from routers.files import router as files_router
from core.logging_config import setup_logging, get_logger

# initialize logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE"),
)

logger = get_logger(__name__)

app = FastAPI()
app.include_router(files_router, prefix="/files", tags=["files"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.delete("/session/{user_id}")
async def delete_session(request: Request, user_id: str):
    index_path = os.path.join(UPLOAD_ROOT, user_id, "faiss_index")
    if not os.path.exists(index_path):
        return {"message": f"No session found for user_id {user_id}. Nothing to reset."}
    detector = DriftDetector(index_path=index_path)
    detector.reset_session()
    return {"message": f"Session for user_id {user_id} has been reset."}


@limiter.limit("15/minute")
@app.post("/generate-answer")
async def generate_answer(
    request: Request,
    user_id: str = Form(...),
    query: str = Form(...),
):
    request_start = time.time()
    logger.info(f"Received query from user_id: {user_id}: {query}")

    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    drift_warning = None

    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail="User data not found")

    index_path = os.path.join(user_dir, "faiss_index")

    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Please upload documents first.",
        )

    vs = VectorStore(index_path=index_path)

    if not vs.load_retriever():
        raise HTTPException(
            status_code=500,
            detail="Failed to load vector store. Please re-upload documents.",
        )

    retriever = vs.get_retriever()

    detector = DriftDetector(index_path=index_path)
    drift_result = detector.analyze(query)

    if drift_result["decision"] == "refuse":
        logger.warning(
            f"Query refused for user {user_id}: {query} - Reason: {drift_result['reason']}"
        )
        return {
            "user_id": user_id,
            "query": query,
            "answer": "I can't answer this — it appears to be outside the scope of your uploaded documents.",
            "decision": drift_result["decision"],
            "confidence": drift_result["confidence"],
            "drift": drift_result,
        }

    elif drift_result["decision"] == "ask_clarification":
        drift_warning = drift_result["reason"]

    graph = GraphBuilder(retriever, llm, evaluator, user_id=user_id)
    result = graph.run(query)

    if drift_warning:
        logger.warning(
            f"Clarification needed for user {user_id}: {query} - Reason: {drift_warning}"
        )
        result["warning"] = drift_warning
        result["answer"] = f"Warning: {drift_warning}\n\n" + result["answer"]

    total_time = time.time() - request_start
    logger.info(
        f"Request completed in {total_time:.2f}s - Decision: {drift_result['decision']}"
    )

    return {
        "user_id": user_id,
        "query": query,
        "answer": result,
        "decision": drift_result["decision"],
        "confidence": drift_result["confidence"],
        "drift": drift_result,
    }


if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT} with module {MODULE}")
    uvicorn.run(MODULE, host=HOST, port=PORT, reload=True)
