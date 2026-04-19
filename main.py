# main.py
# Import constants FIRST - this initializes everything (logging, env, models)
from config.constants import UPLOAD_ROOT, HOST, PORT, MODULE, logger

# Now import everything else
import uvicorn, time, uuid, os
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from core.limiter import limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from vectorstore.vectorstore import VectorStore
from graph_builder.builder import GraphBuilder
from llm.llm import llm, evaluator
from drift_detector.detector import DriftDetector
from routers.files import router as files_router


app = FastAPI(
    title="Semantic Drift RAG API",
    description="""
    A strict, grounded, and self-correcting RAG system with drift detection.
    
    Features
    - Multi-format document ingestion (PDF, TXT, URLs, ZIP)
    - Hybrid retrieval (FAISS + BM25)
    - Pre-generation scope detection
    - Self-correcting LangGraph pipeline
    - Source citations with every answer
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(files_router, prefix="/files", tags=["Files"])

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


@app.get("/")
async def root():
    return {
        "service": "Semantic Drift RAG System",
        "version": "1.0.0",
        "description": "Grounded answers with live drift intelligence",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "document_management": {
                "upload_documents": {
                    "method": "POST",
                    "path": "/files/upload",
                    "description": "Upload documents (PDF, TXT, URL, ZIP)",
                    "parameters": {
                        "user_id": "string (required)",
                        "files": "file(s) (required)",
                    },
                },
                "list_files": {
                    "method": "GET",
                    "path": "/files/{user_id}",
                    "description": "List uploaded files for a user",
                },
                "delete_files": {
                    "method": "DELETE",
                    "path": "/files/{user_id}",
                    "description": "Delete all files and index for a user",
                },
            },
            "query_handling": {
                "generate_answer": {
                    "method": "POST",
                    "path": "/generate-answer",
                    "description": "Query documents with drift detection",
                    "parameters": {
                        "user_id": "string (required)",
                        "query": "string (required)",
                    },
                    "rate_limit": "15 requests per minute",
                },
                "reset_session": {
                    "method": "DELETE",
                    "path": "/session/{user_id}",
                    "description": "Reset session memory without deleting documents",
                },
            },
            "system": {
                "health_check": {
                    "method": "GET",
                    "path": "/health",
                    "description": "System health and metrics",
                },
                "root": {
                    "method": "GET",
                    "path": "/",
                    "description": "This endpoint - API documentation",
                },
            },
        },
        "example_usage": {
            "upload": "curl -X POST http://localhost:8000/files/upload -F 'user_id=user123' -F 'files=@document.pdf'",
            "query": "curl -X POST http://localhost:8000/generate-answer -F 'user_id=user123' -F 'query=What is the main argument?'",
            "list_files": "curl http://localhost:8000/files/user123",
            "reset_session": "curl -X DELETE http://localhost:8000/session/user123",
        },
        "rate_limits": {
            "upload": "15 requests per minute",
            "generate_answer": "15 requests per minute",
            "delete_files": "15 requests per minute",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "Semantic Drift RAG",
        "version": "1.0.0",
    }


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
    request_id = str(uuid.uuid4())[:8]
    request_start = time.time()
    logger.info(f"[{request_id}] Received query from {user_id}: {query[:100]}")

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
            f"[{request_id}] Query refused for user {user_id}: {query[:100]} - Reason: {drift_result['reason']}"
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
        logger.info(f"[{request_id}] Query needs clarification: {drift_warning}")

    graph = GraphBuilder(retriever, llm, evaluator, user_id=user_id)
    result = graph.run(query)

    if drift_warning:
        result["warning"] = drift_warning
        result["answer"] = f"Warning: {drift_warning}\n\n" + result["answer"]

    total_time = time.time() - request_start
    logger.info(
        f"[{request_id}] Request completed in {total_time:.2f}s - Decision: {drift_result['decision']}"
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
