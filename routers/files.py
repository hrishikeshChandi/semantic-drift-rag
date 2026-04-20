import uvicorn, os, shutil, zipfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from core.limiter import limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from document_ingestion.processor import DocumentProcessor
from vectorstore.vectorstore import VectorStore
from config.constants import UPLOAD_ROOT, HOST, PORT, MODULE
from graph_builder.builder import GraphBuilder
from llm.llm import llm, evaluator
from drift_detector.detector import DriftDetector
from fastapi import APIRouter
from core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# @limiter.limit("15/minute")
# @router.post("/upload")
# async def upload_files(
#     request: Request,
#     user_id: str = Form(...),
#     files: List[UploadFile] = File(...),
# ):
#     logger.info(
#         f"Received upload request from user_id: {user_id} with {len(files)} files"
#     )
#     user_dir = os.path.join(UPLOAD_ROOT, user_id)
#     os.makedirs(user_dir, exist_ok=True)

#     saved_files = []

#     for file in files:
#         file_path = os.path.join(user_dir, file.filename)

#         if file.filename.endswith(".zip"):
#             with open(file_path, "wb") as f:
#                 shutil.copyfileobj(file.file, f)
#             with zipfile.ZipFile(file_path, "r") as zip_ref:
#                 zip_ref.extractall(user_dir)
#                 for extracted_file in zip_ref.namelist():
#                     extracted_path = os.path.join(user_dir, extracted_file)
#                     if os.path.isfile(extracted_path) and not extracted_file.endswith(
#                         ".zip"
#                     ):
#                         saved_files.append(extracted_path)
#             os.remove(file_path)
#         else:
#             with open(file_path, "wb") as f:
#                 shutil.copyfileobj(file.file, f)

#             saved_files.append(file_path)

#     try:
#         logger.info(f"Processing documents for user_id: {user_id}")
#         processor = DocumentProcessor()
#         documents = processor.load_and_split_documents(saved_files)
#         index_path = os.path.join(user_dir, "faiss_index")
#         vs = VectorStore(index_path=index_path)
#         vs.create_retriever(documents)
#     except Exception as e:
#         logger.error(
#             f"Failed to process documents for user {user_id}: {str(e)}",
#             exc_info=True,
#         )
#         raise HTTPException(status_code=500, detail=str(e))

#     return {
#         "status": "success",
#         "files": [f.filename for f in files],
#         "message": "Documents processed and indexed",
#     }


@limiter.limit("15/minute")
@router.post("/upload")
async def upload_files(
    request: Request,
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    logger.info(
        f"Received upload request from user_id: {user_id} with {len(files)} files"
    )

    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    os.makedirs(user_dir, exist_ok=True)

    # Save uploaded files (same as before)
    for file in files:
        file_path = os.path.join(user_dir, file.filename)

        if file.filename.endswith(".zip"):
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(user_dir)

            os.remove(file_path)

        else:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

    try:
        logger.info(f"Processing ALL documents for user_id: {user_id}")

        processor = DocumentProcessor()

        # scan ALL files (cumulative)
        all_files = []
        for root, _, file_list in os.walk(user_dir):
            for f in file_list:
                if f.lower().endswith((".pdf", ".txt")):
                    all_files.append(os.path.join(root, f))

        logger.info(f"Total files for indexing: {len(all_files)}")

        documents = processor.load_and_split_documents(all_files)

        index_path = os.path.join(user_dir, "faiss_index")
        vs = VectorStore(index_path=index_path)

        # This already rebuilds fully (as you fixed earlier)
        vs.create_retriever(documents)

    except Exception as e:
        logger.error(
            f"Failed to process documents for user {user_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "files": [f.filename for f in files],
        "message": "Documents processed and indexed (cumulative)",
    }


@router.get("/{user_id}")
async def list_files(request: Request, user_id: str):
    logger.info(f"Received file list request for user_id: {user_id}")
    user_dir = os.path.join(UPLOAD_ROOT, user_id)
    if not os.path.exists(user_dir):
        return {"files": []}
    files = [
        f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))
    ]
    return {"files": files}


@limiter.limit("15/minute")
@router.delete("/{user_id}")
async def delete_files(request: Request, user_id: str):
    logger.info(f"Received delete request for user_id: {user_id}")
    user_dir = os.path.join(UPLOAD_ROOT, user_id)

    if not os.path.exists(user_dir):
        return {"status": "no files"}

    # for item in os.listdir(user_dir):
    #     item_path = os.path.join(user_dir, item)

    #     if os.path.isfile(item_path) or os.path.islink(item_path):
    #         os.remove(item_path)
    #     elif os.path.isdir(item_path):
    #         shutil.rmtree(item_path)
    shutil.rmtree(user_dir)
    return {"status": "deleted"}
