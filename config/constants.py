import os
from dotenv import load_dotenv
import getpass
from langchain_huggingface import HuggingFaceEmbeddings
from core.logging_config import setup_logging, get_logger

load_dotenv()

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE"),
    console_output=os.getenv("CONSOLE_LOGS", "false").lower() == "true",
)

logger = get_logger(__name__)

logger.info("Loading configuration...")

UPLOAD_ROOT = "data"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found, prompting user")
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your GROQ API Key: ")

if not os.getenv("OPENROUTER_API_KEY"):
    logger.warning("OPENROUTER_API_KEY not found, prompting user")
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass(
        "Enter your OpenRouter API Key: "
    )

if not os.getenv("HF_TOKEN"):
    logger.warning("HF_TOKEN not found, prompting user")
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face Token: ")

MODULE = os.getenv("MODULE", "main:app")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

logger.info(f"Configuration loaded: HOST={HOST}, PORT={PORT}, MODULE={MODULE}")

logger.info("Loading the embedding model...")
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embedding model loaded successfully.")
