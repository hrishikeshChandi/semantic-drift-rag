print("Loading constants...")

import os
from dotenv import load_dotenv
import getpass
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

UPLOAD_ROOT = "data"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your GROQ API Key: ")

if not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter your OpenRouter API Key: ")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face Token: ")

MODULE = os.getenv("MODULE", "main:app")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
print("Constants loaded successfully.")

print("Loading the embedding model...")
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")