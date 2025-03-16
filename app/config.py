from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# FAISS index settings
FAISS_INDEX_PATH = ROOT_DIR / "faiss_index" / "docs.index"
TOP_K_DOCS = 4  # Number of relevant documents to retrieve

# ✅ Model settings (Updated for DeepSeek)
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"  # ✅ Use DeepSeek instead of Mistral
HF_ACCESS_TOKEN = None  # No token needed for DeepSeek
DEVICE = "mps"  # For Apple Silicon, use "mps". For CUDA, use "cuda"

# Model paths
MODEL_DIR = ROOT_DIR / "models"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"  # ✅ Keep embedding model
LLM_MODEL_PATH = MODEL_DIR / "deepseek-coder-1.3b.Q4_K_M.gguf"  # ✅ Update model path if needed

# Data paths
DATA_DIR = ROOT_DIR / "data"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Retrieval settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
