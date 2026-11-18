"""
Shared config for all demo scripts.
Defaults to local Ollama. Set env vars to switch to another server or to cloud.
"""
import os
import re
from pathlib import Path
from openai import OpenAI

BENCHMARK_MODELS = ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M", "gemma4:e4b"]

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b-q4_K_M")

TOKEN_LIMIT_PARAM = os.getenv("TOKEN_LIMIT_PARAM", "max_tokens")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "8192"))

CLOUD_BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT", "")
CLOUD_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "gpt-5.6-luna")
CLOUD_TOKEN_LIMIT_PARAM = os.getenv("CLOUD_TOKEN_LIMIT_PARAM", "max_completion_tokens")


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def get_embed_client() -> OpenAI:
    return OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)


def get_cloud_client() -> OpenAI | None:
    if not (CLOUD_BASE_URL and CLOUD_API_KEY):
        return None
    return OpenAI(base_url=CLOUD_BASE_URL, api_key=CLOUD_API_KEY)


# ---------------------------------------------------------------------------
# Ingestion & retrieval 
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents")))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_BASE = os.getenv("COLLECTION_BASE", "bootcamp_docs")

# "recursive": baseline, RecursiveCharacterTextSplitter per page.
# "sections":  one chunk per heading section (heading + paragraphs + lists + tables),
#              long sections split, short siblings merged.
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))            # recursive: max chunk length
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))       # both: overlap when a split is needed
SECTION_MAX_CHARS = int(os.getenv("SECTION_MAX_CHARS", "1500"))  # sections: split above this
SECTION_MIN_CHARS = int(os.getenv("SECTION_MIN_CHARS", "200"))   # sections: merge below this
SKIP_SECTIONS = ["Table of Contents"]

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

EMBED_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "qwen3-embedding": (
        "",
        "Instruct: Given a question about internal company documents, "
        "retrieve passages that answer the question\nQuery:",
    ),
}
_doc_prefix, _query_prefix = next(
    (p for name, p in EMBED_PREFIXES.items() if EMBED_MODEL.startswith(name)), ("", "")
)
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", _doc_prefix)
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", _query_prefix)


def collection_name() -> str:
    """Collection name for the current embedding model and chunk strategy.

    WHY: vectors from different models have different dimensions (nomic 768,
    qwen3-embedding:0.6b 1024) and cannot share a collection; one collection per
    model + strategy lets both builds exist side by side for comparison.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", EMBED_MODEL).strip("-")
    return f"{COLLECTION_BASE}__{slug}__{CHUNK_STRATEGY}"