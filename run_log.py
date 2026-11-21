"""Small helpers that save run information to JSON files."""
import json
import urllib.request
from datetime import datetime, timezone

import config


def ollama_ps():
    """Same info as `ollama ps` (model, VRAM, context), straight from the Ollama API."""
    url = config.EMBED_BASE_URL.removesuffix("/v1") + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)["models"]
    except Exception as e:
        return f"not available: {e}"


def run_info():
    """The settings this run used, so every number can be traced to a config."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "collection": config.collection_name(),
        "chunk_strategy": config.CHUNK_STRATEGY,
        "chunk_size_chars": config.CHUNK_SIZE,
        "chunk_overlap_chars": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "temperature": config.TEMPERATURE,
        "reasoning_effort": config.REASONING_EFFORT,
        "max_tokens": config.MAX_TOKENS,
        "context_length": config.CONTEXT_LENGTH,
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")