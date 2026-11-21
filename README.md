LOCAL RAG
! outputs are in pipeline_outputs/
Chosen configuration last output 19/20 is in pipeline_outputs_gemma_k10.json (best result is in k5 20/20)
rating notes written only in few outputs.

Answer queries based on provided pdf documents.
pdf chunked by sections and embedded with nomic-embed-text
stored in ChromaDB and answered by gemma4:e4b

config.py - parameters and settings
ingest.py - chunks, embeds, stores. Writes ingest_stats_<strategy>
retrieve.py - retrieves top-k chunks
generate.py - generate answer based on retrieved
benchmark.py - compares models. writes benchmark_results.json
run_log.py - helper functions to output run info.
inspect_extraction.py - extraction samples for one page. Saved in extraction_samples/

Setup:
Install Ollama, set ollama context length = 8192

Pull all models i used:
ollama pull qwen3.5:4b-q4_K_M
ollama pull qwen3.5:2b-q4_K_M

or only chosen models:
ollama pull gemma4:e4b
ollama pull nomic-embed-text

Install dependencies:
uv sync

!Put pdfs to documents/ i did not commit them.

Run:
uv run python ingest.py # rebuilds chroma_db/
uv run python pipeline.py # overwrites pipeline_outputs.json

AI assistants used: claude, gemini, gpt, deepseek. 