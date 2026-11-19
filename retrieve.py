"""Question -> embed it -> find the TOP_K most similar chunks in ChromaDB.
Returns the chunks plus source, page and score (Day 12 evaluates these).
Run: uv run python retrieve.py "your question" """
import sys
import time

import chromadb

import config


def embed_query(query):
    """Embed one question with config.EMBED_MODEL. Returns [vector] (Chroma wants a list)."""
    # WHY: query prefix here, doc prefix in ingest - nomic-embed-text expects both, or neither.
    response = config.get_embed_client().embeddings.create(
        model=config.EMBED_MODEL, input=[config.EMBED_QUERY_PREFIX + query]
    )
    return [response.data[0].embedding]


def retrieve_chunks(query, top_k=config.TOP_K):
    collection = chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection(config.collection_name())
    start = time.time()
    result = collection.query(
        query_embeddings=embed_query(query),  # WHY: same embedding model as ingest
        n_results=top_k,
    )
    chunks = [
        {"content": doc, "source": meta["source"], "page": meta["page"], "relevance_score": round(1 - dist, 4)}
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
    return {
        "query": query,
        "retrieved_chunks": chunks,
        "retrieval_time_ms": round((time.time() - start) * 1000),
        "embedding_model": config.EMBED_MODEL,
        "collection": config.collection_name(),  # WHY: says which chunking build was searched
        "top_k": top_k,
    }


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How many days per week can employees work remotely?"
    r = retrieve_chunks(question)

    # ---------------- sanity checks ----------------
    print(f"\n=== RETRIEVE SANITY CHECK ===\nQuestion: {question}\nTime: {r['retrieval_time_ms']} ms\n")
    for c in r["retrieved_chunks"]:
        print(f"{c['relevance_score']:.3f}  {c['source']} p.{c['page']}  | {c['content'][:90].replace(chr(10), ' ')}")
    scores = [c["relevance_score"] for c in r["retrieved_chunks"]]
    print("\nScores between 0 and 1:", all(0 <= s <= 1 for s in scores))
    print("Sorted best first:     ", scores == sorted(scores, reverse=True))
    print("No duplicate chunks:   ", len({c["content"] for c in r["retrieved_chunks"]}) == len(scores))