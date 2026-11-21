"""End to end: question -> retrieve -> generate -> one row per test question.
Writes pipeline_outputs.json (passed/rating_note left empty for YOU to fill in) and a _run_info.json summary.
Run: uv run python pipeline.py      WARNING: overwrites pipeline_outputs.json (and your ratings)."""
import json
import os
import statistics

import config
from generate import generate_answer
from retrieve import retrieve_chunks
from run_log import ollama_ps, run_info, save_json

OUTPUT = os.getenv("PIPELINE_OUTPUT", "pipeline_outputs.json")


def answer_question(query):
    retrieval = retrieve_chunks(query)
    generation = generate_answer(query, retrieval["retrieved_chunks"])
    return {**generation, "retrieval": retrieval}


def main():
    with open("test_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for q in questions:
        r = answer_question(q["question"])
        chunks = r["retrieval"]["retrieved_chunks"]
        sources = [c["source"] for c in chunks]
        # WHY: some questions name several PDFs ("a.pdf, b.pdf") - check each one separately
        expected = [s.strip() for s in q["source_document"].split(",")] if q.get("source_document") else []
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q.get("difficulty"),
            "type": q.get("type"),
            "expected_answer": q["expected_answer"],
            "actual_answer": r["answer"],
            "retrieved_context": [c["content"] for c in chunks],
            "context_sources": [{"source": c["source"], "page": c["page"], "score": c["relevance_score"]} for c in chunks],
            # WHY: automatic retrieval check - were ALL expected PDFs among the top-K? (None for unanswerable)
            "source_in_top_k": all(s in sources for s in expected) if expected else None,
            "expected_sources_found": {s: s in sources for s in expected},
            "model": r["model"],
            "thinking": r["thinking"],
            "finish_reason": r["finish_reason"],
            "embedding_model": config.EMBED_MODEL,
            "collection": r["retrieval"]["collection"],
            "top_k": r["retrieval"]["top_k"],
            "prompt_tokens_estimate": r["prompt_tokens_estimate"],
            "prompt_tokens": r["prompt_tokens"],
            "tokens_generated": r["tokens_generated"],
            "retrieval_time_ms": r["retrieval"]["retrieval_time_ms"],
            "generation_time_ms": r["generation_time_ms"],
            "passed": None,
            "rating_note": "",
        })
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"[{q['id']}] {r['finish_reason']:<6} {r['generation_time_ms']:>6} ms  top={chunks[0]['source']} p.{chunks[0]['page']}  | {r['answer'][:70]!r}")

    def stats(key):
        values = [x[key] for x in rows if x[key] is not None]
        return {"min": min(values), "median": statistics.median(values), "max": max(values)}

    save_json(OUTPUT.replace(".json", "_run_info.json"), {
        **run_info(),
        "output_file": OUTPUT,
        "questions": len(rows),
        "empty_answers": [x["id"] for x in rows if not x["actual_answer"]],
        "finish_not_stop": [x["id"] for x in rows if x["finish_reason"] != "stop"],
        "expected_source_not_in_top_k": [x["id"] for x in rows if x["source_in_top_k"] is False],
        "prompt_tokens": stats("prompt_tokens"),
        "retrieval_time_ms": stats("retrieval_time_ms"),
        "generation_time_ms": stats("generation_time_ms"),
        "ollama_ps": ollama_ps(),
    })

    # ---------------- sanity checks ----------------
    print("\n=== PIPELINE SANITY CHECK ===")
    print(f"Rows written: {len(rows)} / questions: {len(questions)}")
    print(f"Empty answers: {[x['id'] for x in rows if not x['actual_answer']] or 'none'}")
    print(f"finish_reason not 'stop': {[x['id'] for x in rows if x['finish_reason'] != 'stop'] or 'none'}")
    print(f"Expected source NOT in top-{config.TOP_K}: {[x['id'] for x in rows if x['source_in_top_k'] is False] or 'none'}")
    print(f"Max prompt_tokens: {max(x['prompt_tokens'] or 0 for x in rows)} (context {config.CONTEXT_LENGTH})")

if __name__ == "__main__":
    main()