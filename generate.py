"""Chunks + question -> prompt -> local LLM -> answer.
Returns the answer AND exactly what the model saw (context_used).
Run: uv run python generate.py "your question" """
import sys
import time

import config


def generate_answer(query, chunks):
    context = "\n\n".join(f"[{i}] ({c['source']}, page {c['page']})\n{c['content']}" for i, c in enumerate(chunks, 1))
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    estimate = round(sum(len(m["content"]) for m in messages) / 4.4)  # rough: ~4.4 characters per token
    if estimate + config.MAX_TOKENS > config.CONTEXT_LENGTH:
        print(f"WARNING: prompt ~{estimate} tokens + answer {config.MAX_TOKENS} may not fit in {config.CONTEXT_LENGTH}")

    start = time.time()
    response = config.get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.TEMPERATURE,
        reasoning_effort=config.REASONING_EFFORT,           # WHY: "none" = no hidden thinking eating the cap
        **{config.TOKEN_LIMIT_PARAM: config.MAX_TOKENS},    # WHY: parameter NAME comes from config (local vs cloud)
    )
    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    if not answer:
        print(f"WARNING: EMPTY ANSWER for '{query[:50]}' finish_reason={choice.finish_reason}")

    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "model": config.LLM_MODEL,
        "thinking": config.REASONING_EFFORT != "none",
        "finish_reason": choice.finish_reason,
        "generation_time_ms": round((time.time() - start) * 1000),
        "prompt_tokens_estimate": estimate,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "tokens_generated": response.usage.completion_tokens if response.usage else None,
    }


if __name__ == "__main__":
    from retrieve import retrieve_chunks

    question = sys.argv[1] if len(sys.argv) > 1 else "How many days per week can employees work remotely?"
    g = generate_answer(question, retrieve_chunks(question)["retrieved_chunks"])

    # ---------------- sanity checks ----------------
    print(f"\n=== GENERATE SANITY CHECK ===\nQ: {question}\nA: {g['answer']}\n")
    print(f"finish_reason: {g['finish_reason']}  (want 'stop'; 'length' = cut off)")
    print(f"prompt_tokens: {g['prompt_tokens']} (estimate {g['prompt_tokens_estimate']}) - if real is much LOWER, the prompt was cut")
    print(f"tokens_generated: {g['tokens_generated']} | time: {g['generation_time_ms']} ms | thinking: {g['thinking']}")