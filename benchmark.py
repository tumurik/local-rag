import json
import time
from datetime import datetime, timezone

from openai import APIConnectionError, APIStatusError

from config import BENCHMARK_MODELS, LLM_BASE_URL, TOKEN_LIMIT_PARAM, get_llm_client

OUTPUT_FILE = "benchmark_results.json"

REASONING_EFFORT = "none"
TEMPERATURE = 0.3
MAX_TOKENS = 1024


SUMMARY_PARAGRAPH = (
    "Starting in January, the company will replace its fixed office schedule with a "
    "hybrid policy. Employees may work remotely up to three days per week, provided "
    "their manager approves the arrangement in advance and core collaboration hours "
    "of 10:00 to 15:00 are respected. Teams must agree on at least one shared office "
    "day each week. Equipment for home offices, including a monitor and a chair, can "
    "be requested through the IT portal and is capped at 600 euros per employee. "
    "Remote work from outside the EU is limited to 20 working days per calendar year "
    "and requires written approval from HR because of tax and data-protection rules."
)

GROUNDED_CONTEXT = (
    "Travel policy, section 4: Economy class is mandatory for flights under 6 hours. "
    "Business class may be booked for flights of 6 hours or longer with director "
    "approval. Hotel costs are reimbursed up to 140 euros per night in EU capitals "
    "and 110 euros per night elsewhere. Receipts must be submitted within 30 days."
)

REFUSAL_PHRASE = "I cannot find this information in the provided documents."

PROMPTS = [
    {
        "category": "factual",
        "prompt": "What causes tides on Earth? Answer in 2-3 sentences.",
    },
    {
        "category": "reasoning",
        "prompt": (
            "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, what is "
            "the average speed for the entire journey? Show your working briefly, "
            "then give the final answer on its own line."
        ),
    },
    {
        "category": "summarization",
        "prompt": f"Summarize the following paragraph in at most 2 sentences:\n\n{SUMMARY_PARAGRAPH}",
    },
    {
        "category": "structured",
        "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city": ..., "country": ...}]',
    },
    {
        "category": "code",
        "prompt": (
            "Write a Python function is_palindrome(s: str) -> bool that ignores case, "
            "spaces and punctuation. Include a docstring. Return only the code in a "
            "single ```python block, with no explanation."
        ),
    },
    {
        "category": "instruction_following",
        "prompt": (
            "Give exactly 3 benefits of running an LLM locally instead of in the cloud. "
            "Format: exactly 3 lines, each starting with '- ', each at most 10 words. "
            "No introduction, no conclusion, no blank lines."
        ),
    },
    # Grounded prompts to mimic generation step.
    {
        "category": "grounded_answerable",
        "prompt": (
            f"Answer ONLY from the context. If the answer is not in the context, reply "
            f"exactly: '{REFUSAL_PHRASE}'\n\nContext:\n{GROUNDED_CONTEXT}\n\n"
            "Question: What is the hotel reimbursement limit in an EU capital?"
        ),
    },
    {
        "category": "grounded_unanswerable",
        "prompt": (
            f"Answer ONLY from the context. If the answer is not in the context, reply "
            f"exactly: '{REFUSAL_PHRASE}'\n\nContext:\n{GROUNDED_CONTEXT}\n\n"
            "Question: What is the daily meal allowance on business trips?"
        ),
    },
]

def check_structured(text: str) -> dict:
    """Check the structured prompt: a JSON list of 3 objects with city and country.

    Args:
        text: the model's raw message.content.
    Returns:
        {"passed": bool, "detail": str}
    """
    raw = text.strip()
    fenced = raw.startswith("```")
    if fenced:
        # Strip it to tell "bad JSON" from "fenced JSON".
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"passed": False, "detail": f"not valid JSON: {exc.msg}"}
    shape_ok = (
        isinstance(data, list)
        and len(data) == 3
        and all(isinstance(d, dict) and set(d) == {"city", "country"} for d in data)
    )
    if not shape_ok:
        return {"passed": False, "detail": "parses, but not a list of 3 {city, country} objects"}
    if fenced:
        return {"passed": False, "detail": "valid JSON, but wrapped in a markdown fence"}
    return {"passed": True, "detail": "valid JSON with the requested shape"}


def check_instruction_following(text: str) -> dict:
    """Check the format rules: exactly 3 lines, each '- ' and at most 10 words.

    Args:
        text: the model's raw message.content.
    Returns:
        {"passed": bool, "detail": str}
    """
    lines = text.strip().split("\n")
    problems = []
    if len(lines) != 3:
        problems.append(f"{len(lines)} lines instead of 3")
    for i, line in enumerate(lines, 1):
        if not line.startswith("- "):
            problems.append(f"line {i} does not start with '- '")
        elif len(line[2:].split()) > 10:
            problems.append(f"line {i} has {len(line[2:].split())} words")
    return {"passed": not problems, "detail": "; ".join(problems) or "all format rules met"}


def check_refusal(text: str) -> dict:
    """Check the unanswerable grounded prompt: the model should give the refusal phrase.

    Args:
        text: the model's raw message.content.
    Returns:
        {"passed": bool, "detail": str}
    """
    refused = "cannot find this information" in text.lower()
    return {"passed": refused, "detail": "refused as instructed" if refused else "answered instead of refusing"}


# Automatic checks only cover rules a script can verify (JSON shape, line format, refusal).
AUTO_CHECKS = {
    "structured": check_structured,
    "instruction_following": check_instruction_following,
    "grounded_unanswerable": check_refusal,
}


def warm_up(client, model: str) -> float:
    """Send one tiny request so the model is loaded before anything is timed.

    Args:
        client: OpenAI-compatible client from config.get_llm_client().
        model: Ollama model tag.
    Returns:
        Seconds the warm-up took (mostly model load time), for the run log.
    """
    start = time.time()
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say OK."}],
        temperature=0.0,
        reasoning_effort=REASONING_EFFORT,
        **{TOKEN_LIMIT_PARAM: 5},
    )
    return round(time.time() - start, 2)


def run_benchmark(client, model: str, prompt_item: dict) -> dict:
    """Run one prompt against one model with streaming and collect the metrics.

    Args:
        client: OpenAI-compatible client from config.get_llm_client().
        model: Ollama model tag.
        prompt_item: {"category": str, "prompt": str}.
    Returns:
        One result row (see the keys below); quality_rating/rating_note stay blank.
    """
    start = time.time()
    first_token_at = None
    parts = []
    finish_reason = None
    usage = None
    chunk_count = 0

    # Separetes generation speed and time to frist token.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_item["prompt"]}],
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        stream=True,
        stream_options={"include_usage": True},
        **{TOKEN_LIMIT_PARAM: MAX_TOKENS},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        text = delta.content or ""
        reasoning = getattr(delta, "reasoning", None) or ""
        if (text or reasoning) and first_token_at is None:
            first_token_at = time.time()
        if text or reasoning:
            chunk_count += 1
        parts.append(text)
    end = time.time()

    response_text = "".join(parts)
    completion_tokens = usage.completion_tokens if usage else chunk_count
    generation_seconds = (end - first_token_at) if first_token_at else None
    check = AUTO_CHECKS.get(prompt_item["category"])

    return {
        "model": model,
        "category": prompt_item["category"],
        "prompt": prompt_item["prompt"],
        "response": response_text,
        "thinking": REASONING_EFFORT != "none",
        "reasoning_effort": REASONING_EFFORT,
        "finish_reason": finish_reason,
        "time_seconds": round(end - start, 2),
        "ttft_seconds": round(first_token_at - start, 2) if first_token_at else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": completion_tokens,
        "completion_tokens_source": "usage" if usage else "chunk_count",
        "tokens_per_second": (
            round(completion_tokens / generation_seconds, 1)
            if generation_seconds and completion_tokens
            else None
        ),
        "auto_check": check(response_text) if check else None,
        "error": None,
        "quality_rating": None,
        "rating_note": "",
    }


def error_row(model: str, prompt_item: dict, message: str) -> dict:
    """Build a result row for a failed call so the failure is visible in the file.

    Args:
        model: Ollama model tag.
        prompt_item: {"category": str, "prompt": str}.
        message: short error description.
    Returns:
        A result row with the same keys as run_benchmark, metrics set to None.
    """
    return {
        "model": model, "category": prompt_item["category"], "prompt": prompt_item["prompt"],
        "response": "", "thinking": REASONING_EFFORT != "none", "reasoning_effort": REASONING_EFFORT,
        "finish_reason": None, "time_seconds": None, "ttft_seconds": None,
        "prompt_tokens": None, "completion_tokens": None, "completion_tokens_source": None,
        "tokens_per_second": None, "auto_check": None, "error": message,
        "quality_rating": None, "rating_note": "",
    }


def main() -> None:
    """Run every prompt on every model and write benchmark_results.json."""
    client = get_llm_client()
    results = []
    warm_up_seconds = {}

    print(f"Endpoint: {LLM_BASE_URL}")
    print(f"Models: {', '.join(BENCHMARK_MODELS)}")
    print(f"Prompts: {len(PROMPTS)} | reasoning_effort={REASONING_EFFORT} | "
          f"temperature={TEMPERATURE} | {TOKEN_LIMIT_PARAM}={MAX_TOKENS}\n")

    for model in BENCHMARK_MODELS:
        print(f"=== {model} ===")
        try:
            warm_up_seconds[model] = warm_up(client, model)
            print(f"  warm-up (load) {warm_up_seconds[model]} s")
        except (APIConnectionError, APIStatusError) as exc:
            print(f"  !!! WARNING: {model} unavailable ({type(exc).__name__}): {exc}")
            print(f"  !!! Is Ollama running, and is the model pulled? (ollama pull {model})")
            results.extend(error_row(model, p, f"{type(exc).__name__}: {exc}") for p in PROMPTS)
            continue

        for item in PROMPTS:
            try:
                row = run_benchmark(client, model, item)
            except (APIConnectionError, APIStatusError) as exc:
                print(f"  !!! WARNING: {item['category']} failed: {exc}")
                results.append(error_row(model, item, f"{type(exc).__name__}: {exc}"))
                continue
            results.append(row)
            check = row["auto_check"]
            check_text = f" | check: {'PASS' if check['passed'] else 'FAIL'}" if check else ""
            print(f"  {row['category']:<22} {row['time_seconds']:>6.2f} s total | "
                  f"TTFT {row['ttft_seconds']} s | {row['tokens_per_second']} tok/s | "
                  f"{row['completion_tokens']} tok | {row['finish_reason']}{check_text}")
            if not row["response"].strip():
                print(f"  !!! WARNING: EMPTY RESPONSE for {model} / {row['category']} "
                      f"(finish_reason={row['finish_reason']})")
            elif row["finish_reason"] == "length":
                print(f"  !!! WARNING: response cut off at the {MAX_TOKENS}-token cap")
        print()

    output = {
        "run_info": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": LLM_BASE_URL,
            "models": BENCHMARK_MODELS,
            "reasoning_effort": REASONING_EFFORT,
            "thinking": REASONING_EFFORT != "none",
            "temperature": TEMPERATURE,
            "token_limit_param": TOKEN_LIMIT_PARAM,
            "max_tokens": MAX_TOKENS,
            "warm_up_seconds": warm_up_seconds,
            "metric_notes": {
                "time_seconds": "wall time from request to last streamed chunk",
                "ttft_seconds": "time to first token (mostly prompt processing)",
                "tokens_per_second": "completion_tokens / (time_seconds - ttft_seconds): generation speed",
            },
        },
        "results": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()