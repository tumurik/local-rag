"""Local vs cloud comparison for the Day 12 capstone (Exercise 4, Step 3).

Reads eval_results_local.json, eval_results_cloud.json, pipeline_outputs_local.json and
pipeline_outputs_cloud.json from the current folder and prints markdown you paste into the report:
  1. per metric: n, local mean +/- 2*SE, cloud mean +/- 2*SE, paired difference (cloud - local) +/- 2*SE, verdict
  2. per run: median and slowest generation time, average completion tokens, empty or failed answers
  3. per metric: the questions where cloud and local differ most (to name ids in the report)
Based on code_snippets/09_difference_real.py.

    uv run python compare.py
    (also saves the same text to compare_output.md)
"""

import json
import statistics
import sys


class _Tee:
    """Print to the console and to a UTF-8 file at once.
    WHY: PowerShell's `>` writes UTF-16 and a cp1252 console can crash on '−' or '≥'."""

    def __init__(self, path: str):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout
        try:
            self.console.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    def write(self, text):
        self.file.write(text)
        self.console.write(text)

    def flush(self):
        self.file.flush()
        self.console.flush()

LABELS = ("local", "cloud")
BIG_GAP = 0.3


def load_json(path: str) -> list[dict]:
    """Load a JSON list; exit with a clear message if the file is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Missing {path}. Run eval_pipeline.py / pipeline.py for that label first.")


def mean_2se(values: list[float]) -> tuple[float, float]:
    """Mean and 2 x standard error (stdev / sqrt(n)). SE is undefined for n < 2."""
    if not values:
        return float("nan"), float("nan")
    if len(values) < 2:
        return values[0], float("nan")
    return statistics.mean(values), 2 * statistics.stdev(values) / len(values) ** 0.5


def scores_by_id(rows: list[dict]) -> dict:
    """{question id: {metric name: score or None}}. Metrics with an error count as missing."""
    out = {}
    for r in rows:
        out[r["id"]] = {name: (m.get("score") if not m.get("error") else None)
                        for name, m in (r.get("metrics") or {}).items()}
    return out


def completion_tokens(row: dict):
    """Completion tokens as the pipeline recorded them (reference: tokens_generated)."""
    for key in ("tokens_generated", "completion_tokens"):
        if row.get(key) is not None:
            return row[key]
    return None


def is_failed(row: dict) -> bool:
    """Empty answer, a rejected request, or generation cut off by the token cap."""
    answer = (row.get("actual_answer") or "").strip()
    return (not answer or answer.startswith("[request rejected")
            or row.get("finish_reason") in ("length", "error", "content_filter"))


def fmt(m: float, e: float, signed: bool = False) -> str:
    """'0.81 ± 0.06', or '+0.04 ± 0.05' for differences."""
    sign = "+" if signed else ""
    err = "n/a" if e != e else f"{e:.2f}"  # e != e is True only for NaN
    return f"{m:{sign}.2f} ± {err}"


def metric_table(local: dict, cloud: dict) -> list[tuple[str, list[tuple[str, float]]]]:
    """Print the main table; return per-metric lists of (id, cloud - local) for the gap section."""
    metrics = sorted({m for r in local.values() for m in r} & {m for r in cloud.values() for m in r})
    print("| Metric | n | Local (mean ± 2·SE) | Cloud (mean ± 2·SE) | Cloud − local, per question | Verdict |")
    print("|---|---|---|---|---|---|")
    gaps = []
    for m in metrics:
        ids = sorted(i for i in local if i in cloud
                     and local[i].get(m) is not None and cloud[i].get(m) is not None)
        la = [local[i][m] for i in ids]
        lb = [cloud[i][m] for i in ids]
        diffs = [b - a for a, b in zip(la, lb)]
        (ma, ea), (mb, eb), (md, ed) = mean_2se(la), mean_2se(lb), mean_2se(diffs)
        if ed != ed:
            verdict = "too few questions"
        elif md - ed <= 0 <= md + ed:
            verdict = "no clear difference"
        else:
            verdict = "cloud higher" if md > 0 else "local higher"
        print(f"| {m} | {len(ids)} | {fmt(ma, ea)} | {fmt(mb, eb)} | {fmt(md, ed, signed=True)} | {verdict} |")
        gaps.append((m, list(zip(ids, diffs))))
    return gaps


def cost_table(outputs: dict) -> None:
    """Time, tokens and failures per run, from pipeline_outputs_<label>.json."""
    print("| Run | Model | Questions | Median generation (s) | Slowest (s) | Avg completion tokens | Empty / failed |")
    print("|---|---|---|---|---|---|---|")
    for label in LABELS:
        rows = outputs[label]
        times = [r["generation_time_ms"] / 1000 for r in rows if r.get("generation_time_ms") is not None]
        toks = [t for t in (completion_tokens(r) for r in rows) if t is not None]
        failed = [r["id"] for r in rows if is_failed(r)]
        model = rows[0].get("model", "?") if rows else "?"
        med = f"{statistics.median(times):.1f}" if times else "n/a"
        slow = f"{max(times):.1f}" if times else "n/a"
        avg_t = f"{statistics.mean(toks):.0f}" if toks else "n/a"
        failed_txt = f"{len(failed)}" + (f" ({', '.join(failed)})" if failed else "")
        print(f"| {label} | {model} | {len(rows)} | {med} | {slow} | {avg_t} | {failed_txt} |")


def gap_section(gaps) -> None:
    """Questions where cloud and local differ by at least BIG_GAP on a metric."""
    print(f"Questions where |cloud − local| ≥ {BIG_GAP} (read the judge's reasons for these):\n")
    any_gap = False
    for m, pairs in gaps:
        big = sorted((p for p in pairs if abs(p[1]) >= BIG_GAP), key=lambda p: -abs(p[1]))
        if big:
            any_gap = True
            print(f"- {m}: " + ", ".join(f"{i} ({d:+.2f})" for i, d in big))
    if not any_gap:
        print("- none")


def main():
    sys.stdout = _Tee("compare_output.md")
    evals = {lb: scores_by_id(load_json(f"eval_results_{lb}.json")) for lb in LABELS}
    outputs = {lb: load_json(f"pipeline_outputs_{lb}.json") for lb in LABELS}
    print("### Scores (source: eval_results_local.json, eval_results_cloud.json)\n")
    gaps = metric_table(evals["local"], evals["cloud"])
    print("\nVerdict rule: 'no clear difference' when the per-question range (mean ± 2·SE) includes zero.\n")
    print("### Time, tokens, failures (source: pipeline_outputs_local.json, pipeline_outputs_cloud.json)\n")
    cost_table(outputs)
    print()
    gap_section(gaps)


if __name__ == "__main__":
    main()
