"""
inspect_extraction.py: look at extracted PDF text BEFORE chunking.

Compares pypdf's plain extract_text() with extraction_mode="layout" on one page,
prints both, and saves them to extraction_samples/ so the parsing decision in
decision_log.md can point to a real file.

Usage (PowerShell, from the project folder):
    uv run python inspect_extraction.py                      # scan all PDFs, list likely table pages
    uv run python inspect_extraction.py product_guide.pdf 3  # compare page 3 (1-based) of one PDF
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader

DOCS_DIR = Path("documents")
OUT_DIR = Path("extraction_samples")

# Force UTF-8 so printing never fails.
sys.stdout.reconfigure(encoding="utf-8")


def extract_both(page) -> tuple[str, str]:
    """Return (plain, layout) text for one pypdf page."""
    plain = page.extract_text() or ""
    # WHY: layout mode keeps horizontal positions, so table cells stay on one row.
    # layout_mode_space_vertically=False stops it from inserting many blank lines.
    layout = page.extract_text(
        extraction_mode="layout", layout_mode_space_vertically=False
    ) or ""
    return plain, layout


def fix_pipe_tables(text: str) -> str:
    """Restore one-row-per-line in markdown pipe tables that the PDF flattened.

    Consecutive lines containing "|" are treated as one table block: the PDF's
    line wrapping inside that block is undone, then every "| |" (end of one row,
    start of the next) becomes "|\\n|". Lines without "|" are left untouched.
    """
    out, block = [], []

    def flush() -> None:
        if block:
            # WHY: line breaks inside the block come from PDF wrapping, not from the table.
            joined = " ".join(ln.strip() for ln in block)
            # WHY: "| |" is where one row ends and the next starts.
            # Caveat: an EMPTY cell also looks like "| |"; check_pipe_rows() catches that.
            joined = re.sub(r"\|\s+\|", "|\n|", joined)
            out.extend(joined.splitlines())
            block.clear()

    for line in text.splitlines():
        if "|" in line:
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def check_pipe_rows(text: str) -> list[str]:
    """Return warnings for pipe-table rows whose cell count differs from the table's usual count."""
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    if not rows:
        return []
    counts = [ln.count("|") - 1 for ln in rows]
    usual = max(set(counts), key=counts.count)
    return [f"{c} cells (expected {usual}): {ln[:70]}" for ln, c in zip(rows, counts) if c != usual]


def text_stats(text: str) -> dict:
    """Simple numbers that show a 'table fell apart' problem."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    short = [ln for ln in lines if len(ln.strip()) <= 15]
    return {
        "chars": len(text),
        "non_empty_lines": len(lines),
        "avg_line_len": round(sum(len(ln.strip()) for ln in lines) / len(lines), 1) if lines else 0,
        # WHY: plain extraction often puts one table cell per line -> many very short lines.
        "short_lines_le_15_chars": len(short),
        "pipe_chars": text.count("|"),
    }


def table_score(layout_text: str) -> int:
    """Heuristic: count lines with 2+ wide gaps (column-like) or pipe separators."""
    score = 0
    for ln in layout_text.splitlines():
        if len(re.findall(r"\S\s{3,}\S", ln)) >= 2 or ln.count("|") >= 2:
            score += 1
    return score


def scan_all() -> None:
    """List every PDF with page count and the pages most likely to contain a table."""
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR.resolve()}")
        return
    for pdf in pdfs:
        reader = PdfReader(pdf)
        scores = []
        for i, page in enumerate(reader.pages, start=1):
            _, layout = extract_both(page)
            scores.append((table_score(layout), i))
        top = [f"p{i} (score {s})" for s, i in sorted(scores, reverse=True)[:3] if s > 0]
        print(f"{pdf.name}: {len(reader.pages)} pages | likely table pages: {', '.join(top) or 'none detected'}")
    print("\nNext: uv run python inspect_extraction.py <file.pdf> <page>")
    print("The score is only a hint. Open the PDF and confirm the page really has a table.")


def compare_page(pdf_name: str, page_no: int) -> None:
    """Print and save plain vs layout extraction for one page (1-based)."""
    pdf_path = DOCS_DIR / pdf_name
    reader = PdfReader(pdf_path)
    if not 1 <= page_no <= len(reader.pages):
        sys.exit(f"{pdf_name} has {len(reader.pages)} pages; got page {page_no}.")

    plain, layout = extract_both(reader.pages[page_no - 1])
    cleaned = fix_pipe_tables(plain)

    views = [("PLAIN extract_text()", plain), ('extraction_mode="layout"', layout)]
    if cleaned != plain:
        views.append(("PLAIN + fix_pipe_tables()", cleaned))

    for label, text in views:
        print("=" * 80)
        print(f"{label}  |  {pdf_name}  page {page_no}")
        print("=" * 80)
        # WHY: repr-style markers make trailing spaces and line breaks visible.
        for ln in text.splitlines():
            print(f"{ln}⏎")
        print()

    print("=" * 80)
    print("STATS")
    print("=" * 80)
    sp, sl = text_stats(plain), text_stats(layout)
    print(f"{'metric':<26}{'plain':>10}{'layout':>10}")
    for key in sp:
        print(f"{key:<26}{sp[key]:>10}{sl[key]:>10}")

    OUT_DIR.mkdir(exist_ok=True)
    stem = f"{Path(pdf_name).stem}_p{page_no}"
    (OUT_DIR / f"{stem}_plain.txt").write_text(plain, encoding="utf-8")
    (OUT_DIR / f"{stem}_layout.txt").write_text(layout, encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / (stem + '_plain.txt')}  and  {OUT_DIR / (stem + '_layout.txt')}")

    if cleaned != plain:
        (OUT_DIR / f"{stem}_cleaned.txt").write_text(cleaned, encoding="utf-8")
        print(f"Saved: {OUT_DIR / (stem + '_cleaned.txt')}")
        problems = check_pipe_rows(cleaned)
        print("Pipe-table row check: " + ("all rows have the same cell count" if not problems else ""))
        for p in problems:
            print(f"  WARNING {p}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        scan_all()
    elif len(sys.argv) == 3:
        compare_page(sys.argv[1], int(sys.argv[2]))
    else:
        sys.exit(__doc__)