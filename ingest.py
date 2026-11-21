"""
Ingestion: PDFs in documents/ -> text per page -> chunks with metadata -> embeddings -> ChromaDB.
The collection is deleted and rebuilt on every run (cosine space, stable IDs, explicit embeddings).
config.CHUNK_STRATEGY picks "recursive" (baseline) or "sections". Stats go to ingest_stats_<strategy>.json.
Run: uv run python ingest.py            (add --dry-run to chunk only, nothing embedded or stored)
"""
import re
import statistics
import sys
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config

CHARS_PER_TOKEN = 4.4  # rough estimate, only for the printed summary


@dataclass
class Line:
    """One extracted line. heading_size is set when the line is a heading."""
    text: str
    page: int
    heading_size: float | None = None
    path: tuple[str, ...] = ()  # headings this line sits under


@dataclass
class Section:
    """A heading path and the body lines under it."""
    path: tuple[str, ...]
    lines: list[Line] = field(default_factory=list)

    def body_chars(self) -> int:
        return sum(len(line.text) for line in self.lines)


# ---------------------------------------------------------------------------
# 1. Extraction
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, no whitespace: lets a layout line match a font fragment."""
    return re.sub(r"\s+", "", text).lower()


def find_heading_runs(reader: PdfReader) -> list[list[tuple[str, float]]]:
    """Per page: heading texts and font sizes. A heading = bold text larger than the body font
    (the most common non-bold size by character count in the document)."""
    fragments_per_page = []
    for page in reader.pages:
        fragments = []

        def visit(text, cm, tm, font_dict, font_size, fragments=fragments):
            if not text.strip():
                fragments.append(("", 0.0, False))  # whitespace keeps runs together
                return
            # WHY: effective size = font size scaled by the text and page matrices.
            size = round(font_size * abs(tm[3]) * abs(cm[3]) * 2) / 2
            fragments.append((text, size, "Bold" in str((font_dict or {}).get("/BaseFont", ""))))

        page.extract_text(visitor_text=visit)
        fragments_per_page.append(fragments)

    sizes = Counter()
    for fragments in fragments_per_page:
        for text, size, bold in fragments:
            if text and not bold:
                sizes[size] += len(text)
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0

    runs_per_page = []
    for fragments in fragments_per_page:
        runs, current = [], None
        for text, size, bold in fragments:
            if not text:
                continue
            if bold and size > body_size + 0.5:
                if current is not None and current[1] == size:
                    current[0] += text
                else:
                    current = [text, size]
                    runs.append(current)
            else:
                current = None
        runs_per_page.append([(_norm(text), size) for text, size in runs])
    return runs_per_page


def heading_size_of(raw_line: str, runs: list[tuple[str, float]]) -> float | None:
    """Font size if raw_line is (part of) a heading on this page, else None."""
    if not raw_line or raw_line[0].isspace():
        return None
    key = _norm(raw_line)
    if len(key) < 3:
        return None
    for run, size in runs:
        if run == key or run.startswith(key) or run.endswith(key):
            return size
    return None


def fix_pipe_table(block: list[str]) -> list[str]:
    """Flattened Markdown pipe table -> one row per line."""
    joined = " ".join(line.strip() for line in block)
    # WHY: "| |" is where one row ends and the next starts.
    return re.sub(r"\|\s*\|", "|\n|", joined).splitlines()


def rebuild_layout_table(block: list[str]) -> list[str]:
    """One table row from layout mode (cells wrapped over several lines) -> "cell | cell"."""
    width = max(len(line) for line in block)
    padded = [line.ljust(width) for line in block]
    # WHY: a column that is blank in every line separates cells; a gap of 1 is a space inside a cell.
    mask = "".join(" " if all(line[i] == " " for line in padded) else "x" for i in range(width))
    row = []
    for m in re.finditer(r"x+(?: x+)*", mask):
        words = " ".join(line[m.start():m.end()].strip() for line in padded if line[m.start():m.end()].strip())
        row.append(re.sub(r"\s+", " ", words))
    return [" | ".join(row)]


def extract_lines(path: Path) -> list[Line]:
    """PDF -> Lines (text, page, heading size), using pypdf layout mode."""
    reader = PdfReader(path)
    lines: list[Line] = []

    for page_no, (page, runs) in enumerate(zip(reader.pages, find_heading_runs(reader)), start=1):
        # WHY: layout mode keeps table cells on the same row (plain mode puts each cell on its own line).
        blocks, current = [], []
        for raw_line in page.extract_text(extraction_mode="layout").splitlines():
            if raw_line.strip():
                current.append(raw_line.rstrip())
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)

        for block in blocks:
            sizes = [heading_size_of(raw_line, runs) for raw_line in block]
            if any(sizes):
                texts = [re.sub(r"\s+", " ", b).strip() for b in block]
            elif "|---" in "".join(block) or sum(b.count("|") for b in block) >= 6:
                texts, sizes = fix_pipe_table(block), None
            elif any(re.search(r"\S {3,}\S", b.strip()) for b in block):
                texts, sizes = rebuild_layout_table(block), None
            else:
                texts, sizes = [re.sub(r"\s+", " ", b).strip() for b in block], None
            for text, size in zip(texts, sizes or [None] * len(texts)):
                lines.append(Line(text, page_no, size))
            lines.append(Line("", page_no))

    # Join a heading that wraps over two lines ("Meeting 2: ... Policy" + "Updates").
    merged: list[Line] = []
    for line in lines:
        if line.heading_size and merged and merged[-1].heading_size == line.heading_size:
            merged[-1].text = f"{merged[-1].text} {line.text}"
        else:
            merged.append(line)
    return merged


def annotate_paths(lines: list[Line]) -> None:
    """Give every line its stack of headings, e.g. (title, chapter, section).
    A heading closes every open heading of the same or smaller font size."""
    stack: list[tuple[float, str]] = []
    for line in lines:
        if line.heading_size:
            while stack and stack[-1][0] <= line.heading_size:
                stack.pop()
            stack.append((line.heading_size, line.text))
        line.path = tuple(text for _, text in stack)


# ---------------------------------------------------------------------------
# 2. Chunking (all sizes in characters)
# ---------------------------------------------------------------------------

def tidy(lines: list[Line]) -> list[Line]:
    """Drop repeated, leading and trailing blank lines."""
    out: list[Line] = []
    for line in lines:
        if not line.text and (not out or not out[-1].text):
            continue
        out.append(line)
    while out and not out[-1].text:
        out.pop()
    return out


def split_lines(lines: list[Line], splitter) -> list[tuple[str, Line, Line]]:
    """Split the joined text of lines; return (piece, first line, last line) for page metadata."""
    lines = tidy(lines)
    if not lines:
        return []
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line.text) + 1  # +1 for the "\n" join
    text = "\n".join(line.text for line in lines)

    pieces, cursor = [], 0
    for piece in splitter.split_text(text):
        start = text.find(piece, cursor)
        start = cursor if start < 0 else start
        first = lines[bisect_right(starts, start) - 1]
        last = lines[bisect_right(starts, start + len(piece) - 1) - 1]
        pieces.append((piece, first, last))
        cursor = start + 1
    return pieces


def chunk_recursive(lines: list[Line]) -> list[tuple[str, Line, Line, str]]:
    """Baseline: RecursiveCharacterTextSplitter (CHUNK_SIZE / CHUNK_OVERLAP characters) per page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    pieces = []
    for page in sorted({line.page for line in lines}):
        for piece, first, last in split_lines([l for l in lines if l.page == page], splitter):
            pieces.append((piece, first, last, " > ".join(first.path)))
    return pieces


def build_sections(lines: list[Line]) -> list[Section]:
    """One Section per heading: its heading path and the lines until the next heading."""
    sections: list[Section] = []
    for line in lines:
        if line.heading_size or not sections:
            sections.append(Section(path=line.path))
            if line.heading_size:
                continue
        sections[-1].lines.append(line)
    skip = set(config.SKIP_SECTIONS)
    return [s for s in sections if tidy(s.lines) and not skip.intersection(s.path)]


def merge_small_sections(sections: list[Section]) -> list[Section]:
    """Merge a section under SECTION_MIN_CHARS into the next one with the same parent."""
    merged: list[Section] = []
    for section in sections:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.body_chars() < config.SECTION_MIN_CHARS
            and section.path[: len(prev.path) - 1] == prev.path[:-1]
            and prev.body_chars() + section.body_chars() <= config.SECTION_MAX_CHARS
        ):
            common = prev.path[:-1]
            prev_heads = [Line(t, prev.lines[0].page) for t in prev.path[len(common):]]
            next_heads = [Line(t, section.lines[0].page) for t in section.path[len(common):]]
            body = prev_heads + prev.lines + [Line("", prev.lines[0].page)] + next_heads + section.lines
            merged[-1] = Section(path=common, lines=body)
        else:
            merged.append(section)
    return merged


def chunk_sections(lines: list[Line]) -> list[tuple[str, Line, Line, str]]:
    """Heading section = chunk, heading path on top. Sections over SECTION_MAX_CHARS are split,
    and every piece keeps the heading path (so a piece of a table still says where it is from)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.SECTION_MAX_CHARS, chunk_overlap=config.CHUNK_OVERLAP)
    pieces = []
    for section in merge_small_sections(build_sections(lines)):
        path = " > ".join(section.path)
        header = f"[{path}]\n" if path else ""
        for piece, first, last in split_lines(section.lines, splitter):
            pieces.append((header + piece, first, last, path))
    return pieces


def chunk_document(path: Path) -> list[dict]:
    """One PDF -> chunk dicts with a stable id, text and metadata."""
    lines = extract_lines(path)
    annotate_paths(lines)
    if config.CHUNK_STRATEGY == "sections":
        pieces = chunk_sections(lines)
    elif config.CHUNK_STRATEGY == "recursive":
        pieces = chunk_recursive(lines)
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {config.CHUNK_STRATEGY!r}")

    return [
        {
            "id": f"{path.stem}_p{first.page}_c{index}",  # WHY: stable ID, re-ingest never duplicates
            "text": text,
            "metadata": {
                "source": path.name,
                "page": first.page,
                "page_end": last.page,
                "chunk_index": index,
                "section": section,
                "strategy": config.CHUNK_STRATEGY,
                "char_count": len(text),
            },
        }
        for index, (text, first, last, section) in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# 3. Embedding and storage
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed chunks with config.EMBED_MODEL in batches, with the document prefix."""
    client = config.get_embed_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        # WHY: nomic-embed-text needs "search_document: " here and "search_query: " in retrieve.py.
        batch = [config.EMBED_DOC_PREFIX + t for t in texts[i:i + config.EMBED_BATCH_SIZE]]
        response = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        print(f"  embedded {len(vectors)}/{len(texts)}", end="\r")
    print()
    return vectors


def store(chunks: list[dict], vectors: list[list[float]]):
    """Delete and recreate the collection, then add chunks with their vectors."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        client.delete_collection(config.collection_name())
    except (NotFoundError, ValueError):  # WHY: first run, nothing to delete
        pass
    collection = client.create_collection(
        name=config.collection_name(),
        metadata={
            "hnsw:space": "cosine",  # WHY: 1 - distance is then a real similarity in [0, 1]
            "embed_model": config.EMBED_MODEL,
            "embed_doc_prefix": config.EMBED_DOC_PREFIX,
            "chunk_strategy": config.CHUNK_STRATEGY,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "section_max_chars": config.SECTION_MAX_CHARS,
            "section_min_chars": config.SECTION_MIN_CHARS,
        },
    )
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
            embeddings=vectors[i:i + 500],  # WHY: explicit, so Chroma never embeds with its own model
        )
    return collection


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def chunk_stats(chunks: list[dict]) -> dict:
    """Chunk counts per document and chunk length (characters)."""
    lengths = [c["metadata"]["char_count"] for c in chunks]
    return {
        "chunks_total": len(chunks),
        "chunks_per_document": dict(sorted(Counter(c["metadata"]["source"] for c in chunks).items())),
        "chunk_length_chars": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
    }


def main() -> None:
    pdfs = sorted(Path(config.DOCUMENTS_DIR).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.DOCUMENTS_DIR}")

    if config.CHUNK_STRATEGY == "recursive":
        print(f"Strategy: recursive | chunk {config.CHUNK_SIZE} chars, overlap {config.CHUNK_OVERLAP} chars")
    else:
        print(f"Strategy: sections | max {config.SECTION_MAX_CHARS} chars, min {config.SECTION_MIN_CHARS} chars, "
              f"overlap {config.CHUNK_OVERLAP} chars")

    chunks = [chunk for pdf in pdfs for chunk in chunk_document(pdf)]
    stats = chunk_stats(chunks)
    for source, count in stats["chunks_per_document"].items():
        print(f"  {source:<40} {count:>4} chunks")
    length = stats["chunk_length_chars"]
    print(f"  length in characters: min {length['min']}, median {length['median']:.0f}, max {length['max']} "
          f"(max ≈ {length['max'] / CHARS_PER_TOKEN:.0f} tokens, estimated)")
    if "--dry-run" in sys.argv:
        print(f"\nDry run: {len(chunks)} chunks, nothing embedded or stored.")
        return

    print(f"\nEmbedding with {config.EMBED_MODEL} (doc prefix {config.EMBED_DOC_PREFIX!r})")
    start = time.time()
    vectors = embed_texts([c["text"] for c in chunks])
    embed_seconds = round(time.time() - start, 1)
    print(f"  {len(vectors)} vectors, {len(vectors[0])} dimensions, {embed_seconds} s")

    collection = store(chunks, vectors)
    print(f"\nCollection: {collection.name} at {config.CHROMA_PATH}")
    print(f"Chunks stored: {collection.count()}")

    from run_log import run_info, save_json
    save_json(f"ingest_stats_{config.CHUNK_STRATEGY}.json", {
        **run_info(),
        **stats,
        "chunks_stored": collection.count(),
        "embedding_dimensions": len(vectors[0]),
        "embedding_seconds": embed_seconds,
    })


if __name__ == "__main__":
    main()