"""
Ingestion: PDFs in documents -> text per page -> chunks with metadata
(source, page, chunk_index) -> embeddings from config.EMBED_MODEL -> ChromaDB.
The collection is deleted and rebuilt on every run (cosine space, stable IDs,
explicit embeddings). config.CHUNK_STRATEGY picks "recursive" (baseline) or "sections".
"""
import argparse
import re
import statistics
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

CHARS_PER_TOKEN = 4.4  # estimated


@dataclass
class Line:
    """One line of extracted text. heading_size is set when the line is a heading."""
    text: str
    page: int
    heading_size: float | None = None
    path: tuple[str, ...] = ()  # headings this line sits under (filled by annotate_paths)


@dataclass
class Section:
    """A heading path and the body lines under it (the heading itself is in path)."""
    path: tuple[str, ...]
    lines: list[Line] = field(default_factory=list)

    def body_chars(self) -> int:
        return sum(len(line.text) for line in self.lines)


# ---------------------------------------------------------------------------
# 1. Extraction
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, whitespace removed: lets a layout line match a font fragment."""
    return re.sub(r"\s+", "", text).lower()


def find_heading_runs(reader: PdfReader) -> list[list[tuple[str, float]]]:
    """Return, per page, the heading texts and their font sizes.

    A heading is bold text larger than the body font. The body font is the most
    common size by character count in this document (11 pt in the handbook).
    """
    fragments_per_page = []
    for page in reader.pages:
        fragments = []

        def visit(text, cm, tm, font_dict, font_size, fragments=fragments):
            if not text.strip():
                fragments.append(("", 0.0, False))  # whitespace: keeps runs together
                return
            # WHY: effective size = Tf size scaled by the text and page matrices.
            size = round(font_size * abs(tm[3]) * abs(cm[3]) * 2) / 2
            base_font = str((font_dict or {}).get("/BaseFont", ""))
            fragments.append((text, size, "Bold" in base_font))

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
    """Restore one row per line in a flattened Markdown pipe table."""

    joined = " ".join(line.strip() for line in block)
    return re.sub(r"\|\s*\|", "|\n|", joined).splitlines()


def rebuild_layout_table(block: list[str]) -> list[str]:
    """Turn one table row from layout mode into a single "cell | cell" line."""

    width = max(len(line) for line in block)
    padded = [line.ljust(width) for line in block]
    blank = [all(line[i] == " " for line in padded) for i in range(width)]

    cells, start, i = [], None, 0
    while i < width:
        if not blank[i]:
            start = i if start is None else start
            i += 1
            continue
        gap_end = i
        while gap_end < width and blank[gap_end]:
            gap_end += 1
        if start is not None and (gap_end - i >= 2 or gap_end == width):
            cells.append((start, i))
            start = None
        i = gap_end
    if start is not None:
        cells.append((start, width))

    row = []
    for lo, hi in cells:
        words = " ".join(line[lo:hi].strip() for line in padded if line[lo:hi].strip())
        row.append(re.sub(r"\s+", " ", words))
    return [" | ".join(row)]


def extract_lines(path: Path) -> list[Line]:
    """Extract a PDF into Lines (text, page, heading size), page by page."""

    reader = PdfReader(path)
    runs_per_page = find_heading_runs(reader)
    lines: list[Line] = []

    for page_no, (page, runs) in enumerate(zip(reader.pages, runs_per_page), start=1):
        raw = page.extract_text(extraction_mode="layout")
        blocks, current = [], []
        for raw_line in raw.splitlines():
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
            sizes = sizes or [None] * len(texts)
            for text, size in zip(texts, sizes):
                lines.append(Line(text, page_no, size))
            lines.append(Line("", page_no))

    return merge_wrapped_headings(lines)


def merge_wrapped_headings(lines: list[Line]) -> list[Line]:
    """Join a heading that wraps over two lines ("Meeting 2: ... Policy" + "Updates")."""
    merged: list[Line] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if (line.heading_size and prev and prev.heading_size == line.heading_size):
            prev.text = f"{prev.text} {line.text}"
        else:
            merged.append(line)
    return merged


def annotate_paths(lines: list[Line]) -> None:
    """Give every line the stack of headings above it, e.g. (title, chapter, section).

    A heading closes every open heading of the same or a smaller font size.
    """
    stack: list[tuple[float, str]] = []
    for line in lines:
        if line.heading_size:
            while stack and stack[-1][0] <= line.heading_size:
                stack.pop()
            stack.append((line.heading_size, line.text))
        line.path = tuple(text for _, text in stack)


# ---------------------------------------------------------------------------
# 2. Chunking
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
    """Split the joined text of lines; return (piece, first line, last line)."""

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
        end = start + len(piece) - 1
        first = lines[bisect_right(starts, start) - 1]
        last = lines[bisect_right(starts, end) - 1]
        pieces.append((piece, first, last))
        cursor = start + 1
    return pieces


def chunk_recursive(lines: list[Line]) -> list[tuple[str, Line, Line, str]]:
    """Baseline: RecursiveCharacterTextSplitter on each page separately.

    CHUNK_SIZE and CHUNK_OVERLAP are characters. Headings stay in the text as
    ordinary lines; chunks never cross a page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    pieces = []
    for page in sorted({line.page for line in lines}):
        for piece, first, last in split_lines([l for l in lines if l.page == page], splitter):
            pieces.append((piece, first, last, " > ".join(first.path)))
    return pieces


def build_sections(lines: list[Line]) -> list[Section]:
    """One Section per heading: its full heading path and the lines until the next heading."""
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
    """Merge a section below SECTION_MIN_CHARS into the next one under the same parent."""

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
            page = prev.lines[0].page

            def heading_lines(path, page):
                return [Line(text, page) for text in path[len(common):]]

            body = (heading_lines(prev.path, page) + prev.lines + [Line("", page)]
                    + heading_lines(section.path, section.lines[0].page) + section.lines)
            merged[-1] = Section(path=common, lines=body)
        else:
            merged.append(section)
    return merged


def chunk_sections(lines: list[Line]) -> list[tuple[str, Line, Line, str]]:
    """Structure-aware chunks: heading section = chunk, with its heading path on top.

    Sections longer than SECTION_MAX_CHARS (characters, body only) are split with
    RecursiveCharacterTextSplitter; every piece gets the same heading path, so a
    piece of a table still says which chapter and section it belongs to.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.SECTION_MAX_CHARS, chunk_overlap=config.CHUNK_OVERLAP
    )
    pieces = []
    for section in merge_small_sections(build_sections(lines)):
        path = " > ".join(section.path)
        header = f"[{path}]\n" if path else ""
        for piece, first, last in split_lines(section.lines, splitter):
            pieces.append((header + piece, first, last, path))
    return pieces


def chunk_document(path: Path) -> list[dict]:
    """Extract and chunk one PDF into dicts with id, text and metadata."""
    lines = extract_lines(path)
    annotate_paths(lines)
    if config.CHUNK_STRATEGY == "sections":
        pieces = chunk_sections(lines)
    elif config.CHUNK_STRATEGY == "recursive":
        pieces = chunk_recursive(lines)
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {config.CHUNK_STRATEGY!r}")

    chunks = []
    for index, (text, first, last, section) in enumerate(pieces):
        chunks.append({
            "id": f"{path.stem}_p{first.page}_c{index}",
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
        })
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding and storage
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks with config.EMBED_MODEL, in batches."""

    client = config.get_embed_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = [config.EMBED_DOC_PREFIX + t for t in texts[i:i + config.EMBED_BATCH_SIZE]]
        response = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        print(f"  embedded {len(vectors)}/{len(texts)}", end="\r")
    print()
    return vectors


def store(chunks: list[dict], vectors: list[list[float]]):
    """Delete and recreate the collection, then add chunks with their vectors."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    name = config.collection_name()
    try:
        client.delete_collection(name)
    except (NotFoundError, ValueError):
        pass

    collection = client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": config.EMBED_MODEL,
            "embed_doc_prefix": config.EMBED_DOC_PREFIX,
            "chunk_strategy": config.CHUNK_STRATEGY,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "section_max_chars": config.SECTION_MAX_CHARS,
            "section_min_chars": config.SECTION_MIN_CHARS,
        },
    )
    batch = 500
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
            embeddings=vectors[i:i + batch],
        )
    return collection


# ---------------------------------------------------------------------------
# 4. Reporting and main
# ---------------------------------------------------------------------------

def print_summary(chunks: list[dict]) -> None:
    """Chunk counts per document and chunk length statistics (characters)."""
    per_source = Counter(c["metadata"]["source"] for c in chunks)
    for source, count in sorted(per_source.items()):
        print(f"  {source:<40} {count:>4} chunks")
    lengths = [c["metadata"]["char_count"] for c in chunks]
    print(
        f"  length in characters: min {min(lengths)}, median {statistics.median(lengths):.0f}, "
        f"max {max(lengths)} (max ≈ {max(lengths) / CHARS_PER_TOKEN:.0f} tokens, estimated)"
    )


def print_headings(path: Path) -> None:
    """Print the detected heading tree of one PDF, to check heading detection."""
    print(f"\n=== Headings: {path.name} ===")
    for line in extract_lines(path):
        if line.heading_size:
            print(f"  p{line.page:<3} {line.heading_size:>5.1f}pt  {line.text}")


def print_samples(chunks: list[dict], n: int) -> None:
    """Print n chunks spread evenly over the whole corpus."""
    step = max(1, len(chunks) // n)
    for chunk in chunks[::step][:n]:
        meta = chunk["metadata"]
        print(f"\n--- {chunk['id']}  (pages {meta['page']}-{meta['page_end']}, "
              f"{meta['char_count']} chars)  section: {meta['section'] or '-'}")
        print(chunk["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="extract and chunk only; skip embeddings and ChromaDB")
    parser.add_argument("--headings", action="store_true",
                        help="print the heading tree detected in each PDF")
    parser.add_argument("--show", type=int, default=0, metavar="N",
                        help="print N sample chunks")
    args = parser.parse_args()

    pdfs = sorted(Path(config.DOCUMENTS_DIR).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.DOCUMENTS_DIR}")

    print(f"Strategy: {config.CHUNK_STRATEGY} | "
          + (f"chunk_size {config.CHUNK_SIZE} chars, overlap {config.CHUNK_OVERLAP} chars"
             if config.CHUNK_STRATEGY == "recursive" else
             f"section max {config.SECTION_MAX_CHARS} chars, min {config.SECTION_MIN_CHARS} "
             f"chars, overlap {config.CHUNK_OVERLAP} chars"))
    chunks = []
    for pdf in pdfs:
        chunks += chunk_document(pdf)
        if args.headings:
            print_headings(pdf)
    print()
    print_summary(chunks)
    if args.show:
        print_samples(chunks, args.show)
    if args.dry_run:
        print(f"\nDry run: {len(chunks)} chunks, nothing embedded or stored.")
        return

    print(f"\nEmbedding with {config.EMBED_MODEL} (doc prefix {config.EMBED_DOC_PREFIX!r})")
    start = time.time()
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"  {len(vectors)} vectors, {len(vectors[0])} dimensions, "
          f"{time.time() - start:.1f} s")

    collection = store(chunks, vectors)
    print(f"\nCollection: {collection.name} at {config.CHROMA_PATH}")
    print(f"Chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()