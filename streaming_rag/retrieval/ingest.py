"""Corpus ingestion: directory of Markdown -> list[Chunk].

Format-driven, not content-driven (project context §8): swapping the dev
corpus for the real one in data/corpus/ requires only re-running the index
build. Nothing here knows any topic, entity or document ID in advance.

Recognised layout:
    ---
    doc_id: Doc_12
    title: ...
    ---
    # Title
    ## §2 Section heading
    body text...

`doc_id` falls back to the filename stem's leading token, and section
numbers fall back to sequential ordering, so plain Markdown also ingests.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Chunk
from .text import split_sentences

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(?:§\s*([\w.]+)\s+)?(.*)$", re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[m.end():]


def _doc_id_from_path(path: Path) -> str:
    stem = path.stem
    m = re.match(r"(Doc_[\w]+?)(?:_|$)", stem)
    return m.group(1) if m else stem


def parse_document(path: Path, max_chars: int = 600, overlap_sentences: int = 1) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    doc_id = meta.get("doc_id") or _doc_id_from_path(path)
    title = meta.get("title", "")

    matches = list(_SECTION_RE.finditer(body))
    chunks: list[Chunk] = []

    if not matches:
        # No section headings: treat the whole document as section "1".
        chunks.extend(_chunk_section(doc_id, "1", title, body.strip(), max_chars, overlap_sentences))
        return chunks

    for i, m in enumerate(matches):
        section = m.group(1) or str(i + 1)
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        chunks.extend(_chunk_section(doc_id, section, heading, section_text, max_chars, overlap_sentences,
                                      doc_title=title))
    return chunks


def _chunk_section(doc_id: str, section: str, heading: str, text: str,
                    max_chars: int, overlap_sentences: int, doc_title: str = "") -> list[Chunk]:
    if not text:
        return []
    sentences = split_sentences(text)
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for s in sentences:
        if current and size + len(s) > max_chars:
            groups.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(x) for x in current)
        current.append(s)
        size += len(s)
    if current:
        groups.append(current)

    chunks = []
    for n, group in enumerate(groups, start=1):
        body = " ".join(group)
        chunks.append(Chunk(
            chunk_id=f"{doc_id} §{section} #{n}",
            doc_id=doc_id,
            section=section,
            text=body,
            metadata={"heading": heading, "doc_title": doc_title, "n": n},
        ))
    return chunks


# Provenance/licensing notes that ship alongside a corpus are metadata, not
# content: indexing them lets a query that mentions an article title retrieve
# (and cite) the licence file instead of the article.
_METADATA_FILE_RE = re.compile(r"^(readme|license|licence|source|notice)\b", re.IGNORECASE)


def load_corpus(corpus_dir: str | Path, max_chars: int = 600) -> list[Chunk]:
    directory = Path(corpus_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    chunks: list[Chunk] = []
    for pattern in ("**/*.md", "**/*.txt"):
        for path in sorted(directory.glob(pattern)):
            if not _METADATA_FILE_RE.match(path.name):
                chunks.extend(parse_document(path, max_chars=max_chars))
    return chunks
