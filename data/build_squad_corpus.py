"""Builds a real, publicly-licensed corpus from SQuAD v1.1 (validation split).

Why SQuAD: it is real Wikipedia prose (not synthetic), CC BY-SA 4.0 licensed,
and — unlike a plain Wikipedia dump — ships with a `title` per article and a
`context` paragraph per question, so document and section boundaries are
already given rather than guessed. Each (question, answer, context) triple
becomes a qrel: question -> the exact `Doc_ID §Section` the answer came from.
That is what makes it useful here: it lets G4 (citation grounding) and the
retrieval benchmarks be checked against real gold evidence instead of only
synthetic fixtures.

This corpus is used exactly like fixtures/dev_corpus: to prove the
format-driven ingestion pipeline behaves the same on content it has never
seen, per project context §8 ("the engine must behave identically on an
unseen corpus"). It is not the hidden evaluation corpus and never will be.

Usage:
    python -m data.build_squad_corpus --n-articles 12 --out data/corpus
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from pathlib import Path

SOURCE_URL = "https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/validation-00000-of-00001.parquet"
LICENSE_NOTE = (
    "Source: SQuAD v1.1 (validation split), Rajpurkar et al. 2016. "
    "Derived from Wikipedia. Licensed CC BY-SA 4.0. "
    "https://huggingface.co/datasets/rajpurkar/squad"
)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def download(raw_path: Path) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print(f"downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, raw_path)
    return raw_path


def build(n_articles: int, seed: int, out_dir: Path, raw_path: Path) -> None:
    import pandas as pd

    download(raw_path)
    df = pd.read_parquet(raw_path)

    titles = sorted(df["title"].unique())
    # Deterministic selection: every seed picks the same n_articles from the
    # same sorted title list, so the corpus is reproducible without pinning
    # a random module dependency on pandas' sampling internals.
    step = max(1, len(titles) // max(n_articles, 1))
    selected = titles[seed % step :: step][:n_articles] if step > 1 else titles[:n_articles]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qrels: list[dict] = []
    doc_index: dict[str, str] = {}

    for i, title in enumerate(selected):
        doc_id = f"Doc_{i:02d}"
        doc_index[title] = doc_id
        article = df[df["title"] == title]

        # Preserve first-seen order of unique contexts as section order.
        seen_contexts: list[str] = []
        for context in article["context"]:
            if context not in seen_contexts:
                seen_contexts.append(context)
        section_of = {c: str(n + 1) for n, c in enumerate(seen_contexts)}

        lines = [
            "---", f"doc_id: {doc_id}", f"title: {title.replace('_', ' ')}",
            f"source: squad_v1.1_validation", "---", "",
            f"# {title.replace('_', ' ')}", "",
        ]
        for n, context in enumerate(seen_contexts, start=1):
            lines.append(f"## §{n} Passage {n}")
            lines.append("")
            lines.append(context.strip())
            lines.append("")
        (out_dir / f"{doc_id}_{_slug(title)}.md").write_text("\n".join(lines), encoding="utf-8")

        for _, row in article.iterrows():
            section = section_of[row["context"]]
            answers = list(dict.fromkeys(row["answers"]["text"])) if len(row["answers"]["text"]) else []
            qrels.append({
                "query": row["question"],
                "relevant": [f"{doc_id} §{section}"],
                "answers": answers,
                "kind": "squad_real",
            })

    with open(out_dir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for row in qrels:
            f.write(json.dumps(row) + "\n")

    (out_dir / "SOURCE.md").write_text(
        f"{LICENSE_NOTE}\n\nArticles included ({len(selected)}): "
        + ", ".join(t.replace('_', ' ') for t in selected) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(selected)} documents and {len(qrels)} qrels to {out_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-articles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="data/corpus")
    parser.add_argument("--raw", default="data/raw/squad_validation.parquet")
    args = parser.parse_args(argv)
    build(args.n_articles, args.seed, Path(args.out), Path(args.raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
