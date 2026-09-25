"""python -m eval.miniwiki [--n 300] [--seed 0]

Out-of-distribution check: builds a corpus and single-question scenarios from
RAG-mini-Wikipedia (rag-datasets/rag-mini-wikipedia; derived from the CMU
Question-Answer Dataset, Smith et al. 2008, CC BY-SA), whose questions were
written by students rather than SQuAD crowdworkers — so a SQuAD-trained
component cannot have seen this question distribution.

  data/corpus_miniwiki/            3,200 passages, one section each
  eval/scenarios_miniwiki/         n answerable single-question scenarios

There are no passage labels, only answers, so only answer recall and claim
precision (by answer containment) are scored. Yes/no questions and answers
longer than 8 words are skipped: string containment cannot score them.
Used only to compare components, never to tune them.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from .heysquad import _scenario


def build(n: int, seed: int) -> dict:
    import pandas as pd
    passages = pd.read_parquet("data/raw/passages.parquet")["passage"].tolist()
    qa = pd.read_parquet("data/raw/test.parquet")
    corpus = Path("data/corpus_miniwiki")
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    for i, text in enumerate(passages):
        (corpus / f"Doc_{i:04d}.md").write_text(
            f"---\ndoc_id: Doc_{i:04d}\ntitle: \n---\n\n## §1 Passage\n\n{text.strip()}\n", encoding="utf-8")
    rows = []
    for q, a in zip(qa["question"], qa["answer"]):
        a = str(a).strip()
        if not a or a.lower().rstrip(".") in ("yes", "no") or len(a.split()) > 8 or str(q).strip() in {r["query"] for r in rows}:
            continue
        rows.append({"query": str(q).strip(), "relevant": [], "answers": [a], "kind": "miniwiki"})
    with open(corpus / "qrels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    out = Path("eval/scenarios_miniwiki")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, r in enumerate(picked):
        (out / f"mw_{i:04d}.json").write_text(json.dumps(_scenario(f"mw_{i:04d}", r["query"], r["query"], False),
                                                          indent=1), encoding="utf-8")
    return {"passages": len(passages), "usable_questions": len(rows), "scenarios": len(picked)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(build(args.n, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
