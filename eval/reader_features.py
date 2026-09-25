"""python -m eval.reader_features --corpus-dir DIR [--set k=v ...] SWEEP.json [...]

Adds the SQuAD 2.0 reader's answer-vs-no-answer margin to saved refusal-sweep
outputs, in place, without re-running the engine: each recorded sub-query is
re-retrieved with the same (deterministic) retriever configuration and its
top evidence chunks are read by the reader. Margin = best span score minus
the lowest no-answer score over those chunks (the rule used by
eval/model_zoo._reader_claim).
"""
from __future__ import annotations

import argparse
import asyncio
import json

from streaming_rag.config import load_config
from streaming_rag.contracts import SubQuery
from streaming_rag.retrieval import HybridRetriever

from .model_zoo import SquadReader
from .run_quality import apply_overrides


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(), args.overrides)
    config.corpus_dir = args.corpus_dir
    retriever = HybridRetriever(config)
    asyncio.run(retriever.setup())
    reader = SquadReader()
    n_chunks = config.synthesis.ce_evidence_chunks
    for path in args.paths:
        data = json.load(open(path, encoding="utf-8"))
        n = 0
        for item in data["items"]:
            for sub in item["subs"]:
                ev = retriever._search_sync(SubQuery(query_id="r", text=sub["query"], intent_label="",
                                                     trigger="final", utterance_id="r"), config.retrieval.k)
                reads = reader.read(sub["query"], [e.chunk.text for e in ev[:n_chunks]])
                if reads:
                    best = max(reads, key=lambda r: r["score"])
                    sub["reader_margin"] = best["score"] - min(r["null"] for r in reads)
                    sub["reader_span"] = best["text"]
                n += 1
        json.dump(data, open(path, "w", encoding="utf-8"), indent=1)
        print(f"{path}: reader margin added for {n} sub-queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
