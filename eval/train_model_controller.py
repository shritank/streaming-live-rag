"""python -m eval.train_model_controller [--out PATH] [--cover 0.7]

Fits the learned stability scorer behind `controller.mode="model"` (streaming_rag/controller/model_stability.py).

Data: every partial transcript chunk of the TRAIN scenarios (DEV corpus only: eval/scenarios_real_dev and the
HeySQuAD dev sets, clean and ASR). Label: 1 if the text accumulated so far already contains at least `--cover`
(default 0.7) of the content words of some gold sub-question of that turn - i.e. "enough has been said that a
search is worth issuing" - else 0. Turns that never reach the stability scorer (chit-chat, presentation-only) are
skipped. The held-out CHECK scenarios (DIAG corpus) are scored with the fitted model and, on the same chunks, with
the rule scorer, so the two controllers are compared on data neither was tuned on.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
from pathlib import Path

import numpy as np

from streaming_rag.config import load_config
from streaming_rag.controller import model_stability, stability
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.retrieval.text import content_tokens

TRAIN = (("data/corpus", ("eval/scenarios_real_dev", "eval/scenarios_heysquad_dev_clean", "eval/scenarios_heysquad_dev_asr")),)
CHECK = (("data/corpus_test", ("eval/scenarios_real_test", "eval/scenarios_heysquad_diag_clean", "eval/scenarios_heysquad_diag_asr")),)


async def _vocab(corpus_dir: str):
    cfg = load_config()
    cfg.corpus_dir = corpus_dir
    retriever = HybridRetriever(cfg)
    await retriever.setup()
    return retriever.specific_vocabulary, retriever.low_df_bigrams


def _rows(scenario_dir: str, vocab, bigrams, cover: float):
    """(features, label, rule_confidence) for every partial chunk that reaches the stability scorer."""
    out = []
    for path in sorted(glob.glob(f"{scenario_dir}/*.json")):
        d = json.load(open(path, encoding="utf-8"))
        turns = d["ground_truth"]["turns"]
        acc: dict[tuple, str] = {}
        for e in sorted(d["events"], key=lambda e: e["timestamp_ms"]):
            if e["event_type"] != "transcript_chunk":
                continue
            p = e["payload"]
            uid = p["utterance_id"]
            gt = turns.get(uid, {})
            previous = acc.get(uid, "")
            text = previous + p["text"]
            acc[uid] = text
            if gt.get("kind") in ("chit_chat", "presentation_only") or not gt.get("gold_sub_intents") or not text.strip():
                continue
            have = set(content_tokens(text))
            ready = 0
            for intent in gt["gold_sub_intents"]:
                want = set(content_tokens(intent))
                if want and len(want & have) / len(want) >= cover:
                    ready = 1
                    break
            out.append((model_stability.features(text.strip(), previous, vocab, bigrams), ready,
                        stability.score(text.strip(), previous, False, vocab, bigrams)))
    return out


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.r_[np.ones(d), 0.0]
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * reg * w
        H = (Xb * (p * (1 - p) + 1e-6)[:, None]).T @ Xb + l2 * np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w[:-1], float(w[-1])


def _prf(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(y), "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "premature": fp, "missed": fn}


def _show(name: str, m: dict) -> None:
    print(f"  {name:34s} n={m['n']:5d}  acc {m['accuracy']*100:5.1f}%  precision {m['precision']*100:5.1f}%  "
          f"recall {m['recall']*100:5.1f}%  F1 {m['f1']*100:5.1f}%   premature searches {m['premature']:4d}  missed {m['missed']:4d}")


async def main_async(args) -> int:
    train = []
    for corpus, dirs in TRAIN:
        vocab, bigrams = await _vocab(corpus)
        for d in dirs:
            train += _rows(d, vocab, bigrams, args.cover)
    X = np.array([r[0] for r in train]); y = np.array([r[1] for r in train], dtype=float)
    w, b = _fit(X, y, args.l2)
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = _prf((p >= t).astype(int), y.astype(int))["f1"]
        if f > best_f:
            best_t, best_f = float(round(t, 2)), f
    model = {"features": list(model_stability.FEATURES), "weights": [round(float(v), 6) for v in w], "bias": round(b, 6),
             "threshold": best_t, "label_rule": f"gold sub-question content-word coverage >= {args.cover}",
             "trained_on": [d for _, ds in TRAIN for d in ds], "n_train_chunks": len(train), "l2": args.l2}
    Path(args.out).write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"trained on {len(train)} partial chunks ({int(y.sum())} 'ready', {int((1 - y).sum())} 'not yet'); threshold {best_t}")
    print("weights:", dict(zip(model["features"], model["weights"])), "bias", model["bias"])

    print("\ndecision quality per partial chunk (label = 'enough of a gold question has been spoken')")
    ytr = y.astype(int)
    _show("TRAIN (DEV)  rule scorer  (>=0.45)", _prf(np.array([r[2] >= 0.45 for r in train]).astype(int), ytr))
    _show("TRAIN (DEV)  learned model", _prf((p >= best_t).astype(int), ytr))
    for corpus, dirs in CHECK:
        vocab, bigrams = await _vocab(corpus)
        for d in dirs:
            rows = _rows(d, vocab, bigrams, args.cover)
            Xc = np.array([r[0] for r in rows]); yc = np.array([r[1] for r in rows])
            pc = 1.0 / (1.0 + np.exp(-(Xc @ w + b)))
            tag = d.replace("eval/scenarios_", "")
            _show(f"HELD-OUT {tag}  rule", _prf(np.array([r[2] >= 0.45 for r in rows]).astype(int), yc))
            _show(f"HELD-OUT {tag}  model", _prf((pc >= best_t).astype(int), yc))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(model_stability.WEIGHTS_PATH))
    ap.add_argument("--cover", type=float, default=0.7)
    ap.add_argument("--l2", type=float, default=1.0)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
