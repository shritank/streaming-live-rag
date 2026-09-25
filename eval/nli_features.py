"""python -m eval.nli_features IN.json [IN2.json ...]

Adds NLI features (cross-encoder/nli-deberta-v3-xsmall, trained on SNLI +
MultiNLI, not SQuAD) to saved refusal-sweep outputs, in place. For each
recorded claim: premise = the claim sentence, hypothesis = the question with
its question word removed ("What did Microsoft announce that it would
rename OneDrive to?" -> "did Microsoft announce that it would rename OneDrive
to."). SQuAD 2.0-style unanswerable questions are often built by perturbing
the passage (swapped entities, negation, antonyms), so P(contradiction) and
P(entailment) are candidate refusal signals. Offline: the pipeline is not
re-run.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from .model_zoo import ZOO

_WH = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how( many| much| long| far)?)\b\s*", re.I)


class NliScorer:
    def __init__(self):
        from streaming_rag.retrieval import neural
        folder = ZOO / "cross-encoder--nli-deberta-v3-xsmall"
        self._session = neural._session(str(folder / "onnx" / "model.onnx"))
        self._tok = neural._tokenizer(str(folder / "tokenizer.json"), 256)
        self._feeds = neural._feeds
        cfg = json.load(open(folder / "config.json", encoding="utf-8"))
        self.labels = [cfg["id2label"][str(i)].lower() for i in range(len(cfg["id2label"]))]

    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for s in range(0, len(pairs), 32):
            enc = self._tok.encode_batch(pairs[s:s + 32])
            logits = self._session.run(None, self._feeds(self._session, enc))[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))


def hypothesis(question: str) -> str:
    q = _WH.sub("", question.strip()).rstrip("?").strip()
    return (q[:1].upper() + q[1:] + ".") if q else question


def main(argv=None) -> int:
    paths = argv if argv is not None else sys.argv[1:]
    scorer = NliScorer()
    ci, ie = scorer.labels.index("contradiction"), scorer.labels.index("entailment")
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        jobs = []
        for item in data["items"]:
            for sub in item["subs"]:
                for key in ("rd_claim", "ce_claim"):
                    if sub.get(key):
                        # the SUB-QUERY text, i.e. what the running system has
                        # (an ASR transcript for spoken input), never the gold question
                        jobs.append((sub, key, (sub[key], hypothesis(sub["query"]))))
        probs = scorer.probs([j[2] for j in jobs])
        for (sub, key, _), p in zip(jobs, probs):
            sub[f"nli_contra_{key[:2]}"] = float(p[ci])
            sub[f"nli_entail_{key[:2]}"] = float(p[ie])
        json.dump(data, open(path, "w", encoding="utf-8"), indent=1)
        print(f"{path}: {len(jobs)} claims scored (labels {scorer.labels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
