"""All tunables and ablation switches for the streaming RAG engine.

Every field has a safe default so the system runs with zero configuration.
Override via environment variables (see .env.example) or by constructing
Config(**overrides) directly in tests/eval.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"          # hybrid | dense | sparse
    k: int = 8
    rrf_k: int = 60
    # Defaults chosen on the DEV split only (data/corpus qrels + scenarios;
    # final_report.docx §5): BM25 + e5 fused with equal weight, then the
    # cross-encoder reranks the top `rerank_pool` (20). On 1000 dev queries that
    # lifted r@1 from 67.6% (old BM25 + heuristic rerank, where LSA at weight
    # 0.01 contributed nothing) to ~88-89%. The pool was 5 while inference ran
    # on a laptop CPU (5 matched 10/20 on r@1 at a quarter of the cost, ~180 ms
    # vs ~800 ms per query). On the GPU a pool of 20 costs ~+8 ms per isolated query
    # (in the real-time pipeline per-sub-query retrieval goes from ~40 to ~82 ms p50,
    # all during speech; post-speech latency is unchanged) and gave a significant
    # answer-recall gain on DEV (pooled
    # text + HeySQuAD, +1.3 pts [+0.3, +2.6]) that DIAG confirmed identically
    # (final_report.docx section 5.13); 32 adds nothing over 20 at r@1 / r@5.
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    low_confidence_threshold: float = 0.15   # heuristic reranker scale only
    # lsa (TF-IDF+SVD, no download) | e5-small-v2 (ONNX, retrieval/neural.py)
    dense_encoder: str = "e5-small-v2"
    # heuristic (lexical coverage/density features; measured to HURT r@1 by
    # 6.5 points vs no reranking on dev) | none (fused order) | cross-encoder
    # (ONNX ms-marco-MiniLM-L6-v2 over the top `rerank_pool` fused candidates)
    reranker: str = "cross-encoder"
    rerank_pool: int = 20
    # max characters per chunk (sentences are never split; 1 sentence overlap)
    chunk_chars: int = 600
    # RM3-style pseudo-relevance feedback on the BM25 side: expand the query
    # with the top terms of the first-pass top documents
    bm25_prf: bool = False
    # Third fusion channel: TF-IDF over character 3-5-grams within word
    # boundaries. Matches ASR-garbled rare words ("loredo norman hurst" ~
    # "Loreto Normanhurst") that word-level BM25 cannot. 0 = off.
    char_weight: float = 0.0
    # With the cross-encoder, evidence scores are raw logits; the top hit is
    # low-confidence below this logit. Set a priori; a dev sweep over
    # {-8, -5, -3, 0} found -8 and -5 identical and -3/0 worse.
    ce_low_confidence_logit: float = -5.0
    # Adds an artificial delay to every search call, representing a realistic
    # hosted vector-DB / reranker round trip. 0 by default (the dev corpus is
    # small enough that real search is sub-millisecond); the baseline vs
    # streaming comparison sets this explicitly, because "the user waits
    # while retrieval happens after they stop talking" (the guide's problem
    # statement) only shows up in the numbers once retrieval isn't free.
    simulated_latency_ms: float = 0.0


@dataclass
class ControllerConfig:
    mode: str = "rule"            # rule | model (learned stability classifier, controller/model_stability.py)
    min_stability: float = 0.6
    provisional_stability: float = 0.45
    # mode="model": retrieve on a partial when the learned P(ready to search) reaches this; None = the value
    # fitted by `python -m eval.train_model_controller` (stored in controller/model_controller.json)
    model_threshold: float | None = None
    debounce_ms: int = 150
    # Drop conversational lead-ins ("Actually,", "Wait, one more thing —")
    # from sub-query text; they carry no search content.
    strip_discourse_markers: bool = True
    # Which clauses get the utterance's topic terms appended:
    #   short       any clause with < 4 content tokens (original behaviour)
    #   elliptical  only clauses that are not already a self-contained question
    context_carry: str = "elliptical"
    # ASR transcripts spell numbers out ("eighteen thirty"); rewrite them as
    # digits in sub-query text (controller/normalize.py)
    # Evidence-based defaults (final_report.docx §5.4): both are no-ops on
    # punctuated text (the splitter bails out the instant any punctuation is
    # present; the number normaliser only touches recognised spelled-out
    # number-word sequences), so there is no cost on clean/typed input, and
    # they close a real ~8-16pt answer-recall gap on real spoken (HeySQuAD)
    # questions.
    normalize_spoken_numbers: bool = True
    # For punctuation-free (ASR) text only: split before a question word that
    # opens a new question (controller/decompose.py)
    split_unpunctuated_questions: bool = True


@dataclass
class SynthesisConfig:
    # How answers are written:
    #   extractive  every claim is a verbatim corpus sentence (default; measured
    #               on held-out gold data, cannot invent facts)
    #   generative  an LLM writes claims, each verified against the evidence
    #               before it ships (session/generative/, ported from build A;
    #               needs a real LLM provider; NOT yet measured on gold data)
    mode: str = "extractive"
    # generative mode only: how a claim's support is checked
    #   lexical (offline) | nli (local cross-encoder) | llm (judge prompt)
    grounding_checker: str = "lexical"
    grounding_check: bool = True
    max_answer_tokens: int = 512
    # How the claim sentence is chosen and gated for relevance:
    #   lexical        best query-token-overlap sentence, lexical relevance rule
    #   cross-encoder  every sentence of the top `ce_evidence_chunks` chunks is
    #                  scored against the sub-query by the cross-encoder; the
    #                  best is asserted only if its logit >= ce_claim_threshold
    # Not the default: on dev its gain over lexical (+3.9 answer recall at
    # threshold -4) was not significant and it adds ~155 ms after the user
    # stops speaking.
    # lexical is dependency-free; cross-encoder is the evidence-based default
    # (final_report.docx §5.3/§5.8): +6.5pts answer recall on 279 real spoken
    # questions [95% CI +2.5, +10.4], consistent gains on SQuAD-derived dev
    # and diagnostic sets. Needs the same fetched models as retrieval's
    # cross-encoder reranker (already required by the production default).
    selector: str = "cross-encoder"
    ce_evidence_chunks: int = 3
    ce_claim_threshold: float = -4.0
    # chunk: candidate sentences come from the retrieved chunk | section: from
    # the whole section the chunk belongs to (parent-child retrieval)
    evidence_scope: str = "chunk"
    # With a model-based selector, choose each sub-query's claim as soon as
    # its retrieval completes (while the user is still speaking) and reuse it
    # at utterance end, instead of paying for it after the user stops.
    speculative_selection: bool = True
    # Refusal gate for the cross-encoder selector (session/refusal_gate.json):
    #   learned  logistic P(answer correct) from reranker features + the
    #            SQuAD2 reader's answer margin; below the threshold the
    #            sub-question is reported as unsupported instead of answered
    #   none     only the retrieval low-confidence gate applies
    refusal_gate: str = "learned"
    refusal_threshold: float | None = None   # None = the fitted value in the JSON
    # Append the following sentence when it opens with a pronoun or
    # demonstrative ("It is...", "This renewal...") — it continues the chosen
    # sentence's subject, and the answer is often in it.
    anaphoric_continuation: bool = False
    # The opposite direction: when the CHOSEN sentence itself opens with a
    # pronoun, demonstrative or "that day"-style reference ("It was founded in
    # 1802", "As of that day, ..."), prepend the sentence before it so the answer
    # does not start with a dangling reference. Applied after the refusal gate, so
    # the gate's decision is unchanged; the added sentence is verbatim corpus text.
    # On by default: sentences that open with a dangling reference fall from 6-12% to 0.4-4.6% across the four
    # evaluation sets, answer recall is equal or up (Text DEV +1.0, HeySQuAD DEV +0.4; no interval excludes 0).
    anaphoric_context: bool = True


@dataclass
class EngineConfig:
    # streaming | deferred (same pipeline, retrieval only after utterance end)
    # | baseline (single whole-utterance query, no decomposition/refinement)
    mode: str = "streaming"
    # streaming only: during speech, run the retrieval (+ claim selection)
    # that the controller's final pass WOULD issue if the utterance ended now
    # (RetrievalController.preview_final), and reuse it at utterance_end only
    # for an identical sub-query text - answers unchanged by construction.
    speculative_final: bool = True
    per_turn_timeout_ms: int = 15_000
    loop_lag_threshold_ms: int = 100


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "fake"))
    model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "fake-model"))
    # provider="local": base URL of an OpenAI-compatible server for a locally hosted open model (Ollama,
    # LM Studio, llama.cpp, vLLM, or tools/local_llm_server.py). No key, no cost.
    base_url: str = field(default_factory=lambda: _env_str("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key_env: str = "SECRET_LLM_API_KEY"
    temperature: float = 0.0
    seed: int = 7
    timeout_s: float = 10.0
    max_retries: int = 1
    # price table: model -> (usd per 1K input tokens, usd per 1K output tokens)
    price_table: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "fake-model": (0.0, 0.0),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gemini-1.5-flash": (0.000075, 0.0003),
    })


@dataclass
class TelemetryConfig:
    sink: str = "jsonl"           # jsonl | null
    out_path: str = "trace.jsonl"
    buffer_size: int = 50


@dataclass
class Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    time_scale: float = 1.0
    corpus_dir: str = "fixtures/dev_corpus"


def load_config(**overrides) -> Config:
    """Build a Config from defaults + environment, then apply explicit overrides."""
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
