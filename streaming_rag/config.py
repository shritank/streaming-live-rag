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
    # Tuned against both fixtures/dev_corpus and the real SQuAD corpus (see
    # docs/benchmark_report.md §3/§6). The LSA encoder is a weak learner at
    # this corpus scale — a heavier dense weight measurably hurts real-corpus
    # r@5 (83.8% at 0.15 vs 86.8% at 0.01) with no offsetting recall gain on
    # dev_corpus's paraphrase queries (identical at both settings, since RRF's
    # rank-based fusion still lets dense's top hits register even at a small
    # weight). Kept non-zero rather than sparse-only so a stronger encoder
    # swapped into the Embedder protocol has a fusion weight to grow into.
    dense_weight: float = 0.01
    sparse_weight: float = 0.99
    low_confidence_threshold: float = 0.15
    # Adds an artificial delay to every search call, representing a realistic
    # hosted vector-DB / reranker round trip. 0 by default (the dev corpus is
    # small enough that real search is sub-millisecond); the baseline vs
    # streaming comparison sets this explicitly, because "the user waits
    # while retrieval happens after they stop talking" (the guide's problem
    # statement) only shows up in the numbers once retrieval isn't free.
    simulated_latency_ms: float = 0.0


@dataclass
class ControllerConfig:
    mode: str = "rule"            # rule | model | hybrid
    min_stability: float = 0.6
    provisional_stability: float = 0.45
    debounce_ms: int = 150


@dataclass
class SynthesisConfig:
    grounding_check: bool = True
    max_answer_tokens: int = 512


@dataclass
class EngineConfig:
    mode: str = "streaming"       # streaming | baseline
    per_turn_timeout_ms: int = 15_000
    loop_lag_threshold_ms: int = 100


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "fake"))
    model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "fake-model"))
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
