"""Component factory: wires the real Task 1-3 implementations (or the
mocks) into a set of objects the Engine can run. Centralised here so the
CLI and the eval harness never construct components differently.
"""
from __future__ import annotations

from .config import Config
from .contracts import NullTelemetry, Telemetry
from .controller import RetrievalController
from .llm import build_llm_client
from .mocks import MockController, MockRetriever, MockSynthesizer
from .retrieval import HybridRetriever
from .session.store import SessionStore
from .session.synthesis import GroundedSynthesizer


def build_real_components(config: Config, telemetry: Telemetry | None = None,
                           session_store: SessionStore | None = None):
    telemetry = telemetry or NullTelemetry()
    session_store = session_store or SessionStore()
    llm = build_llm_client(config, telemetry)
    retriever = HybridRetriever(config, telemetry=telemetry)
    controller = RetrievalController(llm=llm, telemetry=telemetry, config=config,
                                      corpus_vocab=lambda: retriever.specific_vocabulary)
    synthesizer = GroundedSynthesizer(retriever, session_store, config, llm=llm, telemetry=telemetry)
    return controller, retriever, synthesizer, llm, session_store


def build_mock_components(config: Config, telemetry: Telemetry | None = None,
                           session_store: SessionStore | None = None, fail_mode: str = "none"):
    telemetry = telemetry or NullTelemetry()
    session_store = session_store or SessionStore()
    llm = build_llm_client(config, telemetry)
    controller = MockController(llm=llm, telemetry=telemetry, config={}, fail_mode=fail_mode)
    retriever = MockRetriever(latency_ms=50.0, low_confidence_threshold=config.retrieval.low_confidence_threshold,
                               fail_mode=fail_mode)
    synthesizer = MockSynthesizer(llm=llm, telemetry=telemetry, fail_mode=fail_mode)
    return controller, retriever, synthesizer, llm, session_store


def build_components(config: Config, telemetry: Telemetry | None = None,
                      session_store: SessionStore | None = None, impl: str = "real"):
    if impl == "real":
        return build_real_components(config, telemetry, session_store)
    if impl == "mock":
        return build_mock_components(config, telemetry, session_store)
    raise ValueError(f"unknown impl: {impl!r}")
