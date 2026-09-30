"""Generative (LLM-written) synthesis, ported from build A. Opt-in.

The default synthesizer (`session/synthesis.py`) is extractive: every claim is a
verbatim corpus sentence. That is measured, safe and cannot invent facts, but it
cannot merge information or rephrase. This package is the alternative: a
language model writes claims as JSON, and each claim is verified before it is
shipped (ID check against this turn's evidence, then a support check, with a
number-conflict guard). A late constraint is applied by a keep / amend /
retract decision per existing claim rather than by re-answering.

Select it with `synthesis.mode = "generative"` and a real LLM provider.

Status: unit- and integration-tested, but NOT yet measured against gold answers,
because no provider key was available. Treat any quality claim as unverified
until `eval.run_quality` has been run with `synthesis.mode=generative`.
"""

from .adapter import GenerativeSynthesizer

__all__ = ["GenerativeSynthesizer"]
