"""Corpus-isolation / network check (Task 4 section 4.4 "network-restricted run"), run without a container.

Runs an official gate suite with every non-loopback network connection refused and RECORDED. The default
pipeline is extractive and local (pinned models are read from the local cache), so the expected result is:
the suite completes, every gate passes, and zero outbound connection attempts were made.

    python handoff/final_audit/netcheck.py [run_all arguments ...]
"""
import os
import runpy
import socket
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
attempts = []
_real_connect, _real_connect_ex = socket.socket.connect, socket.socket.connect_ex


def _is_local(addr):
    host = addr[0] if isinstance(addr, tuple) else str(addr)
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def _guard(real):
    def wrapper(self, addr, *a, **k):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(addr):
            attempts.append(repr(addr))
            raise OSError(f"network blocked by netcheck: {addr!r}")
        return real(self, addr, *a, **k)
    return wrapper


socket.socket.connect = _guard(_real_connect)
socket.socket.connect_ex = _guard(_real_connect_ex)
sys.argv = ["eval.run_all"] + (sys.argv[1:] or ["--scenarios", "eval/scenarios_real_corpus", "--corpus-dir",
                                                "data/corpus", "--time-scale", "8", "--reps", "1"])
sys.path.insert(0, os.getcwd())
code = 0
try:
    runpy.run_module("eval.run_all", run_name="__main__")
except SystemExit as e:
    code = e.code or 0
print(f"NETCHECK: outbound (non-loopback) connection attempts = {len(attempts)} {attempts[:5]}; exit code {code}")
sys.exit(0 if not attempts else 3)
