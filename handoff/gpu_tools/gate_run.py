"""Run an eval module with an alternative refusal-gate JSON (experiment only).

    python gate_run.py GATE.json eval.run_quality --scenarios ... [args]
"""
import json
import runpy
import sys

from streaming_rag.session import synthesis

synthesis._REFUSAL_MODEL = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("gate:", synthesis._REFUSAL_MODEL["features"], "threshold", synthesis._REFUSAL_MODEL["threshold"], flush=True)
module = sys.argv[2]
sys.argv = [module] + sys.argv[3:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
