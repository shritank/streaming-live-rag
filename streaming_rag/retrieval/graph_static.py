"""Exact, verified removal of run-time shape arithmetic from the pinned
cross-encoder / SQuAD2-reader ONNX exports, so that nothing of theirs runs on
the CPU execution provider (ONNX Runtime keeps int64 shape math on the CPU).

The exports compute two things from tensor shapes on every call:
  * the attention scale  Sqrt(Cast(Slice(Shape(q))))  = sqrt(head_size);
  * the head split/merge Reshape targets  Concat(batch, seq, 12, 32) and
    Concat(batch, seq, 384), built from Shape -> Gather -> Unsqueeze.

Each such value is read by RUNNING the model at two different (batch, seq)
shapes. A float value that is identical at both shapes becomes a constant; a
Reshape target equal to [batch, seq, c...] at both shapes becomes the
constant [0, 0, c...] (ONNX Reshape: 0 = copy that input dimension, and the
data input of every such Reshape has batch, seq as its leading dims - the
equality check at two shapes enforces it). Nothing else is touched. The
rewritten graph must reproduce the original outputs BIT-FOR-BIT at several
shapes (reference execution, ORT graph optimisations off) or `staticize`
raises.
"""
from __future__ import annotations

import copy

import numpy as np


def _feeds(model, batch: int, seq: int) -> dict:
    rng = np.random.default_rng(batch * 1000 + seq)
    out = {}
    for i in model.graph.input:
        if i.name == "input_ids":
            out[i.name] = rng.integers(1000, 20000, (batch, seq)).astype(np.int64)
        elif i.name == "attention_mask":
            m = np.ones((batch, seq), np.int64)
            m[0, seq // 2:] = 0          # exercise padding
            out[i.name] = m
        elif i.name == "token_type_ids":
            t = np.zeros((batch, seq), np.int64)
            t[:, seq // 3:] = 1
            out[i.name] = t
    return out


def _run(model, feeds, extra: list[tuple[str, int]] = ()):
    """Reference CPU execution with ONNX Runtime's own graph optimisations
    OFF, so a comparison tests this rewrite alone (ORT's fusions may round a
    rewritten graph differently in the last float bit, ~1e-6)."""
    import onnx
    import onnxruntime as ort
    m = copy.deepcopy(model)
    for name, elem in extra:
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, elem, None))
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), sess_options=o, providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))


def staticize(model_bytes: bytes):
    """Return (rewritten onnx.ModelProto, report dict)."""
    import onnx
    from onnx import TensorProto, numpy_helper
    model = onnx.load_from_string(model_bytes)
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}

    def upstream_ops(t, depth=6):
        n = prod.get(t)
        if n is None or depth == 0:
            return set()
        ops = {n.op_type}
        for i in n.input:
            ops |= upstream_ops(i, depth - 1)
        return ops

    sqrts = [n.output[0] for n in g.node if n.op_type == "Sqrt" and "Shape" in upstream_ops(n.input[0])]
    reshapes = [n for n in g.node if n.op_type == "Reshape" and n.input[1] in prod
                and prod[n.input[1]].op_type == "Concat"]
    extra = ([(t, TensorProto.FLOAT) for t in sqrts] + [(n.input[1], TensorProto.INT64) for n in reshapes]
             + [(n.input[0], TensorProto.FLOAT) for n in reshapes])
    shapes = ((2, 37), (3, 61))
    vals = [_run(model, _feeds(model, b, s), extra) for b, s in shapes]

    consts, report = {}, {"sqrt_constants": 0, "reshape_targets": 0}
    for t in sqrts:
        a, b = vals[0][t], vals[1][t]
        if a.shape == b.shape and np.array_equal(a, b):
            consts[t] = a
            report["sqrt_constants"] += 1
    for n in reshapes:
        t, data = n.input[1], n.input[0]
        # target is [batch, seq, c...] AND the data being reshaped really has
        # (batch, seq) as its leading dims (so "0 = copy" copies the same value)
        ok = all(len(v[t]) >= 3 and v[t][0] == bs and v[t][1] == sq
                 and v[data].ndim >= 2 and v[data].shape[:2] == (bs, sq)
                 for v, (bs, sq) in zip(vals, shapes)) and np.array_equal(vals[0][t][2:], vals[1][t][2:])
        if ok:
            consts[t] = np.concatenate([[0, 0], vals[0][t][2:]]).astype(np.int64)
            report["reshape_targets"] += 1

    new = copy.deepcopy(model)
    ng = new.graph
    for t, v in consts.items():
        ng.initializer.append(numpy_helper.from_array(v, name=t))
    # drop the producers of the now-constant tensors, then everything dead
    keep = [n for n in ng.node if not (set(n.output) & set(consts))]
    needed = {o.name for o in ng.output}
    alive = []
    for n in reversed(keep):
        if set(n.output) & needed:
            alive.append(n)
            needed |= set(n.input)
    del ng.node[:]
    ng.node.extend(reversed(alive))
    report["nodes_before"], report["nodes_after"] = len(g.node), len(ng.node)

    for b, s in ((1, 8), (2, 37), (5, 192), (3, 384)):
        f = _feeds(model, b, s)
        ref, got = _run(model, f), _run(new, f)
        for k in ref:
            if not np.array_equal(ref[k], got[k]):
                raise RuntimeError(f"staticize changed output {k} at batch {b} seq {s}")
    report["bit_identical_shapes"] = [[1, 8], [2, 37], [5, 192], [3, 384]]
    return new, report
