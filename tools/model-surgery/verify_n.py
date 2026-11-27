"""Add a longer verify signature to a Gemma 4 .litertlm (Agello PERFORMANCE.md, B5).

The export's `verify` signature checks 4 positions (the input token and 3 drafts). This adds
`verify_<N>` beside it: a copy of the verify subgraph and of its composites' decompositions, with
every number that follows the length rewritten for N positions. The weights are shared (the copy
references the same buffers), so only the graph grows.

How the length is found:
- In the main graph, each dimension is traced from the signature's inputs as an ordered product of
  factors, each `a*T + b` (T the length): the embeddings, positions, mask and per-layer embeddings
  carry T, everything else starts constant. Every op's result is checked against the shapes the
  file stores for T = 4; a mismatch or a case the rules don't cover stops the script.
- A composite's decomposition is matched with the same composite in `decode` (T = 1) and
  `prefill_128` (T = 128): every number in it is solved as `a*T + b` from T = 1 and 4 and checked
  at 128 (the local attention's window slice, 512 + T - 1, is one).

A decomposition may disagree with its composite on a size that doesn't follow the length (ring_kv.py
resizes the local caches in the main graph only): there the traced size is written, and counted.

Checked with check_verify_n.py: on the published E4B file, verify_8 equals verify chained over
positions 0-3 and 4-7, bit for bit. On the 128K file (2,832 ops) it writes as well; the GPU checks it.

Usage (tools venv, ~/.venvs/agello-tools):
  python verify_n.py <in.litertlm> --report               # what follows the length, no output
  python verify_n.py <in.litertlm> <out.litertlm> [--positions 8]
"""

import os
import struct
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers
from litert_lm_builder import litertlm_header_schema_py_generated as H

OPS = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
SECTION = "tf_lite_prefill_decode"
ALIGN = 16384
T_FROM = 4  # the export's verify length


class Stop(Exception):
    pass


# ---------- dimensions: ordered products of factors a*T + b ----------

def const_dim(c):
    return () if int(c) == 1 else ((0, int(c)),)


def t_dim(a=1, b=0):
    return ((a, b),)


def value(dim, t):
    v = 1
    for a, b in dim:
        v *= a * t + b
    return v


def has_t(dim):
    return any(a for a, _ in dim)


def merge(dims):
    out = []
    for d in dims:
        out.extend(d)
    return tuple(f for f in out if f != (0, 1))


def split(dim, targets, where):
    """Splits a dimension's factors, in order, into consecutive groups worth `targets` at T = 4.
    Constant factors may be divided; a factor with T may not."""
    factors = list(dim)
    groups = []
    for target in targets:
        group, worth = [], 1
        while worth < target:
            if not factors:
                raise Stop(f"{where}: cannot split {dim} into {targets}")
            a, b = factors[0]
            v = a * T_FROM + b
            if worth * v <= target and target % (worth * v) == 0:
                group.append(factors.pop(0))
                worth *= v
            elif a == 0 and v % (target // worth) == 0:
                need = target // worth
                factors[0] = (0, v // need)
                group.append((0, need))
                worth *= need
            else:
                raise Stop(f"{where}: cannot split {dim} into {targets}")
        if worth != target:
            raise Stop(f"{where}: cannot split {dim} into {targets}")
        groups.append(tuple(f for f in group if f != (0, 1)))
    if any(f != (0, 1) for f in factors):
        raise Stop(f"{where}: {dim} has factors left after {targets}")
    return groups


def reshape(dims, out_shape, where):
    """Maps input dims onto the stored output shape (T = 4) by matching prefix products."""
    out_shape = [int(x) for x in out_shape]
    flat = merge(dims)
    result, i = [], 0
    # Walk output dims, taking input factors until each output's value is reached.
    factors = list(flat)
    for target in out_shape:
        if target == 1:
            result.append(())
            continue
        group, worth = [], 1
        while worth < target:
            if not factors:
                raise Stop(f"{where}: reshape {dims} to {out_shape}")
            a, b = factors[0]
            v = a * T_FROM + b
            if target % (worth * v) == 0:
                group.append(factors.pop(0))
                worth *= v
            elif a == 0 and v % (target // worth) == 0:
                need = target // worth
                factors[0] = (0, v // need)
                group.append((0, need))
                worth *= need
            else:
                raise Stop(f"{where}: reshape {dims} to {out_shape}: factor {factors[0]} doesn't fit")
        result.append(tuple(f for f in group if f != (0, 1)))
    if any(f != (0, 1) for f in factors):
        raise Stop(f"{where}: reshape {dims} to {out_shape}: factors left {factors}")
    return result


def broadcast(shapes, where):
    rank = max(len(s) for s in shapes)
    out = []
    for axis in range(rank):
        chosen = ()
        for s in shapes:
            j = axis - (rank - len(s))
            if j < 0:
                continue
            d = s[j]
            if value(d, T_FROM) == 1:
                continue
            if value(chosen, T_FROM) == 1:
                chosen = d
            elif chosen != d and (has_t(chosen) or has_t(d)):
                raise Stop(f"{where}: broadcast of {chosen} and {d} on axis {axis}")
        out.append(chosen)
    return out


# ---------- the file ----------

def read_header(path):
    with open(path, "rb") as f:
        header_end = struct.unpack("<Q", f.read(32)[24:32])[0]
        f.seek(0)
        header = f.read(header_end)
    md = H.LiteRTLMMetaData.GetRootAs(header, 32)
    sections = []
    for i in range(md.SectionMetadata().ObjectsLength()):
        o = md.SectionMetadata().Objects(i)
        kind = None
        for j in range(o.ItemsLength()):
            kv = o.Items(j)
            if kv.Key().decode() == "model_type":
                v = H.StringValue()
                v.Init(kv.Value().Bytes, kv.Value().Pos)
                kind = v.Value().decode()
        sections.append((o.BeginOffset(), o.EndOffset(), kind))
    return header_end, sections


def validate_sections(header_end, sections, total):
    """Verify container sections are within [header_end, total] and non-overlapping."""
    cursor = header_end
    for i in sorted(range(len(sections)), key=lambda idx: sections[idx][0]):
        b, e, k = sections[i]
        if b < cursor or e < b or e > total:
            raise Stop(
                f"invalid or overlapping section {i} ({k}): [{b}, {e}) after {cursor} (file size {total})"
            )
        cursor = e


def load(path):
    header_end, sections = read_header(path)
    validate_sections(header_end, sections, os.path.getsize(path))
    index, (begin, end, _) = next((i, s) for i, s in enumerate(sections) if s[2] == SECTION)
    with open(path, "rb") as f:
        f.seek(begin)
        data = f.read(end - begin)
    model = S.Model.GetRootAs(data, 0)
    external = [model.Buffers(i).Offset() for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1]
    graph_end = min(external) if external else len(data)
    mt = S.ModelT.InitFromObj(S.Model.GetRootAs(data[:graph_end] if external else data, 0))
    return header_end, sections, index, data, graph_end, external, mt


class Graph:
    def __init__(self, mt, data):
        self.mt, self.data = mt, data

    def code(self, op):
        c = self.mt.operatorCodes[op.opcodeIndex]
        return OPS.get(max(c.builtinCode, c.deprecatedBuiltinCode))

    def raw(self, tensor):
        b = self.mt.buffers[tensor.buffer]
        if b.data is not None and len(b.data):
            return bytes(b.data)
        if b.offset > 1:
            return self.data[b.offset:b.offset + b.size]
        return None

    def ints(self, tensor):
        blob = self.raw(tensor)
        if blob is None:
            return None
        dtype = {S.TensorType.INT32: np.int32, S.TensorType.INT64: np.int64}.get(tensor.type)
        if dtype is None:
            return None
        return np.frombuffer(blob, dtype=dtype).astype(np.int64)


def shape_of(t):
    return [int(x) for x in t.shape] if t.shape is not None else []


# ---------- the main graph's trace ----------

UNARY = {"QUANTIZE", "DEQUANTIZE", "CAST", "GELU", "TANH", "SIN", "COS", "RSQRT", "SOFTMAX", "LOGISTIC",
         "NEG", "ABS", "SQRT", "EXP", "LOG", "FLOOR"}
BINARY = {"MUL", "ADD", "SUB", "DIV", "MAXIMUM", "MINIMUM", "NOT_EQUAL", "EQUAL", "GREATER", "GREATER_EQUAL",
          "LESS", "LESS_EQUAL", "LOGICAL_AND", "LOGICAL_OR", "SELECT_V2", "POW", "FLOOR_MOD", "FLOOR_DIV",
          "SQUARED_DIFFERENCE"}

# Composites whose decompositions are solved from `decode` (they line up op for op); the others'
# decompositions are traced.
SOLVED_KINDS = ("odml.runtime_bmm",)

# Signature inputs that carry the length, and on which axis.
T_INPUTS = {"embeddings": 1, "input_pos": 0, "mask": 2, "per_layer_embeddings": 1}


def signature_seeds(sub, signature):
    """The signature's inputs, the length on the axes `T_INPUTS` names."""
    seeds = {}
    for io in signature.inputs:
        name = io.name.decode() if isinstance(io.name, bytes) else io.name
        t = int(io.tensorIndex)
        shape = shape_of(sub.tensors[t])
        d = [const_dim(x) for x in shape]
        if name in T_INPUTS:
            axis = T_INPUTS[name]
            if shape[axis] != T_FROM:
                raise Stop(f"input {name} {shape}: axis {axis} isn't {T_FROM}")
            d[axis] = t_dim()
        seeds[t] = d
    return seeds


LENIENT = {"count": 0}


def trace(graph, sub, seeds, decompose, strict=True):
    """Traces a subgraph from `seeds` ({tensor: dims}). Returns {tensor: dims} and the edits to make
    at another length: ('shape_const', tensor, dims), ('reshape_option', op, dims),
    ('slice_size', tensor, sizes). `decompose(subgraph, input dims, where)` traces a composite's
    decomposition and returns its outputs' dims."""
    dims = dict(seeds)
    edits = []
    tensors = sub.tensors
    produced = {int(x) for op in sub.operators for x in op.outputs}

    def get(t):
        if t in dims:
            return dims[t]
        if t in produced:
            raise Stop(f"tensor {t} used before it's produced")
        d = [const_dim(x) for x in shape_of(tensors[t])]
        dims[t] = d
        return d

    def check(t, d, where):
        stored = shape_of(tensors[t])
        got = [value(x, T_FROM) for x in d]
        if got != stored:
            # A decomposition may disagree with its composite on a size that doesn't follow the
            # length (ring_kv.py resizes the local caches in the main graph only; the GPU never
            # runs decompositions): the traced size wins there.
            if strict or len(got) != len(stored) or any(
                    g != s and has_t(x) for g, s, x in zip(got, stored, d, strict=True)):
                raise Stop(f"{where}: traced {got} but the file has {stored}")
            LENIENT["count"] += 1
        dims[t] = d

    for k, op in enumerate(sub.operators):
        kind = graph.code(op)
        ins = [int(x) for x in op.inputs]
        outs = [int(x) for x in op.outputs]
        where = f"op {k} {kind}"
        if kind in UNARY:
            check(outs[0], list(get(ins[0])), where)
        elif kind in BINARY:
            check(outs[0], broadcast([get(t) for t in ins if t >= 0], where), where)
        elif kind == "FULLY_CONNECTED":
            x = get(ins[0])
            n = shape_of(tensors[ins[1]])[0]
            keep = op.builtinOptions is not None and op.builtinOptions.keepNumDims
            out = list(x[:-1]) + [const_dim(n)] if keep else [merge(x[:-1]), const_dim(n)]
            check(outs[0], out, where)
        elif kind == "RESHAPE":
            out = reshape(get(ins[0]), shape_of(tensors[outs[0]]), where)
            check(outs[0], out, where)
            if len(ins) > 1 and ins[1] >= 0 and any(has_t(d) for d in out):
                edits.append(("shape_const", ins[1], out))
            if any(has_t(d) for d in out):
                edits.append(("reshape_option", k, out))
        elif kind == "TRANSPOSE":
            perm = graph.ints(tensors[ins[1]])
            x = get(ins[0])
            check(outs[0], [x[int(p)] for p in perm], where)
        elif kind == "SLICE":
            x = get(ins[0])
            begin = graph.ints(tensors[ins[1]])
            size = graph.ints(tensors[ins[2]])
            if begin is None or size is None:
                raise Stop(f"{where}: begin or size isn't constant")
            out, new_size, new_begin = [], [], []
            for axis, d in enumerate(x):
                full = value(d, T_FROM)
                s = full - begin[axis] if size[axis] == -1 else int(size[axis])
                if has_t(d) and begin[axis] == 0 and s == full:
                    out.append(d)
                    new_size.append(None if size[axis] == -1 else d)
                elif has_t(d) and s < full:
                    # A part of the length axis: only the first rows, a constant count.
                    out.append(const_dim(s))
                    new_size.append(int(size[axis]))
                    if begin[axis] != 0:
                        raise Stop(f"{where}: slice of the length axis from {begin[axis]}")
                else:
                    out.append(const_dim(s))
                    new_size.append(int(size[axis]))
                new_begin.append(int(begin[axis]))
            check(outs[0], out, where)
            if any(isinstance(v, tuple) for v in new_size):
                edits.append(("slice_size", ins[2], new_size))
        elif kind == "CONCATENATION":
            axis = op.builtinOptions.axis
            parts = [get(t) for t in ins]
            rank = len(parts[0])
            axis %= rank
            total_a = total_b = 0
            for p in parts:
                d = p[axis]
                if len(d) > 1:
                    raise Stop(f"{where}: concatenating a product {d}")
                a, b = d[0] if d else (0, 1)
                total_a += a
                total_b += b
            out = list(parts[0])
            out[axis] = ((total_a, total_b),) if (total_a, total_b) != (0, 1) else ()
            for p in parts[1:]:
                for j in range(rank):
                    if j != axis and has_t(p[j]) != has_t(out[j]):
                        raise Stop(f"{where}: inputs differ on axis {j}")
            check(outs[0], out, where)
        elif kind in ("REDUCE_MAX", "SUM", "MEAN", "REDUCE_MIN", "REDUCE_PROD"):
            x = get(ins[0])
            axes = {int(a) % len(x) for a in graph.ints(tensors[ins[1]])}
            keep = op.builtinOptions is not None and op.builtinOptions.keepDims
            out = [(() if j in axes else d) for j, d in enumerate(x)] if keep else \
                [d for j, d in enumerate(x) if j not in axes]
            check(outs[0], out, where)
        elif kind == "UNPACK":
            x = get(ins[0])
            axis = op.builtinOptions.axis % len(x)
            if has_t(x[axis]):
                raise Stop(f"{where}: unpacking the length axis")
            out = [d for j, d in enumerate(x) if j != axis]
            for t in outs:
                check(t, list(out), where)
        elif kind == "PACK":
            x = get(ins[0])
            axis = op.builtinOptions.axis
            out = list(x)
            out.insert(axis % (len(x) + 1), const_dim(len(ins)))
            check(outs[0], out, where)
        elif kind == "GATHER":
            params, indices = get(ins[0]), get(ins[1])
            axis = op.builtinOptions.axis % len(params)
            batch = op.builtinOptions.batchDims
            if has_t(params[axis]):
                raise Stop(f"{where}: gathering along the length axis")
            check(outs[0], list(params[:axis]) + list(indices[batch:]) + list(params[axis + 1:]), where)
        elif kind == "DYNAMIC_UPDATE_SLICE":
            update = get(ins[1])
            operand = get(ins[0])
            for j, d in enumerate(update):
                if has_t(d) and has_t(operand[j]) and d != operand[j]:
                    raise Stop(f"{where}: an update of the length axis smaller than it")
            check(outs[0], list(operand), where)
        elif kind == "STABLEHLO_COMPOSITE":
            name = op.builtinOptions2.name.decode()
            if name in ("odml.rms_norm", "odml.cache_update"):
                out = decompose(op.builtinOptions2.decompositionSubgraphIndex, [get(t) for t in ins], where)
                if len(outs) != len(out):
                    raise Stop(f"{where} {name}: expected {len(outs)} outputs, decomposition returned {len(out)}")
                for t, d in zip(outs, out, strict=True):
                    check(t, list(d), where + " " + name)
            elif name == "odml.runtime_bmm":
                q, kv = get(ins[0]), get(ins[1])
                last = shape_of(tensors[outs[0]])[-1]
                pick = next((d for d in kv[-2:] if value(d, T_FROM) == last), None)
                if pick is None or has_t(pick):
                    raise Stop(f"{where} runtime_bmm: no constant last dim {last} in {kv}")
                check(outs[0], list(q[:-1]) + [pick], where + " runtime_bmm")
            else:
                raise Stop(f"{where}: composite {name} has no rule")
        else:
            raise Stop(f"{where}: no rule")
    return dims, edits


def composites(graph, sub):
    """The composite ops of a subgraph in order: (op index, name, decomposition index)."""
    out = []
    for k, op in enumerate(sub.operators):
        if graph.code(op) == "STABLEHLO_COMPOSITE":
            o = op.builtinOptions2
            out.append((k, o.name.decode(), o.decompositionSubgraphIndex))
    return out


def solve(d, v, p, where):
    """a*T + b from T = 1 and 4, checked at T = 128 when `p` is known."""
    if d == v and (p is None or p == d):
        return (0, d)
    if (v - d) % 3:
        raise Stop(f"{where}: {d}, {v}, {p} aren't linear in the length")
    a = (v - d) // 3
    b = d - a
    if p is not None and a * 128 + b != p:
        raise Stop(f"{where}: {d}, {v}, {p} aren't linear in the length")
    return (a, b)


def int_constants(graph, sub):
    """Small integer constants of a subgraph: {tensor: values}."""
    produced = {int(x) for op in sub.operators for x in op.outputs}
    out = {}
    for t, tensor in enumerate(sub.tensors):
        if t in produced or tensor.type not in (S.TensorType.INT32, S.TensorType.INT64):
            continue
        values = graph.ints(tensor)
        if values is not None and values.size <= 16:
            out[t] = values
    return out


def solve_decompositions(graph, mt):
    """For each of verify's composites, the solved numbers of its decomposition and attributes:
    from the same composite in `decode` (T = 1), checked against `prefill_128` (T = 128) for the
    kinds whose composites line up there (it skips some per-position work)."""
    by_key = {s.signatureKey.decode(): s.subgraphIndex for s in mt.signatureDefs}
    lists = {k: composites(graph, mt.subgraphs[by_key[k]]) for k in ("decode", "verify", "prefill_128")}
    if [n for _, n, _ in lists["decode"]] != [n for _, n, _ in lists["verify"]]:
        raise Stop("decode's and verify's composites differ")
    per_kind = {}
    for kind in {n for _, n, _ in lists["verify"]}:
        mine = [c for c in lists["verify"] if c[1] == kind]
        theirs = [c for c in lists["prefill_128"] if c[1] == kind]
        per_kind[kind] = dict(zip([c[0] for c in mine], theirs, strict=True)) if len(mine) == len(theirs) else {}
    solved = []
    for (kd, nd, gd), (kv, nv, gv) in zip(lists["decode"], lists["verify"], strict=True):
        if nv not in SOLVED_KINDS:
            continue
        where = f"composite {nv} (verify op {kv})"
        check = per_kind[nv].get(kv)
        kp, gp = (check[0], check[2]) if check else (None, None)
        subs = [mt.subgraphs[gd], mt.subgraphs[gv]] + ([mt.subgraphs[gp]] if gp is not None else [])
        codes = [[graph.code(o) for o in x.operators] for x in subs]
        if any(c != codes[1] for c in codes) or any(len(x.tensors) != len(subs[1].tensors) for x in subs):
            raise Stop(f"{where}: decompositions differ across exports")
        third = (lambda f: f(subs[2])) if gp is not None else (lambda f: None)
        shapes = {}
        for t in range(len(subs[1].tensors)):
            sd, sv = shape_of(subs[0].tensors[t]), shape_of(subs[1].tensors[t])
            sp = third(lambda x: shape_of(x.tensors[t]))
            if len(sd) != len(sv) or (sp is not None and len(sp) != len(sv)):
                raise Stop(f"{where}: tensor {t}'s rank differs")
            shapes[t] = [solve(a, b, None if sp is None else sp[j], f"{where} tensor {t}")
                         for j, (a, b) in enumerate(zip(sd, sv, strict=True))]
        consts = {}
        cd, cv = int_constants(graph, subs[0]), int_constants(graph, subs[1])
        cp = int_constants(graph, subs[2]) if gp is not None else None
        for t, values in cv.items():
            if t not in cd or len(cd[t]) != len(values) or (cp is not None and (t not in cp or len(cp[t]) != len(values))):
                raise Stop(f"{where}: constant {t} differs in size across exports")
            line = [solve(int(a), int(b), None if cp is None else int(cp[t][j]), f"{where} constant {t}")
                    for j, (a, b) in enumerate(zip(cd[t], values, strict=True))]
            if any(a for a, _ in line):
                consts[t] = line
        options = {}
        for i, ops in enumerate(zip(*[x.operators for x in subs], strict=True)):
            if graph.code(ops[1]) == "RESHAPE" and ops[1].builtinOptions is not None and \
                    ops[1].builtinOptions.newShape is not None:
                shapes_n = [list(o.builtinOptions.newShape) for o in ops]
                if any(len(s) != len(shapes_n[1]) for s in shapes_n):
                    raise Stop(f"{where}: reshape {i} newShape rank differs across exports")
                line = [solve(int(shapes_n[0][j]), int(shapes_n[1][j]),
                              int(shapes_n[2][j]) if len(shapes_n) > 2 else None, f"{where} reshape {i}")
                        for j in range(len(shapes_n[1]))]
                if any(a for a, _ in line):
                    options[i] = line
        attrs = {}
        refs = [("decode", kd), ("verify", kv)] + ([("prefill_128", kp)] if kp is not None else [])
        loaded = [flexbuffers.Loads(bytes(mt.subgraphs[by_key[k]].operators[op].builtinOptions2.compositeAttributes))
                  for k, op in refs]
        for key, val in loaded[1].items():
            if isinstance(val, int) and not isinstance(val, bool):
                line = solve(int(loaded[0][key]), int(val), int(loaded[2][key]) if len(loaded) > 2 else None,
                             f"{where} attribute {key}")
                if line[0]:
                    attrs[key] = line
        solved.append({"op": kv, "decomposition": gv, "shapes": shapes, "consts": consts,
                       "options": options, "attrs": attrs})
    return solved


def analyze(path):
    """Everything to change for another length: the main graph's trace, the traced decompositions,
    the solved ones."""
    header_end, sections, index, data, graph_end, external, mt = load(path)
    graph = Graph(mt, data)
    signature = next(s for s in mt.signatureDefs if s.signatureKey.decode() == "verify")
    sub = mt.subgraphs[signature.subgraphIndex]
    traced = {}

    def decompose(index, in_dims, where):
        d_sub = mt.subgraphs[index]
        if len(in_dims) != len(d_sub.inputs):
            raise Stop(
                f"{where}: decomposition subgraph {index} expects {len(d_sub.inputs)} inputs, got {len(in_dims)}"
            )
        seeds = {int(t): d for t, d in zip(d_sub.inputs, in_dims, strict=True)}
        d_dims, d_edits = trace(graph, d_sub, seeds, decompose, strict=False)
        if index in traced and traced[index] != (d_dims, d_edits):
            raise Stop(f"{where}: shared decomposition subgraph {index} traced with conflicting dims/edits")
        traced[index] = (d_dims, d_edits)
        return [d_dims[int(t)] for t in d_sub.outputs]

    dims, edits = trace(graph, sub, signature_seeds(sub, signature), decompose)
    solved = solve_decompositions(graph, mt)
    return {"load": (header_end, sections, index, data, graph_end, external, mt), "graph": graph,
            "signature": signature, "dims": dims, "edits": edits, "traced": traced, "solved": solved}


def report(path):
    try:
        a = analyze(path)
    except Stop as e:
        raise SystemExit(f"stopped: {e}")
    sub = a["load"][6].subgraphs[a["signature"].subgraphIndex]
    with_t = sum(1 for d in a["dims"].values() if any(has_t(x) for x in d))
    print(f"verify: {len(sub.operators)} ops traced, {with_t} tensors follow the length")
    kinds = {}
    for e in a["edits"] + [e for _, ed in a["traced"].values() for e in ed]:
        kinds[e[0]] = kinds.get(e[0], 0) + 1
    print(f"decompositions traced: {len(a['traced'])}; solved from decode: {len(a['solved'])}; "
          f"{LENIENT['count']} decomposition shapes taken from the trace over the file's")
    print("edits:", kinds)
    s = a["solved"]
    print(f"solved: {sum(1 for c in s for line in c['shapes'].values() if any(x for x, _ in line))} tensors, "
          f"{sum(len(c['consts']) for c in s)} constants, {sum(len(c['options']) for c in s)} reshape options, "
          f"{sum(len(c['attrs']) for c in s)} attributes follow the length")
    window = sorted({tuple(line) for c in s for line in c["shapes"].values() if any(x and y > 1 for x, y in line)})
    print("  dims with a constant part (the window):", window[:3])
    outputs = {(io.name.decode() if isinstance(io.name, bytes) else io.name): int(io.tensorIndex)
               for io in a["signature"].outputs}
    for name in ("logits", "activations"):
        print(f"  {name} at 8 positions: {[value(x, 8) for x in a['dims'][outputs[name]]]}")


def _aligned_buffer_pack(self, builder):
    data = None
    if self.data is not None and len(self.data):
        array = np.ascontiguousarray(np.asarray(self.data, dtype=np.uint8))
        builder.StartVector(1, array.size, 16)
        builder.head = builder.head - array.size
        builder.Bytes[builder.head:builder.head + array.size] = array.tobytes()
        data = builder.EndVector()
    S.BufferStart(builder)
    if data is not None:
        S.BufferAddData(builder, data)
    S.BufferAddOffset(builder, self.offset)
    S.BufferAddSize(builder, self.size)
    return S.BufferEnd(builder)


S.BufferT.Pack = _aligned_buffer_pack


def clone_subgraph(mt, graph, source, name, dims_of, n, edits, consts=None):
    """A copy of subgraph `source` for length `n`: shapes from `dims_of(tensor)` (None keeps the
    stored shape), `edits` applied, `consts` ({tensor: [(a, b), ...]}) rebuilt. Returns the copy."""
    import copy
    sub = copy.deepcopy(mt.subgraphs[source])
    sub.name = name.encode()
    for t, tensor in enumerate(sub.tensors):
        d = dims_of(t)
        if d is None:
            continue
        new = np.array([value(x, n) for x in d], dtype=np.int32)
        if tensor.shapeSignature is not None and len(tensor.shapeSignature) == len(new):
            tensor.shapeSignature = np.array(
                [-1 if s == -1 else v for s, v in zip(tensor.shapeSignature, new, strict=True)],
                dtype=np.int32,
            )
        tensor.shape = new
    wanted = {}

    def want(t, values, where):
        values = [int(v) for v in values]
        if t in wanted and wanted[t] != values:
            raise Stop(f"{where}: constant {t} wanted as {wanted[t]} and {values}")
        wanted[t] = values

    for e in edits:
        if e[0] == "shape_const":
            _, t, out = e
            old = graph.ints(mt.subgraphs[source].tensors[t])
            if old is None:
                raise Stop(f"{name}: shape constant tensor {t} has no constant integer buffer")
            if len(old) != len(out):
                raise Stop(f"{name}: shape constant tensor {t} length {len(old)} != traced rank {len(out)}")
            want(t, [-1 if o == -1 else value(d, n) for o, d in zip(old, out, strict=True)], name)
        elif e[0] == "reshape_option":
            _, k, out = e
            options = sub.operators[k].builtinOptions
            if options is not None and options.newShape is not None and len(options.newShape):
                if len(options.newShape) != len(out):
                    raise Stop(
                        f"{name}: reshape op {k} newShape length {len(options.newShape)} != traced rank {len(out)}"
                    )
                options.newShape = [
                    -1 if o == -1 else value(d, n) for o, d in zip(options.newShape, out, strict=True)
                ]
        elif e[0] == "slice_size":
            _, t, sizes = e
            want(t, [-1 if v is None else (value(v, n) if isinstance(v, tuple) else v) for v in sizes], name)
    for t, line in (consts or {}).items():
        want(t, [a * n + b for a, b in line], name)
    for t, values in wanted.items():
        tensor = sub.tensors[t]
        dtype = np.int64 if tensor.type == S.TensorType.INT64 else np.int32
        b = S.BufferT()
        b.data = np.frombuffer(np.array(values, dtype=dtype).tobytes(), dtype=np.uint8)
        mt.buffers.append(b)
        tensor.buffer = len(mt.buffers) - 1
    return sub


def write(path, out_path, n):
    a = analyze(path)
    header_end, sections, index, data, graph_end, external, mt = a["load"]
    graph = a["graph"]
    signature = a["signature"]
    name = f"verify_{n}"
    if any(s.signatureKey.decode() == name for s in mt.signatureDefs):
        raise Stop(f"the file already has {name}")
    main = clone_subgraph(mt, graph, signature.subgraphIndex, name,
                          lambda t: a["dims"].get(t), n, a["edits"])
    solved = {}
    for c in a["solved"]:
        decomp = c["decomposition"]
        if decomp in solved:
            prev = solved[decomp]
            for field in ("shapes", "consts", "options", "attrs"):
                if prev[field] != c[field]:
                    raise Stop(f"shared solved decomposition subgraph {decomp} has conflicting {field}")
        solved[decomp] = c
    # The new signature's subgraph goes before its decompositions: the interpreter offers
    # subgraphs to the delegate in index order, and a decomposition is skipped only once the
    # delegate has taken its composite, which it does while building the subgraph that holds it.
    main_index = len(mt.subgraphs)
    mt.subgraphs.append(main)
    copies = 0
    for op in main.operators:
        if graph.code(op) != "STABLEHLO_COMPOSITE":
            continue
        source = op.builtinOptions2.decompositionSubgraphIndex
        sub_name = (mt.subgraphs[source].name or b"").decode() + f"_{name}"
        if source in a["traced"]:
            d_dims, d_edits = a["traced"][source]
            copy = clone_subgraph(mt, graph, source, sub_name, lambda t: d_dims.get(t), n, d_edits)
        elif source in solved:
            c = solved[source]
            def _to_dim(x):
                return ((0, 1),) if x == (0, 1) else ((x,) if x[0] else const_dim(x[1]))
            d_edits = [("reshape_option", k, [_to_dim(x) for x in line])
                       for k, line in c["options"].items()]
            copy = clone_subgraph(mt, graph, source, sub_name,
                                  lambda t: [_to_dim(x) for x in c["shapes"][t]] if t in c["shapes"] else None,
                                  n, d_edits, c["consts"])
            if c["attrs"] and op.builtinOptions2.compositeAttributes is not None:
                attrs = dict(flexbuffers.Loads(bytes(op.builtinOptions2.compositeAttributes)))
                for key, (a_coef, b_coef) in c["attrs"].items():
                    attrs[key] = int(a_coef * n + b_coef)
                op.builtinOptions2.compositeAttributes = np.frombuffer(
                    flexbuffers.Dumps(attrs), dtype=np.uint8)
        else:
            raise Stop(f"decomposition {source} was neither traced nor solved")
        mt.subgraphs.append(copy)
        op.builtinOptions2.decompositionSubgraphIndex = len(mt.subgraphs) - 1
        copies += 1
    import copy as copy_module
    new_signature = copy_module.deepcopy(signature)
    new_signature.signatureKey = name.encode()
    new_signature.subgraphIndex = main_index
    mt.signatureDefs.append(new_signature)
    print(f"{name}: {len(main.operators)} ops, {copies} decompositions copied", flush=True)

    def pack():
        builder = flatbuffers.Builder(graph_end + 2**24)
        builder.Finish(mt.Pack(builder), file_identifier=b"TFL3")
        return bytes(builder.Output())

    new_graph = pack()
    if external:
        delta = 0 if len(new_graph) <= graph_end else -(-(len(new_graph) - graph_end) // ALIGN) * ALIGN
        if delta:
            for b in mt.buffers:
                if b.offset > 1:
                    b.offset += delta
            size = len(new_graph)
            new_graph = pack()
            assert len(new_graph) == size, "offset update changed the graph size"
        section = new_graph + b"\0" * (graph_end + delta - len(new_graph)) + data[graph_end:]
    else:
        delta = 0 if len(new_graph) <= len(data) else -(-(len(new_graph) - len(data)) // ALIGN) * ALIGN
        section = new_graph + b"\0" * (len(data) + delta - len(new_graph))
    print(f"  section grows by {delta:,} bytes", flush=True)

    total = os.path.getsize(path)
    begin, end, _ = sections[index]
    shift = len(section) - (end - begin)
    chunk = 64 * 2**20
    with open(path, "rb") as fin, open(out_path, "wb") as fout:
        def copy_range(start, stop):
            fin.seek(start)
            while start < stop:
                block = fin.read(min(chunk, stop - start))
                fout.write(block)
                start += len(block)
        copy_range(0, begin)
        fout.write(section)
        copy_range(end, total)
    with open(out_path, "r+b") as f:
        header = bytearray(f.read(header_end))
        objects = H.LiteRTLMMetaData.GetRootAs(bytes(header), 32).SectionMetadata()
        for i in range(objects.ObjectsLength()):
            o = objects.Objects(i)
            b, e, _ = sections[i]
            nb = b + (shift if b >= end else 0)
            ne = e + (shift if e >= end else 0)
            for slot, v in ((6, nb), (8, ne)):
                struct.pack_into("<Q", header, o._tab.Pos + o._tab.Offset(slot), v)
        f.seek(0)
        f.write(header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 8
    if "--positions" in args:
        i = args.index("--positions")
        n = int(args[i + 1])
        del args[i:i + 2]
    try:
        if "--report" in args:
            report(args[0])
        elif len(args) == 2:
            write(args[0], args[1], n)
        else:
            raise SystemExit(__doc__)
    except Stop as e:
        raise SystemExit(f"stopped: {e}")
