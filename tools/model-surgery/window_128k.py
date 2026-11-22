"""Open a Gemma 4 E4B .litertlm to the model's trained 131,072-token window (Agello PERFORMANCE.md, F1-F2).

The export sizes every context dimension to the magic number 32003, which LiteRT-LM can only
shrink, and our rotary fix (patch_rope.py --split --block 512) covers positions 0..32767. This
edit, graph only (the weights are untouched):

- F1: every 32003 becomes the prime 131101 (LiteRT-LM then allocates 131,072 positions at most,
  the largest multiple of 256 below it, and anything below on request): tensor shapes, the small
  INT32 shape constants, RESHAPE options, composite attributes (the global cache_update's
  cache_size), and the 0..32002 position constants of an unringed local window, rebuilt;
- F2: unshifted Dual-GATHER rotary tables covering 256 blocks of 512 (positions 0..131,071).
  h = FLOOR_DIV(p, 512) and l = FLOOR_MOD(p, 512) in INT32 index precomputed FP64->FP32
  sin/cos tables ([256, F] for 512*h, [512, F] for l, unshifted in [-1, 1], shared across
  signatures), replacing the one-hot block path, the +2 shift/ADD(-2), and the FP16 l*freq
  MUL/SIN/COS ops. Angle addition combines the four exact [-1, 1] lookups, cutting worst-case
  binary16 rotary phase error ~240-300x (~0.237 rad -> <0.001 rad) across 0..131,071.

Run it on a file with ring-buffer local caches (ring_kv.py): at 131,072 positions full-length
local caches would take 2.7 GB more.

Usage (tools venv, ~/.venvs/agello-tools):
  python window_128k.py <in.litertlm> <out.litertlm>
"""
import os
import struct
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers
from litert_lm_builder import litertlm_header_schema_py_generated as H

src, dst = sys.argv[1:3]
OLD, NEW = 32003, 131101
BLOCK = 512
# Size high rotary table to cover all positions 0..NEW-1 (257 blocks = 131,584 >= 131,101).
BLOCKS = (NEW + BLOCK - 1) // BLOCK
ALIGN = 16384
TARGETS = ("tf_lite_prefill_decode", "tf_lite_mtp_drafter")


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
                v = H.StringValue(); v.Init(kv.Value().Bytes, kv.Value().Pos)
                kind = v.Value().decode()
        sections.append((o.BeginOffset(), o.EndOffset(), kind))
    return header_end, sections


def name_of(t):
    return t.name.decode() if isinstance(t.name, bytes) else t.name


def patch_section(begin, end, kind):
    with open(src, "rb") as f:
        f.seek(begin)
        data = f.read(end - begin)
    model = S.Model.GetRootAs(data, 0)
    external = [model.Buffers(i).Offset() for i in range(model.BuffersLength()) if model.Buffers(i).Offset() > 1]
    graph_end = min(external) if external else len(data)
    print(f"{kind}: {len(data):,} bytes, graph {graph_end:,} bytes; unpacking...", flush=True)
    mt = S.ModelT.InitFromObj(S.Model.GetRootAs(data[:graph_end] if external else data, 0))

    def code_of(o):
        c = mt.operatorCodes[o.opcodeIndex]
        return max(c.builtinCode, c.deprecatedBuiltinCode)

    def opcode(name):
        code = getattr(S.BuiltinOperator, name)
        for i, c in enumerate(mt.operatorCodes):
            if max(c.builtinCode, c.deprecatedBuiltinCode) == code:
                return i
        c = S.OperatorCodeT()
        c.builtinCode, c.deprecatedBuiltinCode, c.version = code, min(code, 127), 1
        mt.operatorCodes.append(c)
        return len(mt.operatorCodes) - 1

    def add_buffer(array):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint8)
        mt.buffers.append(b)
        return len(mt.buffers) - 1

    def raw(t):
        b = mt.buffers[t.buffer]
        if b.data is not None and len(b.data):
            return bytes(b.data)
        if b.offset > 1:     # stored after the graph, with the weights
            return data[b.offset:b.offset + b.size]
        return None

    # ---- F1: the magic number, in every subgraph (signatures and composite decompositions).
    shapes = consts = rebuilt = reshapes = attrs = 0
    for g in mt.subgraphs:
        users = {}
        for o in g.operators:
            for t in o.inputs:
                users.setdefault(int(t), []).append(o)
        for i, t in enumerate(g.tensors):
            if t.shape is None or OLD not in list(t.shape):
                continue
            blob = raw(t)
            if blob is not None:
                if t.type != S.TensorType.INT32:
                    raise SystemExit(f"{kind}: constant {name_of(t)} with the context dimension is not INT32")
                values = np.frombuffer(blob, dtype=np.int32)
                offset = int(values[0])
                if not np.array_equal(values, np.arange(OLD, dtype=np.int32) + offset):
                    raise SystemExit(f"{kind}: constant {name_of(t)} is not a position run")
                if i in users:   # still read: rebuild it at the new length
                    b = S.BufferT()
                    b.data = np.frombuffer((np.arange(NEW, dtype=np.int32) + offset).tobytes(), dtype=np.uint8)
                    mt.buffers[t.buffer] = b
                    rebuilt += 1
                else:
                    continue    # unused (ring_kv.py's window replaced it): left as it is
            t.shape = np.array([NEW if d == OLD else d for d in t.shape], dtype=np.int32)
            if t.shapeSignature is not None and len(t.shapeSignature):
                t.shapeSignature = np.array([NEW if d == OLD else d for d in t.shapeSignature], dtype=np.int32)
            shapes += 1
        for t in g.tensors:
            if t.type != S.TensorType.INT32 or (t.shape is not None and OLD in list(t.shape)):
                continue
            blob = raw(t)
            if blob is None or len(blob) > 32:
                continue
            values = np.frombuffer(blob, dtype=np.int32)
            if OLD in values.tolist():
                b = S.BufferT()
                b.data = np.frombuffer(np.where(values == OLD, NEW, values).astype(np.int32).tobytes(), dtype=np.uint8)
                mt.buffers.append(b)
                t.buffer = len(mt.buffers) - 1
                consts += 1
        for o in g.operators:
            if o.builtinOptionsType == S.BuiltinOptions.ReshapeOptions and o.builtinOptions is not None \
                    and o.builtinOptions.newShape is not None and OLD in list(o.builtinOptions.newShape):
                o.builtinOptions.newShape = [NEW if d == OLD else int(d) for d in o.builtinOptions.newShape]
                reshapes += 1
            if code_of(o) == S.BuiltinOperator.STABLEHLO_COMPOSITE and o.builtinOptions2 is not None \
                    and o.builtinOptions2.compositeAttributes is not None:
                a = flexbuffers.Loads(bytes(o.builtinOptions2.compositeAttributes))
                if isinstance(a, dict) and any(isinstance(v, int) and not isinstance(v, bool) and v == OLD for v in a.values()):
                    a = {k: (NEW if isinstance(v, int) and not isinstance(v, bool) and v == OLD else v) for k, v in a.items()}
                    o.builtinOptions2.compositeAttributes = np.frombuffer(flexbuffers.Dumps(a), dtype=np.uint8)
                    attrs += 1
    print(f"  F1: {shapes} tensor shapes, {consts} shape constants, {rebuilt} position runs rebuilt, "
          f"{reshapes} reshape options, {attrs} composite attributes: {OLD} -> {NEW}", flush=True)

    # ---- F2: unshifted Dual-GATHER rotary tables ([256, F] high + [512, F] low), per signature.
    # Tables are shared across all signatures in the section via `table_buffers`.
    table_buffers = {}

    def shared_table_buffer(key, array):
        if key not in table_buffers:
            table_buffers[key] = add_buffer(array)
        return table_buffers[key]

    signatures = [(s.signatureKey.decode(), s.subgraphIndex) for s in mt.signatureDefs] or [("main", 0)]
    for sig_name, index in signatures:
        g = mt.subgraphs[index]
        by_name = {name_of(t): i for i, t in enumerate(g.tensors)}
        if "rope_position4" not in by_name:
            raise SystemExit(f"{kind}/{sig_name}: no split rotary path (run patch_rope.py --split first)")

        def tensor(name, ttype, shape, buffer=0):
            t = S.TensorT()
            t.name, t.type, t.buffer = name.encode(), ttype, buffer
            t.shape = np.array(shape, dtype=np.int32)
            g.tensors.append(t)
            return len(g.tensors) - 1

        def const(name, array, ttype):
            return tensor(name, ttype, list(np.shape(array)) or [1], add_buffer(array))

        def op(code, inputs, outputs, options_type=0, options=None):
            o = S.OperatorT()
            o.opcodeIndex = opcode(code)
            o.inputs = np.array(inputs, dtype=np.int32)
            o.outputs = np.array(outputs, dtype=np.int32)
            o.builtinOptionsType, o.builtinOptions = options_type, options
            return o

        def producer(t):
            return next(o for o in g.operators if t in [int(x) for x in o.outputs])

        def consumers(t):
            return [o for o in g.operators if t in [int(x) for x in o.inputs]]

        pos4 = by_name["rope_position4"]
        N = int(g.tensors[pos4].shape[1])
        low4 = by_name["rope_position_low4"]
        low = by_name["rope_position_low"]

        # Inspect frequency sets before mutating the operator list.
        muls = consumers(low)
        assert len(muls) == 2, f"{kind}/{sig_name}: expected 2 rotary MULs, found {len(muls)}"
        freq_sets = []
        for mul in muls:
            inv = next(int(t) for t in mul.inputs if int(t) != low)
            freqs = np.frombuffer(raw(g.tensors[inv]), dtype=np.float32).astype(np.float64).reshape(-1)
            F = freqs.size
            mul_reshapes = consumers(int(mul.outputs[0]))
            assert len(mul_reshapes) == 1 and code_of(mul_reshapes[0]) == S.BuiltinOperator.RESHAPE
            trig = consumers(int(mul_reshapes[0].outputs[0]))
            assert len(trig) == 2
            sin_op = next(o for o in trig if code_of(o) == S.BuiltinOperator.SIN)
            cos_op = next(o for o in trig if code_of(o) == S.BuiltinOperator.COS)
            sin_l, cos_l = int(sin_op.outputs[0]), int(cos_op.outputs[0])
            ts = by_name[f"rope_high_sin_{F}"]
            tc = by_name[f"rope_high_cos_{F}"]
            add_s, add_c = producer(ts), producer(tc)
            assert code_of(add_s) == S.BuiltinOperator.ADD and code_of(add_c) == S.BuiltinOperator.ADD
            assert len(consumers(sin_l)) == 2 and len(consumers(cos_l)) == 2
            assert len(consumers(ts)) == 2 and len(consumers(tc)) == 2
            freq_sets.append((freqs, F, mul, mul_reshapes[0], sin_op, cos_op, sin_l, cos_l, ts, tc))

        block_c = const("rope_block_size_i", np.full([1, 1, 1, 1], BLOCK, np.int32), S.TensorType.INT32)
        l4 = tensor("rope_position_l4", S.TensorType.INT32, [1, N, 1, 1])
        l1 = tensor("rope_position_l", S.TensorType.INT32, [N])
        l1_shape = const("rope_position_l_shape", np.array([N], np.int32), S.TensorType.INT32)
        h4 = tensor("rope_position_h4", S.TensorType.INT32, [1, N, 1, 1])
        h1 = tensor("rope_position_h", S.TensorType.INT32, [N])
        h1_shape = const("rope_position_h_shape", np.array([N], np.int32), S.TensorType.INT32)
        reshape_l = S.ReshapeOptionsT(); reshape_l.newShape = [N]
        reshape_h = S.ReshapeOptionsT(); reshape_h.newShape = [N]
        new_split = [
            op("FLOOR_MOD", [pos4, block_c], [l4], S.BuiltinOptions.FloorModOptions, S.FloorModOptionsT()),
            op("RESHAPE", [l4, l1_shape], [l1], S.BuiltinOptions.ReshapeOptions, reshape_l),
            op("FLOOR_DIV", [pos4, block_c], [h4], S.BuiltinOptions.FloorDivOptions, S.FloorDivOptionsT()),
            op("RESHAPE", [h4, h1_shape], [h1], S.BuiltinOptions.ReshapeOptions, reshape_h),
        ]
        # Replace the one-hot path and low-position float reshape with pure INT32 h1 and l1.
        dead = {by_name[n] for n in ("rope_candidates", "rope_candidates_ge0", "rope_candidates_lt2048",
                                     "rope_block_valid", "rope_block_onehot", "rope_candidates_f",
                                     "rope_candidates_masked", "rope_position_low4", "rope_position_low")}
        old_low = [o for o in g.operators if any(int(t) in dead for t in o.outputs)]
        at = min(g.operators.index(o) for o in old_low)
        for o in old_low:
            g.operators.remove(o)
        g.operators[at:at] = new_split

        # Per frequency set: four unshifted GATHERs ([256, F] high + [512, F] low, sin & cos).
        tables = 0
        for freqs, F, mul, mul_reshape, sin_op, cos_op, sin_l, cos_l, ts, tc in freq_sets:
            high_angles = np.outer(BLOCK * np.arange(BLOCKS), freqs)     # [BLOCKS, F] in FP64
            low_angles = np.outer(np.arange(BLOCK), freqs)               # [BLOCK, F] in FP64
            old_high_t = {by_name[f"rope_sel_{label}_{F}"] for label in ("sin", "cos")} | \
                         {by_name[f"rope_max_{label}_{F}"] for label in ("sin", "cos")} | \
                         {by_name[f"rope_max4_{label}_{F}"] for label in ("sin", "cos")} | \
                         {ts, tc}
            old_ops = [o for o in g.operators if any(int(t) in old_high_t for t in o.outputs)]
            old_ops.extend([mul, mul_reshape, sin_op, cos_op])
            for o in old_ops:
                g.operators.remove(o)

            # Insert right before the angle-addition ops consuming ts/tc/sin_l/cos_l.
            angle_consumers = consumers(ts) + consumers(tc) + consumers(sin_l) + consumers(cos_l)
            at = min(g.operators.index(o) for o in angle_consumers)
            new_ops = []
            for part, idx_t, angles, targets in (
                ("high", h1, high_angles, (("sin", np.sin, ts), ("cos", np.cos, tc))),
                ("low", l1, low_angles, (("sin", np.sin, sin_l), ("cos", np.cos, cos_l))),
            ):
                for label, fn, dst_t in targets:
                    arr = fn(angles).astype(np.float32)
                    buf = shared_table_buffer((part, label, F, freqs.tobytes()), arr)
                    table = tensor(f"rope_{part}_{label}_table_{F}", S.TensorType.FLOAT32, list(arr.shape), buf)
                    rows = tensor(f"rope_{part}_{label}_rows_{F}", S.TensorType.FLOAT32, [N, F])
                    shape_c = const(f"rope_{part}_{label}_shape_{F}", np.array([1, N, 1, F], np.int32), S.TensorType.INT32)
                    gather_o = S.GatherOptionsT(); gather_o.axis, gather_o.batchDims = 0, 0
                    reshape_o = S.ReshapeOptionsT(); reshape_o.newShape = [1, N, 1, F]
                    new_ops.extend([
                        op("GATHER", [table, idx_t], [rows], S.BuiltinOptions.GatherOptions, gather_o),
                        op("RESHAPE", [rows, shape_c], [dst_t], S.BuiltinOptions.ReshapeOptions, reshape_o),
                    ])
                    tables += 1
            g.operators[at:at] = new_ops
        print(f"  {sig_name}: Dual-GATHER rotary ({BLOCKS}x{BLOCK}, {tables} unshifted tables)", flush=True)

    def pack():
        builder = flatbuffers.Builder(graph_end + 2**22)
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
        out = new_graph + b"\0" * (graph_end + delta - len(new_graph)) + data[graph_end:]
    else:
        delta = 0 if len(new_graph) <= len(data) else -(-(len(new_graph) - len(data)) // ALIGN) * ALIGN
        out = new_graph + b"\0" * (len(data) + delta - len(new_graph))
    print(f"  section grows by {delta:,} bytes", flush=True)
    return out


header_end, sections = read_header(src)
new_sections = {i: patch_section(b, e, k) for i, (b, e, k) in enumerate(sections) if k in TARGETS}
total = os.path.getsize(src)
shifts, running = [], 0
for i, (b, e, k) in enumerate(sections):
    shifts.append(running)
    if i in new_sections:
        running += len(new_sections[i]) - (e - b)

CHUNK = 64 * 2**20
with open(src, "rb") as fin, open(dst, "wb") as fout:
    def copy(start, stop):
        fin.seek(start)
        while start < stop:
            chunk = fin.read(min(CHUNK, stop - start))
            fout.write(chunk)
            start += len(chunk)
    cursor = 0
    for i, (b, e, k) in enumerate(sections):
        copy(cursor, b)
        if i in new_sections:
            fout.write(new_sections[i])
        else:
            copy(b, e)
        cursor = e
    copy(cursor, total)

with open(dst, "r+b") as f:
    header = bytearray(f.read(header_end))
    objects = H.LiteRTLMMetaData.GetRootAs(bytes(header), 32).SectionMetadata()
    for i in range(objects.ObjectsLength()):
        o = objects.Objects(i)
        b, e, _ = sections[i]
        new_end = b + shifts[i] + (len(new_sections[i]) if i in new_sections else e - b)
        for slot, value in ((6, b + shifts[i]), (8, new_end)):
            struct.pack_into("<Q", header, o._tab.Pos + o._tab.Offset(slot), value)
    f.seek(0)
    f.write(header)
print(f"wrote {dst} (window {NEW}, rotary {BLOCKS} blocks)")
