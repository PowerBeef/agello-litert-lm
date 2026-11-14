"""Open a Gemma 4 E4B .litertlm to the model's trained 131,072-token window (Agello PERFORMANCE.md, F1-F2).

The export sizes every context dimension to the magic number 32003, which LiteRT-LM can only
shrink, and our rotary fix (patch_rope.py --split --block 512) covers positions 0..32767. This
edit, graph only (the weights are untouched):

- F1: every 32003 becomes the prime 131101 (LiteRT-LM then allocates 131,072 positions at most,
  the largest multiple of 256 below it, and anything below on request): tensor shapes, the small
  INT32 shape constants, RESHAPE options, composite attributes (the global cache_update's
  cache_size), and the 0..32002 position constants of an unringed local window, rebuilt;
- F2: the rotary block split reaches 256 blocks of 512. The block's phase is read with GATHER
  from tables of 256 rows, indexed by h = FLOOR_DIV(p, 512), and the position within it is
  l = FLOOR_MOD(p, 512), both in INT32: the candidates p - 512k it replaces were cast to float,
  which on the GPU is float16, whose largest finite value is 65,504. The tables keep the +2 the
  one-hot path stored, and l is the same value, so below 32,768 every result is the same.

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
BLOCK, BLOCKS = 512, 131072 // 512
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

    # ---- F2: the rotary block split, per signature.
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
        block_c = const("rope_block_size_i", np.full([1, 1, 1, 1], BLOCK, np.int32), S.TensorType.INT32)
        l4 = tensor("rope_position_l4", S.TensorType.INT32, [1, N, 1, 1])
        h4 = tensor("rope_position_h4", S.TensorType.INT32, [1, N, 1, 1])
        h1 = tensor("rope_position_h", S.TensorType.INT32, [N])
        h1_shape = const("rope_position_h_shape", np.array([N], np.int32), S.TensorType.INT32)
        cast_o = S.CastOptionsT(); cast_o.inDataType, cast_o.outDataType = S.TensorType.INT32, S.TensorType.FLOAT32
        reshape_h = S.ReshapeOptionsT(); reshape_h.newShape = [N]
        new_low = [
            op("FLOOR_MOD", [pos4, block_c], [l4], S.BuiltinOptions.FloorModOptions, S.FloorModOptionsT()),
            op("CAST", [l4], [low4], S.BuiltinOptions.CastOptions, cast_o),
            op("FLOOR_DIV", [pos4, block_c], [h4], S.BuiltinOptions.FloorDivOptions, S.FloorDivOptionsT()),
            op("RESHAPE", [h4, h1_shape], [h1], S.BuiltinOptions.ReshapeOptions, reshape_h),
        ]
        # The one-hot path it replaces: candidates, the block test, its casts, select and max.
        dead = {by_name[n] for n in ("rope_candidates", "rope_candidates_ge0", "rope_candidates_lt2048",
                                     "rope_block_valid", "rope_block_onehot", "rope_candidates_f",
                                     "rope_candidates_masked")}
        dead.add(low4)
        old_low = [o for o in g.operators if any(int(t) in dead for t in o.outputs)]
        at = min(g.operators.index(o) for o in old_low)
        for o in old_low:
            g.operators.remove(o)
        g.operators[at:at] = new_low

        # Per frequency set: the +2 tables read with GATHER at h.
        muls = consumers(low)
        assert len(muls) == 2, f"{kind}/{sig_name}: expected 2 rotary MULs, found {len(muls)}"
        tables = 0
        for mul in muls:
            inv = next(int(t) for t in mul.inputs if int(t) != low)
            freqs = np.frombuffer(raw(g.tensors[inv]), dtype=np.float32).astype(np.float64).reshape(-1)
            F = freqs.size
            angles = np.outer(BLOCK * np.arange(BLOCKS), freqs)          # [BLOCKS, F]
            for fn, label in ((np.sin, "sin"), (np.cos, "cos")):
                r4 = by_name[f"rope_max4_{label}_{F}"]                   # [1, N, 1, F], was the +2 value
                sel = by_name[f"rope_sel_{label}_{F}"]
                red = by_name[f"rope_max_{label}_{F}"]
                old = [o for o in g.operators if any(int(t) in (sel, red, r4) for t in o.outputs)]
                at = min(g.operators.index(o) for o in old)
                for o in old:
                    g.operators.remove(o)
                table = const(f"rope_rows2_{label}_{F}", (fn(angles) + 2).astype(np.float32), S.TensorType.FLOAT32)
                rows = tensor(f"rope_rows_{label}_{F}", S.TensorType.FLOAT32, [N, F])
                shape_c = const(f"rope_rows_shape_{label}_{F}", np.array([1, N, 1, F], np.int32), S.TensorType.INT32)
                gather_o = S.GatherOptionsT(); gather_o.axis, gather_o.batchDims = 0, 0
                reshape_o = S.ReshapeOptionsT(); reshape_o.newShape = [1, N, 1, F]
                g.operators[at:at] = [
                    op("GATHER", [table, h1], [rows], S.BuiltinOptions.GatherOptions, gather_o),
                    op("RESHAPE", [rows, shape_c], [r4], S.BuiltinOptions.ReshapeOptions, reshape_o),
                ]
                tables += 1
        print(f"  {sig_name}: rotary to {BLOCKS} blocks of {BLOCK} ({tables} tables read with GATHER)", flush=True)

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
