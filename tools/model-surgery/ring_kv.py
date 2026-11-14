"""Make a Gemma 4 .litertlm's local-attention KV caches ring buffers (Agello PERFORMANCE.md, E2).

Gemma 4's sliding-window layers attend to the last 512 positions only, yet the export sizes
their caches like the global ones, to the whole context (the magic number 32003). This edit
gives every local cache R slots (default 1,536: a 1,024-token prefill chunk plus the 511 older
positions its first query still reads), in the main graph (all signatures) and in the MTP
drafter (which reads cache 22):

- the local caches' shapes, and every tensor of the local attention path whose size followed
  them (cache_update outputs, scores, mask select, softmax), get R in place of 32003; constant
  shape operands on that path are duplicated with R, so the global layers keep theirs;
- the local cache_update composites get cache_size R and is_ring_buffer: the GPU kernel writes
  position p to slot p mod R (the runtime must pass the update length in params[3]);
- the local window mask, shared by every local layer, follows the ring's slot order. For query
  position p and slot s, d = FLOOR_MOD(p - s, R) is how many positions back the slot's latest
  content is; the slot is kept when d < 512 (inside the window) and d <= p (it was written).
  It replaces GREATER_EQUAL(p, keys) and LESS(p, keys + 512) on the 0..32002 constants.

In exact arithmetic the outputs are unchanged: the same keys and values are read, from other
slots. The weights are untouched (they follow the graph, which moves by 16 KB multiples).

The runtime needs agello-one a44d5f25 or later: the update length in params[3], window
detection on position keys only, and the value product's loop clamped to the cache.

From our E4B file (PowerBeef02/gemma-4-E4B-it-litert-lm-fp16-rope at a1693e22, sha256
6353a233...f655c) this writes sha256 23551f43...2fa2b46e, 3,660,759,040 bytes.

Usage (tools venv, ~/.venvs/agello-tools):
  python ring_kv.py <in.litertlm> <out.litertlm> [--slots 1536]
Debugging (no rings; the window rewritten with the same op kinds, outputs unchanged in exact
arithmetic), used to show integer SUB and FLOOR_MOD are exact on the GPU past 2,048:
  python ring_kv.py <in> <out> --sub-window-only     d = p - key; d >= 0 and d < 512
  python ring_kv.py <in> <out> --mod-window-only     the same through FLOOR_MOD(d, 2^20)
"""
import os
import struct
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers
from litert_lm_builder import litertlm_header_schema_py_generated as H

args = sys.argv[1:]
R = 1536
if "--slots" in args:
    i = args.index("--slots")
    R = int(args[i + 1])
    del args[i:i + 2]
SUB_ONLY = "--sub-window-only" in args   # debugging: no rings, the window as d = p - key only
if SUB_ONLY:
    args.remove("--sub-window-only")
MOD_ONLY = "--mod-window-only" in args   # debugging: no rings, d = FLOOR_MOD(p - key, 2^20)
if MOD_ONLY:
    args.remove("--mod-window-only")
    SUB_ONLY = True
src, dst = args
MAGIC = 32003         # the export's context dimension
WINDOW = 512
ALIGN = 16384         # section alignment in the container
TARGETS = ("tf_lite_prefill_decode", "tf_lite_mtp_drafter")
assert R % MAGIC != 0 and R >= 1024 + WINDOW - 1


def _aligned_buffer_pack(self, builder):
    """BufferT.Pack honoring the schema's force_align: 16 for inline weights."""
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

    def buffer_bytes(t):
        b = mt.buffers[t.buffer]
        if b.data is not None and len(b.data):
            return bytes(b.data)
        if b.offset > 1:
            return data[b.offset:b.offset + b.size]
        return None

    def composite_name(o):
        if code_of(o) != S.BuiltinOperator.STABLEHLO_COMPOSITE:
            return None
        n = o.builtinOptions2.name
        return n.decode() if isinstance(n, bytes) else n

    signatures = [(s.signatureKey.decode(), s.subgraphIndex) for s in mt.signatureDefs] or [("main", 0)]
    rings = 0
    for sig_name, index in signatures:
        g = mt.subgraphs[index]

        def tensor(name, ttype, shape, buffer=0):
            t = S.TensorT()
            t.name, t.type, t.buffer = name.encode(), ttype, buffer
            t.shape = np.array(shape, dtype=np.int32)
            g.tensors.append(t)
            return len(g.tensors) - 1

        def const_i32(name, array):
            array = np.asarray(array, dtype=np.int32)
            return tensor(name, S.TensorType.INT32, list(array.shape), add_buffer(array))

        def op(code, inputs, outputs, options_type=0, options=None):
            o = S.OperatorT()
            o.opcodeIndex = opcode(code)
            o.inputs = np.array(inputs, dtype=np.int32)
            o.outputs = np.array(outputs, dtype=np.int32)
            o.builtinOptionsType, o.builtinOptions = options_type, options
            return o

        def int_const(i):
            t = g.tensors[i]
            if t.type != S.TensorType.INT32:
                return None
            raw = buffer_bytes(t)
            return None if raw is None else np.frombuffer(raw, dtype=np.int32)

        producer = {}
        for o in g.operators:
            for t in o.outputs:
                producer[int(t)] = o

        # 1. The local window: LOGICAL_AND(GREATER_EQUAL(p, keys), LESS(p, keys + 512)).
        windows = []
        for o in g.operators:
            if code_of(o) != S.BuiltinOperator.LOGICAL_AND:
                continue
            ge, lt = producer.get(int(o.inputs[0])), producer.get(int(o.inputs[1]))
            if ge is None or lt is None:
                continue
            if code_of(ge) == S.BuiltinOperator.LESS:
                ge, lt = lt, ge
            if code_of(ge) != S.BuiltinOperator.GREATER_EQUAL or code_of(lt) != S.BuiltinOperator.LESS:
                continue
            if int(ge.inputs[0]) != int(lt.inputs[0]):
                continue
            keys, shifted = int_const(int(ge.inputs[1])), int_const(int(lt.inputs[1]))
            if keys is None or shifted is None or keys.size != MAGIC or not np.all(shifted - keys == WINDOW):
                continue
            windows.append((o, ge, lt))
        assert len(windows) == 1, f"{kind}/{sig_name}: expected one local window, found {len(windows)}"
        and_op, ge, lt = windows[0]
        pos = int(ge.inputs[0])
        pos_shape = list(g.tensors[pos].shape)
        keys_shape = list(g.tensors[int(ge.inputs[1])].shape)
        assert keys_shape[-1] == MAGIC and pos_shape[-1] == 1, (pos_shape, keys_shape)
        if SUB_ONLY:
            keys_t = int(ge.inputs[1])
            shape = [max(a, b) for a, b in zip(pos_shape, keys_shape)]
            zero_c = const_i32("window_zero", np.zeros([1] * len(pos_shape), np.int32))
            window_c = const_i32(f"window_{WINDOW}", np.full([1] * len(pos_shape), WINDOW, np.int32))
            back = tensor(name_of(g.tensors[pos]) + "_minus_key", S.TensorType.INT32, shape)
            sub_o = S.SubOptionsT(); sub_o.fusedActivationFunction = S.ActivationFunctionType.NONE
            ge.inputs = np.array([back, zero_c], dtype=np.int32)     # p - k >= 0
            lt.inputs = np.array([back, window_c], dtype=np.int32)   # p - k < 512
            at = min(g.operators.index(ge), g.operators.index(lt))
            sub_op = op("SUB", [pos, keys_t], [back], S.BuiltinOptions.SubOptions, sub_o)
            if MOD_ONLY:
                big_c = const_i32("window_mod_big", np.full([1] * len(pos_shape), 1 << 20, np.int32))
                modded = tensor(name_of(g.tensors[pos]) + "_minus_key_mod", S.TensorType.INT32, shape)
                ge.inputs = np.array([modded, zero_c], dtype=np.int32)
                lt.inputs = np.array([modded, window_c], dtype=np.int32)
                g.operators[at:at] = [sub_op, op("FLOOR_MOD", [back, big_c], [modded],
                                                 S.BuiltinOptions.FloorModOptions, S.FloorModOptionsT())]
            else:
                g.operators[at:at] = [sub_op]
            print(f"  {sig_name}: window as p - key{' mod 2^20' if MOD_ONLY else ''} (no rings)", flush=True)
            continue
        out_shape = [max(a, b) for a, b in zip(pos_shape, keys_shape[:-1] + [R])]
        slots = const_i32(f"ring_slots_{R}", np.arange(R, dtype=np.int32).reshape(keys_shape[:-1] + [R]))
        size_c = const_i32(f"ring_size_{R}", np.full([1] * len(pos_shape), R, np.int32))
        window_c = const_i32(f"ring_window_{WINDOW}", np.full([1] * len(pos_shape), WINDOW, np.int32))
        back = tensor(name_of(g.tensors[pos]) + "_ring_minus_slot", S.TensorType.INT32, out_shape)
        dist = tensor(name_of(g.tensors[pos]) + "_ring_distance", S.TensorType.INT32, out_shape)
        sub_o = S.SubOptionsT(); sub_o.fusedActivationFunction = S.ActivationFunctionType.NONE
        new_ops = [
            op("SUB", [pos, slots], [back], S.BuiltinOptions.SubOptions, sub_o),
            op("FLOOR_MOD", [back, size_c], [dist], S.BuiltinOptions.FloorModOptions, S.FloorModOptionsT()),
        ]
        ge.inputs = np.array([pos, dist], dtype=np.int32)       # p >= d: the slot was written
        lt.inputs = np.array([dist, window_c], dtype=np.int32)  # d < 512: inside the window
        at = min(g.operators.index(ge), g.operators.index(lt))
        g.operators[at:at] = new_ops

        # 2. The local caches: K [1, 2, T, 256] and V [1, 2, 256, T] graph inputs.
        inputs = {name_of(g.tensors[int(t)]): int(t) for t in g.inputs}
        local = set()
        for n, t in inputs.items():
            shape = list(g.tensors[t].shape)
            if "kv_cache_k" in n and len(shape) == 4 and shape[2] == MAGIC and shape[3] == 256:
                local.add(t)
            if "kv_cache_v" in n and len(shape) == 4 and shape[3] == MAGIC and shape[2] == 256:
                local.add(t)
        assert local, f"{kind}/{sig_name}: no local cache inputs"

        # 3. Taint the local path forward, in graph order: an op reading a tainted tensor taints
        #    its outputs that carry the context dimension.
        tainted = set(local) | {int(ge.outputs[0]), int(lt.outputs[0])}
        changed_ops = []
        for o in g.operators:
            ins = [int(t) for t in o.inputs if t >= 0]
            if not any(t in tainted for t in ins):
                continue
            outs = [int(t) for t in o.outputs if MAGIC in list(g.tensors[int(t)].shape)]
            if not outs:
                continue
            for t in ins:
                shape = list(g.tensors[t].shape)
                if MAGIC in shape and t not in tainted:
                    raise SystemExit(f"{kind}/{sig_name}: op reads tainted and untainted context tensors ({name_of(g.tensors[t])})")
            tainted.update(outs)
            changed_ops.append(o)

        for t in tainted:
            ten = g.tensors[t]
            ten.shape = np.array([R if d == MAGIC else d for d in ten.shape], dtype=np.int32)
            if ten.shapeSignature is not None and len(ten.shapeSignature):
                ten.shapeSignature = np.array([R if d == MAGIC else d for d in ten.shapeSignature], dtype=np.int32)

        # Constant shape operands of the tainted ops (RESHAPE shapes, broadcast shapes...):
        # duplicated with R, so ops elsewhere that share them keep the context dimension.
        dup = 0
        for o in changed_ops:
            new_inputs = []
            for t in o.inputs:
                t = int(t)
                values = int_const(t) if t >= 0 else None
                if values is not None and MAGIC in values.tolist() and values.size <= 8:
                    new_inputs.append(const_i32(name_of(g.tensors[t]) + f"_ring{R}",
                                                np.where(values == MAGIC, R, values).reshape(list(g.tensors[t].shape))))
                    dup += 1
                else:
                    new_inputs.append(t)
            o.inputs = np.array(new_inputs, dtype=np.int32)
            if o.builtinOptionsType == S.BuiltinOptions.ReshapeOptions and o.builtinOptions is not None \
                    and o.builtinOptions.newShape is not None:
                o.builtinOptions.newShape = [R if d == MAGIC else int(d) for d in o.builtinOptions.newShape]

        # 4. The local cache_update composites become rings of R slots.
        updates = 0
        for o in changed_ops:
            if composite_name(o) != "odml.cache_update":
                continue
            attrs = flexbuffers.Loads(bytes(o.builtinOptions2.compositeAttributes))
            assert attrs["cache_size"] == MAGIC and attrs["head_size"] == 256, attrs
            attrs["cache_size"] = R
            attrs["is_ring_buffer"] = True
            o.builtinOptions2.compositeAttributes = np.frombuffer(flexbuffers.Dumps(attrs), dtype=np.uint8)
            updates += 1
        rings += updates
        print(f"  {sig_name}: {len(local)} local cache inputs, {len(tainted)} tensors to {R}, "
              f"{dup} shape constants duplicated, {updates} cache_update rings, window mask on slots", flush=True)

    def pack():
        builder = flatbuffers.Builder(graph_end + 2**20)
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
print(f"wrote {dst} (local caches: {R} slots)")
