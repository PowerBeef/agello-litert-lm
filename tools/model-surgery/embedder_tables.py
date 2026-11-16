"""Where a Gemma 4 .litertlm keeps its embedding tables, and how to read them without the CPU
embedders (the GPU-resident speculative round, stage 0: Agello's
build/drafts/gpu-resident-round-design-2026-11-16.md).

Both embedders are dequantized gathers: `tf_lite_embedder` is one EMBEDDING_LOOKUP on an INT2
[262144, 2560] table with a scale per row (zero points all 0), then a MUL by sqrt(2560);
`tf_lite_per_layer_embedder` is 42 lookups on INT2 [262144, 256] tables, concatenated in op
order and reshaped to [1, 1, 42, 256]. An element is the row's 2-bit field, low bits first in
each byte, read as signed (-2..1), times the row's float32 scale, then times the MUL constant.

Prints each table's file offset, size and scale vector, and with --check runs both embedders in
the LiteRT interpreter (CPU) for 0, 1, 2, 105, 106, 262143 and N random token ids, and compares
the rows read from the mapped file bit for bit (2026-11-16, the 128K file: identical for 1,006
ids in both). --manifest writes the offsets as JSON.

Usage (in the tools venv, ~/.venvs/agello-tools):
  python embedder_tables.py <model.litertlm> [--check 1000] [--manifest tables.json]
"""
import argparse
import json
import mmap

import numpy as np
from ai_edge_litert import schema_py_generated as S

from inspect_e4b import OPS, sections

EMBEDDERS = ["tf_lite_embedder", "tf_lite_per_layer_embedder"]
# QuantizationParameters' vtable slots (schema.fbs): min, max, scale, zero_point.
SCALE_SLOT, ZERO_POINT_SLOT = 8, 10


def vector(table, slot):
    """(start, length) of a flatbuffer vector field, start relative to the section."""
    offset = table._tab.Offset(slot)
    return table._tab.Vector(offset), table._tab.VectorLen(offset)


def read_tables(mm, path, wanted):
    begin, end, _ = next(s for s in sections(path) if s[2] == wanted)
    model = S.Model.GetRootAs(mm[begin:end], 0)
    graph = model.Subgraphs(0)
    tables, mul = [], None
    for index in range(graph.OperatorsLength()):
        op = graph.Operators(index)
        code = model.OperatorCodes(op.OpcodeIndex())
        name = OPS.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()))
        if name == "EMBEDDING_LOOKUP":
            tensor = graph.Tensors(op.Inputs(1))
            start, size = vector(model.Buffers(tensor.Buffer()), 4)
            quant = tensor.Quantization()
            scale_start, scale_count = vector(quant, SCALE_SLOT)
            zero_start, zero_count = vector(quant, ZERO_POINT_SLOT)
            rows, cols = tensor.Shape(0), tensor.Shape(1)
            zeros = np.frombuffer(mm, np.int64, count=zero_count, offset=begin + zero_start)
            assert size * 4 == rows * cols and scale_count == rows and not zeros.any(), wanted
            tables.append({
                "op": index, "file_offset": begin + start, "bytes": size, "rows": rows, "cols": cols,
                "scale_offset": begin + scale_start,
            })
        elif name == "MUL":
            tensor = graph.Tensors(op.Inputs(1))
            mul = float(np.frombuffer(bytes(model.Buffers(tensor.Buffer()).DataAsNumpy()), np.float32)[0])
        elif name not in ("GREATER_EQUAL", "LESS", "LOGICAL_AND", "SELECT_V2", "RESHAPE", "CONCATENATION"):
            raise SystemExit(f"{wanted}: unexpected op {name}; the GPU gather wouldn't match it")
    return {"section": [begin, end], "tables": tables, "mul": mul}


def dequantize(mm, layout, ids):
    rows = []
    for table in layout["tables"]:
        packed = np.frombuffer(mm, np.uint8, count=table["bytes"], offset=table["file_offset"])
        scale = np.frombuffer(mm, np.float32, count=table["rows"], offset=table["scale_offset"])
        fields = (packed.reshape(table["rows"], table["cols"] // 4)[ids][:, :, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
        values = np.where(fields >= 2, fields.astype(np.int8) - 4, fields.astype(np.int8)).reshape(len(ids), table["cols"])
        rows.append(values.astype(np.float32) * scale[ids][:, None])
    out = np.concatenate(rows, axis=1)
    return out * np.float32(layout["mul"]) if layout["mul"] is not None else out


def reference(mm, layout, ids):
    from ai_edge_litert.interpreter import Interpreter

    begin, end = layout["section"]
    interpreter = Interpreter(model_content=bytes(mm[begin:end]))
    runner = interpreter.get_signature_runner(next(iter(interpreter.get_signature_list())))
    return np.stack([next(iter(runner(token_ids=np.array([[t]], np.int32)).values())).reshape(-1) for t in ids])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--check", type=int, default=0, help="random token ids to compare")
    parser.add_argument("--manifest")
    parser.add_argument("--dump-reference", help="with --check: the ids (int32) and the interpreter's rows (float32), embedder then per-layer, for tools/agello-gpu-round")
    args = parser.parse_args()
    with open(args.model, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    layouts = {wanted: read_tables(mm, args.model, wanted) for wanted in EMBEDDERS}
    for wanted, layout in layouts.items():
        first = layout["tables"][0]
        print(f"{wanted}: {len(layout['tables'])} INT2 table(s) [{first['rows']}, {first['cols']}], "
              f"{sum(t['bytes'] for t in layout['tables']) / 2**20:.0f} MiB, mul {layout['mul']}")
        for table in layout["tables"][:3]:
            print(f"  op {table['op']}: rows at {table['file_offset']}, scales at {table['scale_offset']}")
    if args.check:
        ids = np.concatenate([[0, 1, 2, 105, 106, 262143],
                              np.random.default_rng(7).integers(0, 262144, args.check)]).astype(np.int32)
        dumps = [ids.astype(np.int32).tobytes()]
        for wanted, layout in layouts.items():
            got, want = dequantize(mm, layout, ids), reference(mm, layout, ids)
            dumps.append(want.astype(np.float32).tobytes())
            same = np.array_equal(got.astype(np.float32).view(np.uint32), want.astype(np.float32).view(np.uint32))
            print(f"{wanted}: {len(ids)} ids {'bit-identical' if same else 'DIFFER, max ' + str(np.abs(got - want).max())}")
            if not same:
                raise SystemExit(1)
        if args.dump_reference:
            with open(args.dump_reference, "wb") as f:
                f.write(np.int32(len(ids)).tobytes() + b"".join(dumps))
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(layouts, f, indent=1)


if __name__ == "__main__":
    main()
