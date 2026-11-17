"""Check a verify_<N> signature against verify on the CPU (Agello PERFORMANCE.md, B5).

Runs LiteRT's interpreter on the main section of a file that `verify_n.py` wrote:
- verify_<N> over positions 0..N-1, from an empty cache;
- verify over positions 0..3, then over 4..7, ... (each call reading the cache the last wrote),
  so every row of verify_<N> has a counterpart computed by the export's own signature.
Prints, per row, the largest logit difference and whether the greedy picks agree.

On the published E4B file (2026-11-17) every row matched bit for bit, logits and caches. A file
with ring-buffer local caches (ring_kv.py) can't run on the CPU at all, its own verify included:
the decompositions keep the full cache length, which the GPU never reads; check those on the GPU.

Usage (tools venv): python check_verify_n.py <file.litertlm> <scratch dir> [--positions 8]
"""

import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

import verify_n as V

args = sys.argv[1:]
n = 8
if "--positions" in args:
    i = args.index("--positions")
    n = int(args[i + 1])
    del args[i:i + 2]
path, scratch = args
if n % 4:
    raise SystemExit("the comparison chains verify calls of 4: use a multiple of 4")

_, sections = V.read_header(path)
begin, end, _ = next(s for s in sections if s[2] == V.SECTION)
tflite = os.path.join(scratch, "prefill_decode.tflite")
if not os.path.exists(tflite) or os.path.getsize(tflite) != end - begin:
    with open(path, "rb") as fin, open(tflite, "wb") as fout:
        fin.seek(begin)
        left = end - begin
        while left:
            block = fin.read(min(left, 64 << 20))
            fout.write(block)
            left -= len(block)
print(f"section: {end - begin:,} bytes", flush=True)

interpreter = Interpreter(model_path=tflite, num_threads=os.cpu_count())
long_run = interpreter.get_signature_runner(f"verify_{n}")
short = interpreter.get_signature_runner("verify")
details = short.get_input_details()
rng = np.random.default_rng(0)
embeddings = rng.normal(0, 1, (1, n, 2560)).astype(np.float32)
per_layer = rng.normal(0, 0.1, (1, n, 42, 256)).astype(np.float32)
window = details["mask"]["shape"][-1]


def inputs(start, count, caches):
    feed = {name: caches[name] for name in caches}
    feed["embeddings"] = embeddings[:, start:start + count]
    feed["per_layer_embeddings"] = per_layer[:, start:start + count]
    feed["input_pos"] = np.arange(start, start + count, dtype=np.int32)
    mask = np.zeros((1, 1, count, window), dtype=bool)
    for row in range(count):
        mask[0, 0, row, :start + row + 1] = True
    feed["mask"] = mask
    params = np.zeros((1, 1, 1, 7), dtype=np.int32)
    params[0, 0, 0, :4] = [start, start + count, start + count, count]
    feed["param_tensor"] = params
    return feed


empty = {name: np.zeros(d["shape"], dtype=d["dtype"]) for name, d in details.items() if name.startswith("kv_cache_")}
print(f"running verify_{n} on the CPU...", flush=True)
result = long_run(**inputs(0, n, empty))
caches = dict(empty)
chained = []
for start in range(0, n, 4):
    print(f"running verify over {start}..{start + 3}...", flush=True)
    out = short(**inputs(start, 4, caches))
    chained.append(out["logits"][0])
    caches = {name: out[name] for name in caches}
chained = np.concatenate(chained)
logits = result["logits"][0]
worst = 0.0
agree = 0
for row in range(n):
    diff = float(np.max(np.abs(logits[row] - chained[row])))
    same = int(np.argmax(logits[row])) == int(np.argmax(chained[row]))
    agree += same
    worst = max(worst, diff)
    print(f"row {row}: max |logit difference| {diff:.6f}, greedy picks {'agree' if same else 'DIFFER'}")
print(f"{agree} of {n} picks agree; largest difference {worst:.6f}")
picks = [int(np.argmax(logits[row])) for row in range(n)]
spread = [float(np.std(logits[row])) for row in range(n)]
print(f"the rows are real: picks {picks}, logit spread per row {min(spread):.2f} to {max(spread):.2f}, "
      f"{len({tuple(np.round(logits[row][:16], 3)) for row in range(n)})} distinct rows of {n}")
for name in caches:
    if not np.array_equal(result[name], caches[name]):
        print(f"cache {name} differs")
        break
else:
    print("every cache the two write is identical")
