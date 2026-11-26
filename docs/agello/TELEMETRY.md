# Telemetry: probes into our LiteRT-LM build

Goal: see what every turn, step and kernel does in our own build (LiteRT-LM + LiteRT + ML Drift
on Metal), catch errors and silent degradations as they happen, and point at optimization
opportunities with numbers. Cheap enough to leave on in development builds; detailed modes are
opt-in. Research behind it: four inventories of the three code bases and of Apple's APIs
(findings below, with the probe each one calls for).

## What exists today, and why it isn't enough

| Facility | Where | Problem |
| --- | --- | --- |
| `BenchmarkInfo` | LiteRT-LM `io_types.cc`, `tasks.cc` | Benchmark mode changes behavior (busy-wait on GPU, forced prefill waits, stats reset); per-step marks go to a copy that's discarded (`tasks.cc:222, 536`); unsynchronized reads while the execution thread writes |
| `ExecutorStats` | LiteRT-LM executor | Always on, MTP acceptance and timings, but only reachable through benchmark mode |
| LiteRT profiler | `litert/runtime/profiler.*` | On Metal, an attached profiler replays the whole GPU graph on every inference (`delegate_kernel.cc:123`) |
| `LITERT_PERFETTO_TRACE_EVENT` | ~30 call sites in LiteRT | Defined empty (`perfetto_profiling.h:18-21`) |
| ML Drift `InferenceContext::Profile` | `inference_context.mm:390-546` | Per-kernel timestamps, but an isolated replay; offline |
| RuntimeDebugger | LiteRT-LM, compile-time | Dumps every tensor to disk; far too heavy for timing |
| Kernel names | ML Drift | Every kernel is `ComputeFunction`; no pipeline, encoder or command-buffer labels, so Instruments can't attribute GPU time |
| Logs | absl to stderr, LiteRT logger, os_log in Swift | Three sinks; nothing structured |

## Design

One small library, `@agello_telemetry` (`agello/telemetry/` in the LiteRT-LM fork, its own Bazel
repository so LiteRT and ML Drift can depend on it), with a C ABI and header-only C++ helpers.
Every probe is an inline check of one atomic level before any work.

| Level | Name | Records | Cost when on |
| --- | --- | --- | --- |
| 0 | off | nothing; probes are a load and a branch | ~1 ns |
| 1 | counters (default in our builds) | fixed set of counters and anomaly events | one relaxed atomic add |
| 2 | spans | per-kind aggregates (count, total, max, log2 histogram), DynamicTracing signposts | ~40 ns per span |
| 3 | trace | spans also go to per-thread ring buffers, written as a Chrome/Perfetto trace | +~20 ns |
| 4 | deep | per-kernel GPU timing (one encoder per kernel), per-row numerical probes | changes timing; for diagnosis only (graph dumps work at any level via `AGELLO_TELEMETRY_GRAPH_DIR`) |

- **Counters** (fixed enum, no strings): prompt and generated tokens; runs per signature (prefill
  128/1024, decode, `verify`, `verify8` + `verify8_rows_compared`/`verify8_rows_agreed`, drafter);
  host buffer locks, copies and bytes; command buffers
  committed; GPU waits and wait time; mask fills and bytes; embedding lookups (CPU model runs);
  sampler path (GPU/CPU) and fallbacks; logit-mask path; KV copies and context switches; program
  and weight cache hits, misses and rebuilds; allocations on the decode path; run-option
  parses; stop reasons (EOS, max tokens, KV full, cancel); errors by status code.
- **Spans** (fixed enum of kinds, plus interned names for LiteRT's existing trace sites): turn,
  template render, tokenize, prefill, prefill chunk, embedding lookup, mask fill, parameter fill,
  bind, run (`run.prefill`, `run.decode`, `run.verify`, `run.verify8`, `run.drafter`), GPU wait,
  decode step, sample, constraint, detokenize, stream callback, MTP draft and
  verify (`gpu.verify`, `gpu.verify8`), KV clone and restore, task queue wait, engine and session
  init phases, delegate invoke.
  Since the probes of 2026-11-10: `token`, one event per token a decode step yields with the
  token's id as its argument (the span's count is the tokens written, and a level 3 trace gives
  them in order); `mtp.draft` and `mtp.verify` around the two halves of an MTP round;
  `engine.init` and `session.create`. The counter `kv_copy_bytes` adds the bytes each KV copy
  moves (with Metal 4 `copyBufferMappingsFromBuffer:` helper snapshots on v2 ring-KV, only local
  rings are blitted: ~12–15 MB instead of a full-window copy).
  Signposts go to an `os_log` handle with the DynamicTracing category (about 11 ns per pair until
  Instruments records; PointsOfInterest costs about 410 ns per signpost even when nothing
  records, so only turn-level spans use it).
- **GPU time**: every command buffer gets a completion handler that adds `GPUEndTime -
  GPUStartTime` to the current run's signature, plus `kernelStartTime` scheduling latency. Level
  4 adds per-kernel timing: one encoder per kernel with stage-boundary timestamps (the only
  sampling point Apple GPUs support on the classic API), or Metal 4 counter heaps with precise
  timestamps where the queue is Metal 4.
- **Anomalies** (always on from level 1): a bounded ring of events with severity, code, two
  integers and a short static message, mirrored to `os_log` at error/fault level. Watchdogs:
  waits past 1 s and 5 s with the waiting session and task states, sessions destroyed with
  pending tasks, streams dropped without cancel, invalid sampled tokens, sampler or mask-runner
  CPU fallbacks, GPU family not recognized (ML Drift falls back to Apple1), weight-cache load
  failures and weights prepared on every load, stale program-cache entries, GPU models split
  across partitions or left on the CPU, KV context switches, constraint creation skipped.
- **Memory** on every snapshot: `phys_footprint`, its peak, `ledger_tag_graphics_footprint`,
  `os_proc_available_memory` (iOS). The Metal side reads `MTLDevice.currentAllocatedSize`, the
  thermal state and low power mode every 64th command buffer: the snapshot's `memory` carries
  the last reading and its peak (`gpu_allocated_mb`, `gpu_allocated_peak_mb`), and `device` the
  thermal state and low power. They're the last reading, not the state at the snapshot.
- **Numerical probes** (logits on the CPU sampler path, done): NaN and Inf count (an error
  anomaly), flat rows, max logit, top-1 margin, near ties (top two within 1/16: four FP16 steps
  for logits in [16, 32), where FP16 rounding can flip the argmax), and from level 2 the entropy
  and top-1 probability at temperature 1. Model and MTP drafter logits have separate gauges.
  Levels 1–3 check the first row of every 32nd sampler call (0.38 ms for 262,144 logits, about
  0.1% of decode); level 4 checks every row. Next: stats kernels on the GPU (ML Drift optional
  nodes on rotary angles and attention scores) for FP16 saturation and precision loss.
- **Gauges**: value distributions (count, mean, min, max, and the 10th, 50th and 90th
  percentiles of a uniform sample of up to 4,096 values), in the snapshot's `gauges`.
- **Kernel identity** (always, no per-dispatch cost): ML Drift sets pipeline labels from node
  names; debug groups per kernel at level 3+.

### Interfaces

- C (LiteRT-LM, `c/telemetry.h`): `litert_lm_telemetry_set_level`, `_get_level`,
  `_snapshot_json` (counters, span aggregates, GPU time, anomalies, memory),
  `_reset`, `_write_trace(path)`; the environment variable `AGELLO_TELEMETRY=<level>` sets the
  level at startup.
- Swift: `LiteRTLM.Telemetry.level`, `.snapshot()`, `.reset()`, `.writeTrace(to:)`.
- agello-bench: each record carries the telemetry delta of its case; `telemetry.py` turns a run
  into a report (per-step host and GPU breakdown, achieved bandwidth against the weights read per
  token, top kernels, anomalies, regressions against a baseline run).
- Agello's harness: `DeviceHarness.record` adds the delta for each scenario, so every iPhone or
  Mac run carries the engine's own account of what happened.

## Using it

| What | How |
| --- | --- |
| Level | `AGELLO_TELEMETRY=<0-4>` (environment, at load), `litert_lm_telemetry_set_level`, `LiteRTLM.Telemetry.level`, agello-bench `--telemetry N` |
| Snapshot | `litert_lm_telemetry_snapshot_json`, `Telemetry.snapshot()`, or `AGELLO_TELEMETRY_OUT=<path>` for any binary at exit |
| Trace | level 3, then `litert_lm_telemetry_write_trace` / agello-bench `--trace FILE`; open in Perfetto or `chrome://tracing`. Each thread keeps its last 262,144 events (8 MB, allocated at level 3). LiteRT's own trace sites, a few hundred events per decode step, stay out of the trace unless `AGELLO_TELEMETRY_TRACE_ALL=1`, so a 1,500-token reply fits whole (50,699 events on the M6); their aggregates are kept either way |
| Logits | `AGELLO_TELEMETRY_LOGITS_OUT=<path>` (at load): one JSON line per row the sampler reads, with the run that read it (`decode`, `verify`, `drafter`), the sampler call, the row, and the eight best token ids with their logits. It probes every row, about a quarter millisecond each: for comparing two builds' logits, not for speed runs |
| Instruments | level 2+: subsystem `com.patricedery.agello.litertlm`, category DynamicTracing (os_signpost intervals) |
| Per case | agello-bench resets before each case and stores the snapshot in the record's `telemetry`; the engine load's goes to `<out>.load-telemetry.json` |
| Report | `python3 tools/agello-bench/telemetry.py results/run.jsonl` (also a load snapshot, `results/run.jsonl.load-telemetry.json`, or an Agello `harness-report.json`) |
| GPU graphs | `AGELLO_TELEMETRY_GRAPH_DIR=<dir>` (works at **any** telemetry level, independent of level 4 timing): ML Drift writes each initialized graph there, built or restored from the program cache, as `gpu-graph-<n>-<kernels>-nodes.json` (kernel names, work groups, FLOPs, input and output shapes); `telemetry.py --graph <dir>` summarizes them |
| Kernel shapes | `AGELLO_KERNEL_SHAPES=1` (at level 4): keys per-kernel GPU timings by output and input tensor shapes (`BxHxWxC`), splitting one kernel kind by layer type (e.g. local ring attention vs global attention). Dynamic spans cap at 512, so use for one graph/case at a time |

Build notes: LiteRT-LM's probes (`@agello_telemetry`), LiteRT's probes (`agello/litert`), and ML
Drift's probes (`agello/ml_drift`) all live in this single repository (`PowerBeef/agello-litert-lm`,
branch `main`). Edit any probe in place and rebuild with `agello/build-xcframework.sh` (or
`--mac-only` for Mac CLI runs) — no `--override_repository` or re-pin step is needed.

Overhead on the M6 (speed sweep, ABAB, same build): levels 1 and 2 within ±0.3% of level 0 for
prefill and decode, replies identical.

### Metal waits

The Metal delegate busy-spins a CPU core for every GPU wait by default when benchmark mode is on.
Asking for passive waits, with command buffers still committed as they're encoded (LiteRT
`77ed9cab`), measured −0.3% decode and +0.4% prefill against busy-waiting (within noise), and
2.0 ms of process CPU per generated token instead of 29.3. Since 2026-11-11 (Agello `6db56c6`),
Agello leaves `ExperimentalFlags.enableBenchmark = false` at every load and reads reply stats from
`LiteRTLM.Telemetry`, so the shipping app runs with passive GPU waits (`reference/facts.md` F-019).
In `agello-bench`, pass `--no-benchmark-mode` to match the app.

## Historical first readings (2026-10-14, M6, 2K prompt, 128 generated, kv 8192)

> [!NOTE]
> The tables below are the original 2026-10-14 baseline measurements taken before bounded attention,
> zero-copy NEON FP16 greedy sampling (`0` host read copies), `MTLBinaryArchive`, MSL 4.1
> `agello_flash_attention`, `verify_8`, ring/sparse KV, and GPU Vision landed. Kept for historical
> context; current iPhone 17 Pro and M6 figures live in `iRT/docs/BACKEND.md` and `iRT/Vendor/README.md`.

| | Without MTP | With MTP (4 tokens per round on this text) |
| --- | --- | --- |
| Step | 21.7 ms per token, of which the model run 21.0 ms (it doesn't return early: async falls back to sync on Metal) | 7.5 ms per token |
| GPU | 20.7 ms per decode run in ~9 command buffers; 95% busy over the turn; 106 GB/s effective weight bandwidth | verify 22.6 ms per run; drafter 0.66 ms per run; 83% busy; 98 GB/s |
| Host per token | sampling 0.50 ms (CPU sampler: measured 2026-10-14 before direct logits reads and zero-copy NEON FP16 greedy sampling), parameter fill 0.24 ms, embedding lookups 0.14 ms | drafter steps' host work between GPU runs (CPU sampling, readbacks, lookups) leaves the GPU idle 17% |
| Prefill | 2,049 tokens in 3 chunks, 932 ms, 98% GPU | same |
| Other | 2 CPU model runs per prompt token (embedder, per-layer embedder): 4,483 inferences per case; one tensor buffer allocated per decode step; token history copied for every mask fill | |
| Load (warm cache) | delegate graph conversion ~950 ms of a 1.1 s load | |

### GPU kernels (2026-10-14, level 4, 2K prompt, 64 generated)

Shares of summed kernel time (each kernel in its own encoder, so absolute times run high):

| | T = 8,192 | T = 32,768 |
| --- | --- | --- |
| Decode | int4 matmuls 68%, int8 matmuls 11%, LM head (int2) 6%, RMS norms 4%, softmax 3.6% | int4 matmuls 61%, **softmax 12.7%** (3.1 ms per token, 0.8 ms at 8K) |
| Prefill | Neural Accelerator int8 convolutions 39%, mask select 11.5%, softmax 16.4%, int4→int8 weight conversion 8.4%, activation quantization 3.7% | **mask select 23%, softmax 37%**, convolutions 20.5% (same work as at 8K), weight conversion 4.4% |

The matmul work doesn't depend on T; the select and the softmax scale with it. Bounding them to
the filled length (T4's first step, `_archive/tracks/T4-attention.md`) lifted prefill +140% at 32K
and +37.5% at 8K, and decode +14% at 32K.

### Load, delegation and host copies (2026-10-14, 2K prompt, 64 generated, MTP)

| | Reading |
| --- | --- |
| Delegation | 6 compiled models, 20,392 nodes: 20,335 on the GPU in 5 partitions (one per GPU subgraph: no CPU round trips inside a model), 3 in XNNPack, 54 on TFLite's own CPU kernels (the CPU embedders) |
| Caches | Cold load: 5 program-cache misses, weight cache missed, built, flushed and reloaded (5 misses, 5 hits), 8.3 s. Warm: 5 program-cache hits, 1.1 s, of which delegate graph conversion 0.5 s |
| GPU graphs | Decode 1,766 kernels (15.9 GFLOP), verify 1,817 (63.5), prefill-1024 1,678 (6,318), prefill-128 1,632 (790), drafter 164 |
| Host copies | 113 GPU → host reads and 274 host → GPU writes per case (4.9 MB per generated token, prefill included; since eliminated on greedy decode by direct locks + zero-copy NEON FP16 greedy sampling: `0` host read copies) |
| Async | The executor asks for asynchronous runs; the Metal backend has no post-dispatch event, so every run completes synchronously (fact `gpu_async`) |

Direct non-packed buffer reads/writes (`_archive/tracks/T5-unified-memory.md`) lifted MTP decode
+13.6% and sped up raw `kv_clone` snapshots 7×; subsequent zero-copy NEON FP16 greedy sampling
(`73000de4`) eliminated host logits read copies entirely (`0.0 KB/tok`).

### Logits (2026-10-14, level 4, 2K prompt, 128 generated, MTP)

| | Model (129 positions) | Drafter (96 drafts) |
| --- | --- | --- |
| NaN / Inf | 0 | 0 |
| Max logit | mean 23.8, max 29.3 (under Gemma's soft cap of 30) | mean 17.5 |
| Top-1 margin | min 1.03, p10 8.9, p50 12.6: no near ties | min 0.13, p50 12.3 |
| Top-1 probability | p10 1.000, min 0.27 | p10 0.998, min 0.52 |
| Entropy, nats | p90 0.004, max 4.0 | p90 0.019, max 3.2 |

## Status

1. Done (2026-10-14): core library, C and Swift APIs, LiteRT-LM probes (turns, executor, MTP,
   sampler, execution manager watchdogs), LiteRT's trace macros mapped to spans, command-buffer
   GPU time, `agello-bench` integration and `telemetry.py`; overhead within noise.
2. Done (2026-10-14): pipeline labels, GPU family probe and per-kernel timing; GPU graph dumps
   (`AGELLO_TELEMETRY_GRAPH_DIR`); LiteRT buffer lock/copy counters, cache events, delegation
   coverage, numerical logits probes (`ProbeLogits`, `ProbeLogitsHalf`, `ProbeLogitsHalfDSHWBC4`).
3. Done (2026-11-10 to 2026-11-24): `token`, `mtp.draft`/`mtp.verify`, `engine.init`,
   `session.create`, `runs_verify8`/`run.verify8`/`gpu.verify8` spans & counters,
   `AGELLO_KERNEL_SHAPES=1` shape-keyed level-4 kernel attribution, and intra-MTP cancel telemetry.
