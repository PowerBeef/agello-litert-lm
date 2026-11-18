# Tools

> **Scope:** the tools for inspecting model files (`litertlm_peek`, `model_info_cli`,
> `inspect_e4b.py`), rewriting them (`patch_rope.py`), running the engine from the command line
> (`litert_lm_main` and its siblings), measuring builds (agello-bench, `compare.py`,
> `telemetry.py`), our engine telemetry (levels, environment variables, graph dumps), the GPU
> simulation switch, and the scripts behind this reference (`tools/agello-docs`). Exact commands and
> real options. **Out of scope:** building the framework ([build-and-release.md](litert-lm/build-and-release.md));
> Agello's device harness (Agello's `agello-device-testing` skill); every flag's row
> ([generated/cli-flags.md](generated/cli-flags.md)) and every setting ([configuration.md](configuration.md));
> keeping the reference current ([maintenance.md](maintenance.md)).
> **Pinned to:** `main` `b7b856a0`: LiteRT-LM at the root, LiteRT in `agello/litert`, ML Drift in
> `agello/ml_drift`, and `tools/agello-bench`, `tools/model-surgery` and `tools/agello-docs` (in
> `main` since 2026-11-15); Google's base `df6df5b8` for the tools our tree removed; Agello at
> `93122b647a` and `a691212`.
> **Applies to:** Apple (macOS for the command-line tools; iOS through Agello's harness).
> **Last verified:** 2026-11-17

## Summary

Google's own tools: `litertlm_peek` and `model_info_cli` read a `.litertlm` file without loading a
model, and `litert_lm_advanced_main` runs one from the command line with most of the 94 flags
(`litert_lm_main` reads only seven). Our tree removed their sources in the prunes of 2026-11-11
(`7e4d9b8d`, `edb2c7db`), so they're read and built in Google's tree; Agello's `agello` command line
(its `docs/CLI.md`) runs our runtime on the Mac instead. Ours, in `tools/`: `inspect_e4b.py` maps a
graph's attention, `patch_rope.py` rewrites the rotary-position math (the #3913 fix behind the file
Agello ships), and agello-bench measures a build (an "arm") with fixed cases, read back with
`compare.py` and `telemetry.py`. Our runtime commits add engine telemetry (levels 0 to 4, set by
`AGELLO_TELEMETRY`) and a switch that makes ML Drift pick another Apple GPU's kernels
(`AGELLO_SIMULATE_APPLE_GPU`). `tools/agello-docs` generates this reference's tables, makes and
checks its citations, and moves them after a re-pin. agello-bench's README was fixed on 2026-10-16
(C6) but still names our framework by the former `agello-next` branch, and the telemetry note's
level table puts graph dumps at level 4 when they work at any level.

## Concepts

| Tool | Where | Needs | Use it to |
| --- | --- | --- | --- |
| `litertlm_peek_main` / `litert-lm-peek` | Google's `python/litert_lm_builder` (our tree removed it, `7e4d9b8d`) | Bazel in Google's tree, or the `litert-lm-builder` PyPI package | print a file's header, metadata and sections; dump sections |
| `model_info_cli` | Google's `schema/model_info` (our tree removed its source, `edb2c7db`) | Bazel in Google's tree | print `ModelInfo`: modalities, budgets, backends, context |
| `inspect_e4b.py` | `tools/model-surgery` | tools venv (`ai_edge_litert`, `flatbuffers`, `litert_lm_builder`) | signatures, op histogram, composites, per-softmax masks and caches |
| `patch_rope.py` | `tools/model-surgery` (since `f2e7afa0`) | tools venv, `numpy` | rewrite rotary positions (#3913) |
| `litert_lm_main`, `litert_lm_advanced_main`, `embedding_litert_lm_main` | Google's `runtime/engine` (our tree removed their sources, `edb2c7db`) | Bazel in Google's tree | run a model from the command line |
| agello-bench | `tools/agello-bench` | Swift 6, macOS 27 | measure an arm with fixed cases; JSONL records |
| `compare.py`, `telemetry.py` | same | Python 3 standard library | compare arms; explain records with telemetry |
| Engine telemetry | our `agello/telemetry`, LiteRT and ML Drift hooks | our builds | counters, spans, GPU time, anomalies, traces |
| `AGELLO_SIMULATE_APPLE_GPU` | our ML Drift and LiteRT | our builds | run another Apple GPU family's kernel paths |
| `tools/agello-docs` | `tools/agello-docs` (`/Users/patricedery/Coding projects/LiteRT-LM`) | Python 3, git; tools venv for model tables | regenerate `generated/`, make citations, check the reference |

The tools venv is `~/.venvs/agello-tools`, named in `inspect_e4b.py`'s usage line and in
`tools/agello-docs/regen.sh`.

## How it works

### Reading a model file

#### `litertlm_peek`

Google's peek tool prints the header, the system metadata and every section of a `.litertlm` file,
decoding the metadata protos as text, and can write each section to a folder.
<!-- c:LM-FMT-036 --> Its only options are `--litertlm_file` (required) and `--dump_files_dir`
([`python/litert_lm_builder/litertlm_peek_main.py:37-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_peek_main.py#L37-L51)).
The PyPI package installs it as the console script `litert-lm-peek`, next to `litert-lm-builder`
([`python/litert_lm_builder/BUILD:285-290`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/BUILD#L285-L290)).

Our tree removed it with the rest of `python/` (prune `7e4d9b8d`), so run it in Google's tree or
from the PyPI package:

```sh
# in a checkout of Google's LiteRT-LM
bazelisk run //python/litert_lm_builder:litertlm_peek_main -- \
  --litertlm_file="$HOME/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm"
# or, with the PyPI package in the tools venv
litert-lm-peek --litertlm_file ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm --dump_files_dir /tmp/e4b-sections
```

Dumping sections writes the whole file out again (3.66 GB for E4B): use a scratch folder and delete
it. What the sections are is on [file-format.md](litert-lm/file-format.md); the tables for our files
are in [generated/models/](generated/models/e4b-ours.md).

#### `model_info_cli`

`//schema/model_info:model_info_cli` takes one argument, the model path, calls `GetModelInfo` and
prints its report; with no argument or on an error it prints a message and exits 1
([`schema/model_info/model_info_cli.cc:24-48`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info_cli.cc#L24-L48),
[`schema/model_info/BUILD:72-79`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/BUILD#L72-L79)).

Our tree removed `model_info_cli.cc` (prune `edb2c7db`), though `schema/model_info/BUILD` still
declares the target, so build it in Google's tree:

```sh
# in a checkout of Google's LiteRT-LM
bazelisk run //schema/model_info:model_info_cli -- "$HOME/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm"
```

The report covers modalities, thinking and tool support, speculative decoding, vision budgets,
default sampler parameters, backends per modality, the minimum runtime version and the maximum
context. <!-- c:LM-FMT-037 --> Two things to know when reading it for Gemma 4: thinking and
tool-calling support come from `LlmMetadata` fields the published files don't set, so both read
false <!-- c:MOD-E2B-048 -->; and with no `max_num_tokens` in the metadata, the maximum context is
found by reading the main graph: our build reads growing prefixes of the section (16 MiB to 1 GiB)
until one verifies as the graph's flatbuffer, then the whole section, refused above 2 GiB, so the E2B and E4B files
both report 32,003 (ours: `cc97925b`). <!-- c:MOD-E2B-049 -->
The same reader is in the C API (`model_info.h`) and the Swift `ModelInfo`
([swift-api.md](litert-lm/swift-api.md#modelinfo)).

#### `inspect_e4b.py`

`tools/model-surgery/inspect_e4b.py` reads one TFLite section of a `.litertlm` file and one of its
signatures and prints the signature's inputs and outputs, an op histogram, every composite with its
attributes, and for each `SOFTMAX` the ops and graph inputs behind its mask and the KV caches feeding
it; `--layers` prints one row per attention instead, `--names` the tensor names around each softmax.
Defaults: section `tf_lite_prefill_decode`, signature `decode`
([`tools/model-surgery/inspect_e4b.py:1-11`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/model-surgery/inspect_e4b.py#L1-L11),
[`tools/model-surgery/inspect_e4b.py:151-157`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/model-surgery/inspect_e4b.py#L151-L157)).

```sh
~/.venvs/agello-tools/bin/python tools/model-surgery/inspect_e4b.py \
  ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm --signature decode --layers
~/.venvs/agello-tools/bin/python tools/model-surgery/inspect_e4b.py \
  ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm --section tf_lite_mtp_drafter --signature mtp_drafter
```

`tools/agello-docs/gen_model_tables.py` imports it for the per-layer attention table in
`generated/models/` ([below](#toolsagello-docs)).

### Rewriting rotary positions: `patch_rope.py`

`patch_rope.py` (ours, added in `f2e7afa0` on the old `agello` branch; in `main` since `163873b8` and
unchanged) rewrites the
rotary-angle computation of the `tf_lite_prefill_decode` and `tf_lite_mtp_drafter` sections and
writes a new file whose later sections move by 16 KiB-aligned deltas, with the header's offsets
updated. Its modes: a float shift after the rotary `CAST` (`<shift>`), the same shift in INT32 before
it (`<shift> --int`), and `--split`, which computes positions as block × h + l with exact phases up to
32,767; `--block` takes 512, 1024 or 2048 (default 2048). <!-- c:X-TLS-005 -->

```sh
~/.venvs/agello-tools/bin/python tools/model-surgery/patch_rope.py \
  ~/Models/agello/gemma-4-E4B-it.litertlm --split /tmp/gemma-4-E4B-it-fp16-rope.litertlm --block 512
shasum -a 256 /tmp/gemma-4-E4B-it-fp16-rope.litertlm   # expect 6353a233...f655c, then delete it
```

Run on Google's E4B file (revision `2eee7ac3`), `--split --block 512` reproduces the file Agello ships
byte for byte, and `--block 2048` the first published revision. <!-- c:MOD-E4B-069 --> With 512-token
blocks, positions are `512h + l`: `l` goes through the original `MUL`, `SIN` and `COS`, the phase of
`512h` comes from small precomputed tables. <!-- c:MOD-E4B-065 --> The tables cover positions up to
32,767, which LiteRT-LM never passes with this file. <!-- c:MOD-E4B-073 --> It patches only
`tf_lite_prefill_decode` and `tf_lite_mtp_drafter`, so it can't patch a `-gpu` file.
<!-- c:MOD-E2B-060 --> Why the fix exists, and its evidence, is on
[models/gemma4-e4b.md](models/gemma4-e4b.md#rotary-positions-and-litert-lm-3913).

`tools/model-surgery` also holds the later model tools, each described in its own header:
`ring_kv.py` (ring-buffer local KV caches, `290a1eec`), `window_128k.py` (the 131,072-token window,
`72b715e3`), `embedder_tables.py` (`ffc639ef`) and `verify_n.py` with `check_verify_n.py` (a longer
verify signature, `0a32b7cf`).

### Running the engine from the command line

Google's `runtime/engine` has three binaries: `litert_lm_main`, `litert_lm_advanced_main` and
`embedding_litert_lm_main`. Our tree removed their sources (`litert_lm_main.cc`,
`litert_lm_advanced_main.cc`, `litert_lm_lib.cc`, `shared_flags.cc` and the embedding ones) in the
prune `edb2c7db` and kept the BUILD targets, so they build only in Google's tree; what follows is
Google's code at our base. Together they define 94 flags, 61 of them in `shared_flags.cc`, every one
listed with its type, default and help text in [generated/cli-flags.md](generated/cli-flags.md). The
two LLM binaries link the shared flags but use them differently: `litert_lm_main` reads only
`--backend` (default `gpu`), `--model_path`, `--input_prompt`, `--input_prompt_file`,
`--visual_token_budget`, `--num_cpu_threads` and `--enable_ynnpack`, so the other shared flags are
accepted and do nothing there, while `litert_lm_advanced_main` reads the shared flags into its
settings and has its own `--backend`, whose default is `cpu`
([`runtime/engine/litert_lm_main.cc:92-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/litert_lm_main.cc#L92-L151),
[`runtime/engine/litert_lm_advanced_main.cc:55-56`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/litert_lm_advanced_main.cc#L55-L56),
[`runtime/engine/litert_lm_advanced_main.cc:257-286`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/litert_lm_advanced_main.cc#L257-L286)). <!-- c:X-TLS-004 -->
To test anything beyond a prompt on the default settings, use `litert_lm_advanced_main` and pass
`--backend=gpu`. The flags that matter for Agello's questions, all read by the advanced binary:

| Flag | Default | What it does |
| --- | --- | --- |
| `--model_path`, `--backend` | `""`, `cpu` | the file and the main backend |
| `--input_prompt`, `--input_prompt_file`, `--multi_turns` | | what to send; `--multi_turns` asks for more turns |
| `--max_num_tokens` | `0` | the allocated context; 0 takes the file's value, else a size from the prompt length (4,096 for an empty hint) <!-- c:LM-ENG-008 --> |
| `--enable_speculative_decoding` | `false` | MTP with the file's drafter |
| `--force_f32` | `false` | FP32 activations: the #3913 workaround, at roughly half the decode speed on the iPhone <!-- c:MOD-E4B-082 --> |
| `--benchmark`, `--benchmark_prefill_tokens`, `--benchmark_decode_tokens` | `false`, `0`, `0` | benchmark mode: timings, and (in our build) the active Metal wait <!-- c:LM-ENG-090 --> |
| `--async` | `true` | asynchronous execution |
| `--sampler_backend` | `""` | the sampler backend |
| `--cache_dir`, `--disable_cache`, `--disable_weight_cache`, `--disable_gpu_program_cache` | `""`, `false` | program and weight caches ([ml-drift-delegate.md](litert/ml-drift-delegate.md#what-is-cached-on-disk-keyed-by-what)) |
| `--gpu_enable_metal_residency_set` | `false` | the Metal residency set (Apple only) |
| `--gpu_external_tensor_mode` | `false` | external-tensor mode for the GPU executor |
| `--configure_magic_numbers` | `true` | rewrite the file's 32003 context dimension at load |
| `--visual_token_budget`, `--max_num_images` | `-1`, `1` | image tokens per image, images per prompt |
| `--use_session`, `--min_log_severity`, `--prefill_chunk_size` | `false`, `-1`, `-1` | the session API instead of `Conversation`; log level; CPU prefill chunks (no effect on the GPU) <!-- c:LM-EXE-088 --> |

Google's v0.18.0 release (2026-10-06) has `litert_lm_main` as an asset, `litert_lm_main.macos_arm64`
<!-- c:LM-BLD-053 -->, built with `--define=litert_runtime_link_mode=dynamic` <!-- c:LM-BLD-052 -->,
so it needs the `prebuilt/` shared libraries beside it to use a GPU. <!-- c:LM-BLD-059 --> Our tree's
`runtime/engine/BUILD` is Google's: both LLM targets reach `//runtime/core:engine_advanced_impl`
through `:engine_impl_selected` (`litert_lm_advanced_main` by way of `:litert_lm_lib`), which in our
build selects the statically linked Metal accelerator on iOS and macOS unless the link mode is
dynamic, as our framework does; but with their sources gone, neither target builds from our tree
([`runtime/engine/BUILD:100-106`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/BUILD#L100-L106),
[`runtime/engine/BUILD:186-194`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/BUILD#L186-L194),
[`runtime/engine/BUILD:252-279`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/BUILD#L252-L279),
[`runtime/engine/BUILD:329-333`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/BUILD#L329-L333)). <!-- c:X-TLS-003 -->
To run our runtime from the command line on the Mac, use Agello's `agello` tool (built with SwiftPM
from Agello's `AgelloCLI/`, described in Agello's `docs/CLI.md`), which loads our framework through
the app's own engine code; it is a development tool, never shipped.

### agello-bench

agello-bench lives in `tools/agello-bench`, in `main` since 2026-11-15 (`163873b8`, from the old
`agello` branch's tip `61c03d8e`, with three later fixes to `compare.py`, `telemetry.py` and the
README).
`BenchKit` holds the cases, prompts, scoring, machine record and runner; the `agello-bench`
executable loads one engine and runs cases on it.

#### Arms

An arm is a LiteRT-LM checkout: its Swift wrapper and the binary its `Package.swift` names. By
default the package builds against the checkout two folders up; `LITERTLM_PACKAGE` points it at
another one, whose folder must be named `LiteRT-LM`
([`tools/agello-bench/Package.swift:6-9`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Package.swift#L6-L9)). In our
tree that default is the fork's root on `main`, whose `Package.swift` names
`dist/CLiteRTLM.xcframework` <!-- c:LM-BLD-047 -->, so a default build measures the local build in
`dist/`, on the Metal path natively on the Mac. <!-- c:X-TLS-006 --> The release Agello ships has
the same single binary target; its release script writes the release's URL and checksum into the
published manifest. <!-- c:LM-BLD-049 --> The README says the same of the Mac (C6, resolved).
<!-- c:LM-BLD-063 --> A second arm is another build of ours: another checkout, or the same one
after rebuilding `dist/`.

```sh
cd "/Users/patricedery/Coding projects/LiteRT-LM/tools/agello-bench"
swift build -c release --scratch-path .build-next                       # this checkout's dist/ (build it first)
LITERTLM_PACKAGE="<another checkout>/LiteRT-LM" \
  swift build -c release --scratch-path .build-base                     # another build of ours
.build-next/release/agello-bench env                                    # machine and runtime SHA-256
```

#### Running cases

`run` needs `--model` and at least one `--suite` or `--case`. Its options, from the usage text:
`--backend gpu|cpu`, `--fp32`, `--kv N` (default 8,192), `--mtp`, `--cache DIR`, `--cold`,
`--repeat N`, `--warmup N`, `--label NAME`, `--out FILE`, `--hash-model`, `--no-benchmark-mode`,
`--telemetry N` and `--trace FILE`
([`tools/agello-bench/Sources/agello-bench/main.swift:5-28`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L5-L28)).
The README lists the same options since 2026-10-16
([`tools/agello-bench/README.md:33-39`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/README.md#L33-L39)); `--trace` raises
the telemetry level to at least 3
([`tools/agello-bench/Sources/agello-bench/main.swift:150-151`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L150-L151)). <!-- c:X-TLS-007 -->

```sh
.build-next/release/agello-bench run --model ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm \
  --suite suites/short.json --kv 8192 --mtp --label next --cache ~/Models/agello/cache/bench \
  --warmup 1 --telemetry 2 --no-benchmark-mode --out results/next-short.jsonl
python3 compare.py results/base-short.jsonl results/next-short.jsonl --base base
python3 telemetry.py results/next-short.jsonl
```

How a run is set up:

- **Benchmark mode by default.** The runner opts into experimental APIs, sets
  `ExperimentalFlags.enableBenchmark` to the run's benchmark mode (on unless `--no-benchmark-mode`),
  raises telemetry to level 2 when benchmark mode is off, and sets speculative decoding from `--mtp`
  ([`tools/agello-bench/Sources/BenchKit/Bench.swift:55-58`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/BenchKit/Bench.swift#L55-L58)).
  Benchmark mode keeps LiteRT-LM's active Metal wait; Agello has run without benchmark mode since
  2026-11-11, with the passive wait, so pass `--no-benchmark-mode` to measure what the app runs.
  <!-- c:RT-RUN-028 -->
- **Telemetry on any build.** `EngineTelemetry` looks the `litert_lm_telemetry_*` functions up at
  run time, so the runner also loads a build without them (Google's), where every call does nothing
  ([`tools/agello-bench/Sources/BenchKit/EngineTelemetry.swift:4-18`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/BenchKit/EngineTelemetry.swift#L4-L18)).
- **The load's own snapshot.** With `--out`, the engine load's telemetry is written to
  `<out>.load-telemetry.json`
  ([`tools/agello-bench/Sources/agello-bench/main.swift:180-182`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L180-L182)); load-time counters (cache
  hits, delegation) read 0 in the per-case records, as [generated/engine-run.md](generated/engine-run.md)
  shows, so look for them there.
- **Per-build caches.** `--cache DIR` uses `DIR/<build id>` when the runtime reports a build id and
  deletes the other builds' folders there (names of 12 lowercase hex digits), and
  `AGELLO_BENCH_HOLD=<seconds>` keeps the process alive after the last case
  ([`tools/agello-bench/Sources/agello-bench/main.swift:88-103`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L88-L103),
  [`tools/agello-bench/Sources/agello-bench/main.swift:187-189`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L187-L189));
  both came with the old branch's tip, after `85b1a776`. <!-- c:X-TLS-008 --> The program cache key has no build identity, which is why. <!-- c:RT-DLG-048 -->

Cases and suites (digit copies, `speed`, `engine-benchmark`, `helper-turn`) and the record fields
are described in the README itself; the protocol for comparing arms (ABAB order, one warm-up, five
rounds for speed, a change inside the noise band isn't a change) is there too. Results of record
are copied into `docs/agello/results/<device>/<date>/bench/` with a README per folder; the bench's
own `results/` folder is git-ignored.

#### Reading results

`compare.py` takes record files, `--base LABEL` and `--metrics` (default `prefillTPS decodeTPS
ttft`), leaves warm-ups out, and prints digit-copy fidelity, identical replies against the base,
per-case medians and interquartile ranges, each arm's change with a 95% bootstrap interval, and the
noise band ([`tools/agello-bench/compare.py:1-13`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/compare.py#L1-L13),
[`tools/agello-bench/compare.py:52-54`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/compare.py#L52-L54)).
`telemetry.py` reads bench records (`--label`, `--case`, `--weights-gb`), an Agello harness report,
a `.load-telemetry.json` file, or a folder of GPU graph dumps (`--graph DIR`), and explains where
each token's time went
([`tools/agello-bench/telemetry.py:1-9`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/telemetry.py#L1-L9)).

#### Where the README is stale

C6 is resolved: since `72bf0f97` (2026-10-16) the README calls `research/3913` a tag, lists every
suite and every run option. <!-- c:LM-BLD-064 --> <!-- c:X-TLS-007 --> What is still stale: it
names our framework by the former branch, `agello-next` <!-- c:LM-BLD-063 -->, and it still
describes arms as Google's release binaries, a nightly or our builds, while we no longer build or
measure Google's runtime (Agello's `docs/BACKEND.md`). Fixing it is a commit on `main`, outside this
reference.

### Telemetry

Our runtime builds carry low-overhead probes in LiteRT-LM (`agello/telemetry`, ours `4aed9274`),
LiteRT and ML Drift, the release Agello ships included (its ML Drift commits run to 2026-11-14); our
build's iOS slice defines 8 Agello telemetry symbols. <!-- c:MD-ARC-053 -->

#### Levels

| Level | Name | Adds |
| --- | --- | --- |
| 0 | off | nothing |
| 1 | counters | counters and anomalies (the default) |
| 2 | spans | span aggregates (count, total, maximum, histogram) and signposts; per-command-buffer GPU time |
| 3 | trace | per-thread span ring buffers, written as a Chrome/Perfetto trace; a debug group per kernel |
| 4 | deep | diagnostic modes that change timing: per-kernel GPU timing, numerical probes |

The header defines these levels; the library starts at level 1 unless `AGELLO_TELEMETRY` sets
another, clamped to 0..4. The counter, span, gauge, anomaly and logit probes read the level with one
relaxed atomic load before recording anything; engine facts (`SetFact`) are stored at every level,
0 included ([`agello/telemetry/agello/agello_telemetry.h:18-26`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.h#L18-L26),
[`agello/telemetry/agello/agello_telemetry.cc:623-630`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.cc#L623-L630)). <!-- c:X-TLS-001 --> <!-- c:LM-CAPI-040 -->
On the GPU side, ML Drift counts command buffers from level 1 and records their GPU time from
level 2 <!-- c:MD-MTL-041 -->, pushes a debug group per node from level 3, and at level 4 encodes
each kernel in its own encoder <!-- c:MD-MTL-032 -->, recording each kernel's GPU time as a span
named `gpu.k.<run>:<kernel>`, followed by the kernel's output and input shapes when
`AGELLO_KERNEL_SHAPES=1` (ours: `a5db0fc4`). <!-- c:MD-MTL-033 --> Level 4 changes the timing it measures.

#### What it records

The span names are fixed in the header: turn-level (`turn.prefill`, `turn.decode`), host steps
(`template`, `tokenize`, `task.queue_wait`, `embedding_lookup`, `mask_fill`, `param_fill`, `sample`,
`logits_readback`, `detokenize`, `stream_callback`, ...), runs (`run.prefill`, `run.decode`,
`run.verify`, `run.verify8`, `run.drafter`), KV work (`kv_clone`, `context_switch`), host copies
(`host.lock_read`, `host.unlock_write`) and GPU time per run kind
([`agello/telemetry/agello/agello_telemetry.h:117-166`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.h#L117-L166)).
`stream_callback` wraps each streamed callback call inside the decode loop of `Tasks::Decode` and
counts it in `stream_chunks`, so it covers everything above `Tasks` for those chunks; the last call
after the loop, which delivers text flushed from the detokenizer, is neither spanned nor counted ([`runtime/core/tasks.cc:808-815`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L808-L815),
[`runtime/core/tasks.cc:825-844`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L825-L844)). <!-- c:X-TLS-002 -->
Counters include runs per signature, host lock copies and bytes, command buffers, GPU waits, mask
fills, embedding lookups, sampler backend, KV copies, context switches, cache hits and misses,
delegation coverage and MTP rounds; [generated/engine-run.md](generated/engine-run.md) shows one
run's values. Our LiteRT adds delegation counters at compile time and a `partial_delegation`
warning <!-- c:RT-RUN-017 -->, and records the Metal device and ML Drift family as the `gpu` fact,
with an error anomaly for a GPU ML Drift doesn't recognize. <!-- c:RT-DLG-072 -->

#### Getting it out

| From | How |
| --- | --- |
| Environment | `AGELLO_TELEMETRY=<0-4>` at library load; `AGELLO_TELEMETRY_OUT=<path>` writes the snapshot at exit and `<path>.trace.json` at level 3 or more <!-- c:LM-CAPI-040 --> |
| C | `litert_lm_telemetry_set_level`, `_get_level`, `_snapshot_json` (free with `_free_string`), `_reset`, `_write_trace` ([`c/telemetry.h:30-55`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/telemetry.h#L30-L55)) <!-- c:LM-CAPI-039 --> |
| Swift | `Telemetry.level` (`.off` ... `.deep`), `snapshotJSON()`, `snapshot()`, `reset()`, `writeTrace(to:)` ([`swift/Telemetry.swift:21-62`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Telemetry.swift#L21-L62)) <!-- c:LM-SWIFT-044 --> |
| agello-bench | `--telemetry N`, `--trace FILE`; a snapshot per case in each record |
| Agello's harness | `AGELLO_TELEMETRY` passed to the app under test, default 2 (Agello `a691212`, `scripts/device-harness.sh:133`) <!-- c:X-TLS-009 --> |
| Instruments | level 2 or more: os_signpost intervals (subsystem `com.patricedery.agello.litertlm`, per `docs/agello/TELEMETRY.md`) |

#### GPU graph dumps

With `AGELLO_TELEMETRY_GRAPH_DIR=<dir>` set and not empty, ML Drift writes every GPU graph it
initializes, built or restored from the program cache, as `gpu-graph-<n>-<nodes>-nodes.json`, with
each node's name, work-group size, FLOPs and input and output shapes, at any telemetry level.
<!-- c:MD-MTL-034 --> ([`ml_drift/metal/inference_context.mm:679-682`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/metal/inference_context.mm#L679-L682))
`docs/agello/TELEMETRY.md`'s level table lists graph dumps under level 4; its own "GPU graphs" row
and the code say any level, and the code wins. `telemetry.py --graph <dir>` summarizes a folder of
dumps, and [generated/kernels-by-family.md](generated/kernels-by-family.md) is made from them.

### The GPU simulation switch

`AGELLO_SIMULATE_APPLE_GPU` (ours, ML Drift `ef08a67`) makes ML Drift build its Apple GPU info from
the variable instead of the Metal device name, so family and core-count decisions follow that GPU
while the hardware stays the real one. <!-- c:MD-ARC-016 --> <!-- c:MD-ARC-017 -->
([`ml_drift/common/gpu_info.cc:1373-1384`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/common/gpu_info.cc#L1373-L1384))
Our LiteRT adds the value to both cache keys, so real and simulated caches never mix.
<!-- c:MD-ARC-018 -->

```sh
# agello-bench on the Mac: the A17 Pro's kernel paths (Apple9)
AGELLO_SIMULATE_APPLE_GPU="Apple A17 Pro" .build-next/release/agello-bench run \
  --model ~/Models/agello/gemma-4-E4B-it-fp16-rope.litertlm --suite suites/short.json --kv 8192 \
  --label sim-a17 --cache ~/Models/agello/cache/sim --out results/sim-a17.jsonl
# Agello on the iPhone (from the Agello repository)
AGELLO_SIMULATE_APPLE_GPU="Apple A17 Pro" scripts/device-harness.sh inference
```

- **Use a full name.** The value is lowercased, underscores read as spaces, and matched against the
  device-name table, so "Apple A17 Pro" (or `Apple_A17_Pro`), "Apple M2" and "Apple M1" work, while
  "A17 Pro" matches nothing: our build then gives the unknown chip the device's own family (ours,
  G1 `e5e4c6eb`; Apple1 in the release Agello ships), so it simulates nothing, and raises
  `gpu_family_unknown`. <!-- c:APL-057 -->
- **The harness forwards it**, when set, as `TEST_RUNNER_AGELLO_SIMULATE_APPLE_GPU`, which `xcodebuild` passes
  to the test process without the prefix. <!-- c:APL-056 -->
- **Metal still describes the real device**: Apple's `supportsFamily` and the memory limit don't
  change. <!-- c:APL-058 --> The run's `gpu` fact shows the real device with the simulated family.
  <!-- c:MD-ARC-019 -->
- **Not newer than the hardware.** Simulating an Apple10 name on Apple9 hardware with iOS 26 or later
  would select `ConvAppleMPP` there; that hasn't been run. <!-- c:APL-060 -->
- Only our builds read it. Its effect on kernels is on
  [ml-drift/architecture.md](ml-drift/architecture.md#the-simulation-switch-ours) and
  [apple-platform.md](apple-platform.md#the-gpu-simulation-switch).

### `tools/agello-docs`

The scripts behind this reference are in `tools/agello-docs/` (in `main` since 2026-11-15; the
re-pin's additions, `repin.py` and `frozen/`, since 2026-11-18). `pins.json` names
every repository: local path, fork, branch, pin, upstream, base, and for LiteRT and ML Drift the
`subdir` they live in, the `imported` record (import commit, fork pin and the fork's commits) and
`base_changed` (the files the old forks changed, so every other file reads at Google's base);
`previous_pins` keeps the pins before a re-pin for `repin.py`; then the local and remote model files
and the Hugging Face config revisions. The other scripts read it. The procedure after a pin moves is
[maintenance.md](maintenance.md#after-a-re-pin-or-rebase).

| Script | Command | Does |
| --- | --- | --- |
| `cite.py` | `cite.py <repo> <path> <start> [<end>]` or `--symbol "<text>" [--context N]`; `--json`; `--at-base` | a Markdown permalink, or the ledger source record (with the SHA-1 of the cited lines); repos `litert-lm`, `litert`, `ml-drift`, `litert-lm-shipped` (paths relative to the component); `--at-base` cites Google's base for a file our tree removed |
| `check_refs.py` | `[--only <pages>...] [--no-tags] [--numbers] [--repin]` | permalinks (commit is a pin or base, file and lines exist), relative links and anchors (skipping `_work/`), ledger citations still hashing to their lines, claim tags; `--numbers` warns about untagged numbers; `--repin` prints `repin.py status` |
| `repin.py` | `status`; `apply`; `links [--apply]`; `labels [--apply]`; `show <group> [--ids A,B]` | after a re-pin: which citations moved unchanged and which claims need a verdict; re-cites the unchanged ones and merges; the same for the pages' permalinks; link texts whose line numbers no longer match their anchors; each listed claim with its old and new lines, for verifiers |
| `claims.py` | `check`, `list`, `show --ids`, `check-verdicts <group> <batch>`, `merge`, `check-applied`, `stats` | the claim and verdict workflow; `merge` applies verdict batches in name order, a later correction's text carried forward and `corrected_sources` taken from `verified` verdicts too |
| `regen.sh` | `regen.sh [--all \| --code]` | every generator, then `check_refs.py`; `--all` adds the Swift symbol graph and the per-family engine runs; `--code` is what moving the code pins changes (code tables, the symbol graph, the engine runs), without the model tables or the Hugging Face configs |
| `gen_model_tables.py` | `[--only <key>...]`, in the tools venv (`python -I`) | `generated/models/`: sections, signatures, ops, composites, weights, per-layer attention; remote files by HTTP range, or downloaded to `$SCRATCH` and deleted when their graphs are needed |
| `gen_hf_configs.py` | `[--update]` | `generated/hf/` and `hf-configs.md`; `--update` moves each Hugging Face pin to the current revision first |
| `capture_empirical.sh` | (no options) | runs agello-bench's `.build-next` arm once natively and with "Apple A17 Pro", "Apple M2" and "Apple M1" simulated, with graph dumps, then `gen_empirical.py`; scratch deleted |
| `gen_binaries.py` | `[<our CLiteRTLM>]` | `generated/binaries.md` for our binary (default `dist/`) from `nm`, `otool` and `strings`, then the frozen record `frozen/binaries-2026-10-16.md` (Google's v0.18.0 next to ours then), appended unchanged and never regenerated |
| `gen_swift_api.sh` | (no options) | the `LiteRTLM` symbol graph in a scratch folder, then `swift-api-inventory.md` |
| `gen_c_api.py`, `gen_protos.py`, `gen_flags.py`, `gen_patch_log.py`, `gen_versions.py` | (no options) | the C API inventory (clang AST), proto fields, CLI flags, our commits since Google's bases (✓ for the shipped release's), the version matrix |

Two limits to know: `gen_model_tables.py` reads at most 256 MiB of a section's graph, so it can't
parse E4B's 836.8 MB inline per-layer embedder; and the attention table it renders in
`generated/models/e4b-ours.md` has its header only, the rows being in the matching `.json`. Both are
noted in [maintenance.md](maintenance.md#open-questions).

## On Apple

- The command-line tools and agello-bench run on macOS; the iPhone is measured through Agello's
  harness, which runs the same `BenchKit` cases and reports telemetry per scenario.
- An agello-bench arm runs Metal natively on the Mac (our framework's macOS slice); Google's v0.18.0
  macOS binary, which the bench measured until 2026-10-28, used WebGPU. <!-- c:RT-OTH-005 -->
- The iPhone app on the Mac (`PLATFORM=mac` harness) uses the iOS slice and its Metal path.
  <!-- c:APL-061 -->

## Settings

| Variable | Read by | Effect | Claim |
| --- | --- | --- | --- |
| `AGELLO_TELEMETRY` | our LiteRT-LM build, at load | telemetry level 0..4 (default 1) | LM-CAPI-040 |
| `AGELLO_TELEMETRY_OUT` | same, at exit | snapshot (and trace at 3+) to a file | LM-CAPI-040 |
| `AGELLO_TELEMETRY_GRAPH_DIR` | our ML Drift | GPU graph dumps, any level | MD-MTL-034 |
| `AGELLO_SIMULATE_APPLE_GPU` | our ML Drift and LiteRT | another Apple GPU's kernel paths; joins cache keys | MD-ARC-016, MD-ARC-018 |
| `AGELLO_BOUNDED_ATTENTION`, `AGELLO_BOUNDED_WINDOW` | our LiteRT | `0` turns our bounded-attention pass, or its sliding window, off | RT-DLG-038 |
| `AGELLO_SLICE_ATTENTION`, `AGELLO_MASK_BOUNDS` | our LiteRT | `0` turns our prefill row slices, or the per-row mask bounds, off | RT-DLG-038 |
| `AGELLO_SHARED_WEIGHT_CACHE` | our LiteRT | `0` uses upstream's weight-cache key (one entry per context size) | RT-DLG-055 |
| `AGELLO_VERIFY_DIRECT_COPIES` | our LiteRT | `1` checks each direct Metal copy against ML Drift's conversion | RT-DLG-078 |
| `ENABLE_YNNPACK` | Swift `ExperimentalFlags` (Google's, in our wrapper) | YNNPACK when the build has it | LM-SWIFT-034 |
| `LITERTLM_PACKAGE` | agello-bench `Package.swift` | the arm to build against | this page |
| `AGELLO_BENCH_HOLD`, `AGELLO_BENCH_RELEASE_TRACE` | agello-bench | stay alive after the last case; follow the footprint after an engine is freed | X-TLS-008 |

Where each variable is read: [`agello/telemetry/agello/agello_telemetry.cc:52-59`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.cc#L52-L59),
[`agello/telemetry/agello/agello_telemetry.cc:874-891`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.cc#L874-L891),
[`ml_drift/metal/inference_context.mm:679-682`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/metal/inference_context.mm#L679-L682),
[`ml_drift/common/gpu_info.cc:1373-1384`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/common/gpu_info.cc#L1373-L1384),
[`ml_drift_delegate/delegate/delegate_kernel.cc:113-125`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/litert/ml_drift_delegate/delegate/delegate_kernel.cc#L113-L125),
[`ml_drift_delegate/delegate/composite/bounded_attention.cc:631-661`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/litert/ml_drift_delegate/delegate/composite/bounded_attention.cc#L631-L661),
[`ml_drift_delegate/delegate/buffer_handler_metal.mm:255-260`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/litert/ml_drift_delegate/delegate/buffer_handler_metal.mm#L255-L260).
`AGELLO_BOUNDED_ATTENTION`, `AGELLO_BOUNDED_WINDOW`, `AGELLO_SLICE_ATTENTION` and `AGELLO_MASK_BOUNDS`
are read once per process, and each enters the program cache key. <!-- c:RT-DLG-038 -->
`AGELLO_SIMULATE_APPLE_GPU` is read again whenever ML Drift creates a Metal environment, once per
`InferenceContext`. <!-- c:MD-ARC-020 --> Set all of them before the engine loads and don't change them
while it runs. The full settings tables are in [configuration.md](configuration.md).

## Limits and failure modes

- **Benchmark mode changes what is measured.** agello-bench runs with benchmark mode on by default,
  so its Metal wait is active (busy), while Agello has run without it since 2026-11-11 and gets the
  passive wait; pass `--no-benchmark-mode` to match the app. <!-- c:OUR-034 -->
- **Telemetry level 4 changes timing.** Use it for where time goes, not for speeds.
- **Caches across builds.** LiteRT-LM's cache keys ignore the runtime build; a shared cache folder
  can restore another build's programs. agello-bench's per-build folders and Agello's
  per-build folders guard against it. <!-- c:RT-DLG-048 -->
- **Disk.** A cache folder holds about 2.5 GB for E4B; `litertlm_peek --dump_files_dir` and
  `patch_rope.py` write multi-GB files. Use scratch folders and delete them.
- **Stale notes.** agello-bench's README and TELEMETRY.md's level table, above.

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Peek options | `main` | [`python/litert_lm_builder/litertlm_peek_main.py:37-51`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/python/litert_lm_builder/litertlm_peek_main.py#L37-L51) |
| Model info CLI | `main` | [`schema/model_info/model_info_cli.cc:24-48`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/schema/model_info/model_info_cli.cc#L24-L48) |
| Attention map | `inspect_e4b.py` `main` | [`tools/model-surgery/inspect_e4b.py:151-157`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/model-surgery/inspect_e4b.py#L151-L157) |
| CLI binaries | `litert_lm_main`, `litert_lm_advanced_main` targets | [`runtime/engine/BUILD:252-279`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/BUILD#L252-L279) |
| Bench usage | `usage` | [`tools/agello-bench/Sources/agello-bench/main.swift:5-28`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/agello-bench/main.swift#L5-L28) |
| Bench engine setup | `Bench.load` | [`tools/agello-bench/Sources/BenchKit/Bench.swift:55-58`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/tools/agello-bench/Sources/BenchKit/Bench.swift#L55-L58) |
| Telemetry levels | `agello::telemetry::Level` | [`agello/telemetry/agello/agello_telemetry.h:18-26`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.h#L18-L26) |
| Span names | `AGELLO_TELEMETRY_SPANS` | [`agello/telemetry/agello/agello_telemetry.h:117-166`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/telemetry/agello/agello_telemetry.h#L117-L166) |
| C telemetry | `litert_lm_telemetry_*` | [`c/telemetry.h:30-55`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/telemetry.h#L30-L55) |
| Graph dumps | `DumpGraphForTelemetry` | [`ml_drift/metal/inference_context.mm:679-682`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/metal/inference_context.mm#L679-L682) |
| Simulation switch | `GetGpuInfoFromDeviceDescription` | [`ml_drift/common/gpu_info.cc:1373-1384`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/ml_drift/ml_drift/common/gpu_info.cc#L1373-L1384) |


## Open questions

- Google's `litert_lm_main` and `litert_lm_advanced_main` haven't been built or run for this
  reference, and their sources are gone from our tree; that they would link the Metal accelerator
  statically is read from the BUILD files only.
- `tools/agello-docs` statements carry no claim tags: they describe the reference's own tools,
  committed in `main` on 2026-11-15 (`repin.py` and `frozen/` on 2026-11-18), after the pin.
