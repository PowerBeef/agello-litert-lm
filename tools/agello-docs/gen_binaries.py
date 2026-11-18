#!/usr/bin/env python3
"""generated/binaries.md: our CLiteRTLM (what Agello ships since 2026-10-28), from nm/otool/strings.

  gen_binaries.py [<our CLiteRTLM>]
Default: the ios-arm64 slice of dist/. The page ends with the record of 2026-10-16, which put Google's
v0.18.0 prebuilt (shipped until 2026-10-28) next to our build of then (frozen/binaries-2026-10-16.md):
we no longer build or compare against Google's runtime.
"""
import collections
import os
import re
import subprocess
import sys

import common

FROZEN = os.path.join(common.HERE, "frozen", "binaries-2026-10-16.md")
OURS = os.path.join(common.repo("litert-lm")["path"], "dist/CLiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM")

# Components to look for (symbol substrings). Closed parts per docs/agello/AUDIT.md §3; the open ones are for contrast.
MARKERS = [
    ("ML Drift Metal accelerator (open since 2026-10-08)", ["GpuMetalAccelerator", "CreateMlDriftMetalDelegate"]),
    ("TopK Metal sampler (closed)", ["TopKMetalSampler", "LiteRtTopKMetalSampler"]),
    ("Gemma model constraint provider / FST (closed)", ["GemmaModelConstraintProvider"]),
    ("llguidance constraints", ["LlgConstraint", "llg_"]),
    ("GPU artisan executor (closed)", ["GpuArtisan", "gpu_artisan", "LlmGpuArtisan"]),
    ("TENSOR_LLM / Google Tensor", ["TensorLlm", "TENSOR_LLM", "google_tensor"]),
    ("LiteRT compiled-model executor (open)", ["LlmLiteRtCompiledModelExecutor"]),
    ("MTP drafter (open)", ["LlmLiteRtMtpDrafter"]),
    ("XNNPACK", ["xnn_create_runtime", "xnn_initialize"]),
    ("KleidiAI", ["kai_"]),
    ("SentencePiece", ["sentencepiece"]),
    ("HuggingFace tokenizers", ["tokenizers_new_from_str", "HuggingFaceTokenizer"]),
    ("minijinja", ["minijinja"]),
    ("Agello telemetry (ours)", ["agello_telemetry", "litert_lm_telemetry_"]),
    ("Agello build info (ours)", ["litert_lm_get_build_info"]),
]


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def describe(path):
    exported = run("nm", "-gU", "-j", path).split()
    defined = run("nm", "-U", "-j", path).split()
    all_syms = run("nm", "-j", path).split()
    strings = run("strings", "-a", path)
    linked = [l.strip().split(" (")[0] for l in run("otool", "-L", path).splitlines()[1:]]
    c_api = sorted({s.lstrip("_") for s in exported if s.lstrip("_").startswith("litert_lm_")})
    found = {}
    for name, needles in MARKERS:
        n_def = sum(1 for sym in defined if any(n in sym for n in needles))
        n_str = sum(strings.count(n) for n in needles)
        found[name] = (n_def, n_str)
    return {"bytes": os.path.getsize(path), "exported": len(exported), "symbols": len(all_syms),
            "c_api": c_api, "linked": linked, "markers": found}


def build_id(path):
    info = os.path.join(os.path.dirname(path), "..", "..", "..", "BUILD-INFO.txt")
    m = re.search(r"Build id: (\w+)", open(info).read()) if os.path.exists(info) else None
    return m.group(1) if m else "?"


def main():
    ours = sys.argv[1] if len(sys.argv) > 1 else OURS
    o = describe(ours)
    md = [common.gen_header("gen_binaries.py", ["litert-lm"]),
          f"# CLiteRTLM binary: ours (build `{build_id(ours)}`)\n\n",
          "iOS arm64 slice of `dist/`. For each component: defined symbols whose (mangled) name contains one of its "
          "markers, then mentions in the binary's strings. Defined symbols mean the code is linked in; a name that "
          "appears only in strings is a reference (a dlopen name, a setting, a log message), not an implementation. "
          "Defined matches can still be open code that only configures or loads a closed part (`GpuArtisanConfig`, "
          "the dlopen wrapper for the Metal sampler). The record at the end compares Google's v0.18.0 prebuilt with "
          "our build of 2026-10-16.\n\n",
          "| | Ours |\n| --- | --- |\n",
          f"| Size | {o['bytes'] / 1e6:,.1f} MB |\n",
          f"| Symbols (all / exported) | {o['symbols']:,} / {o['exported']:,} |\n",
          f"| Exported `litert_lm_*` C functions | {len(o['c_api'])} |\n"]
    for name, _ in MARKERS:
        od, os_ = o["markers"][name]
        md.append(f"| {name} (defined / in strings) | {od} / {os_} |\n")
    md.append("\n## Linked libraries and frameworks\n\n" + "".join(f"- `{x}`\n" for x in o["linked"]))
    md.append("\n## Exported C functions\n\n" + ", ".join(f"`{x}`" for x in o["c_api"]) + "\n\n")
    md.append(open(FROZEN).read())
    print(common.write_gen("binaries.md", "".join(md)))


if __name__ == "__main__":
    main()
