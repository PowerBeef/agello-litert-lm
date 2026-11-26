#!/usr/bin/env python3
"""generated/swift-api-inventory.md: the public API of the LiteRTLM and LiteRTLMTools Swift modules.

Usage: gen_swift_api.py [--log <build log>] <LiteRTLM.symbols.json> [<LiteRTLMTools.symbols.json> ...]
(gen_swift_api.sh builds the symbol graphs in a scratch folder and calls this.)
"""
import argparse
import collections
import json
import os
import sys

import common

KEY = "litert-lm"


def decl(sym):
    return "".join(f["spelling"] for f in sym.get("declarationFragments", [])).strip()


def doc(sym, first_only=False):
    lines = [l["text"] for l in sym.get("docComment", {}).get("lines", [])]
    text = " ".join(l.strip() for l in lines).strip()
    if first_only and text:
        text = text.split(". ")[0].rstrip(".") + "."
    return text.replace("|", "\\|")


def where(sym):
    loc = sym.get("location")
    if not loc:
        return ""
    path = loc["uri"].replace("file://", "")
    root = common.repo(KEY)["path"]
    rel = os.path.relpath(path.replace("%20", " "), root)
    line = loc["position"]["line"] + 1
    return f"[`{rel}:{line}`]({common.permalink(KEY, rel, line)})"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="", help="Optional swift dump-symbol-graph build log")
    p.add_argument("inputs", nargs="+", help="One or more *.symbols.json files (plus optional trailing log)")
    args = p.parse_args()
    graphs = []
    log_path = args.log
    for item in args.inputs:
        if item.endswith(".json"):
            graphs.append(item)
        elif not log_path:
            log_path = item
    if not graphs:
        p.error("at least one *.symbols.json file is required")
    return graphs, log_path


def main():
    graph_paths, log_path = parse_args()
    log = open(log_path).read() if log_path and os.path.isfile(log_path) else ""
    modules = []
    syms = []
    seen_ids = set()
    for gp in graph_paths:
        graph = json.load(open(gp))
        mod_name = graph.get("module", {}).get("name")
        if mod_name and mod_name not in modules:
            modules.append(mod_name)
        for s in graph.get("symbols", []):
            if s.get("accessLevel") not in ("public", "open"):
                continue
            precise = s.get("identifier", {}).get("precise")
            key = precise or (tuple(s.get("pathComponents", [])), decl(s))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            syms.append((mod_name or "", s))

    by_type = collections.defaultdict(list)
    tops = {}
    top_module = {}
    for mod_name, s in syms:
        path = s["pathComponents"]
        if len(path) == 1:
            tops[path[0]] = s
            top_module[path[0]] = mod_name
        else:
            by_type[path[0]].append(s)

    mod_list = ", ".join(f"`{m}`" for m in modules)
    mod_label = "Modules" if len(modules) > 1 else "Module"
    md = [common.gen_header("gen_swift_api.sh", [KEY]), "# LiteRTLM Swift API inventory\n\n",
          f"{mod_label} {mod_list}, {len(syms)} public symbols, {len(tops)} top-level declarations. "
          "Built with `swift package dump-symbol-graph --minimum-access-level public` against `dist/CLiteRTLM.xcframework`.\n\n"]
    if "LiteRTLMFoundationModels" in log and "error" in log:
        md.append("> The `LiteRTLMFoundationModels` target (`swift/apple_fm/`) didn't compile against the macOS 27 SDK "
                  "when this was generated, so it isn't listed.\n\n")
    kinds = collections.Counter(tops[t]["kind"]["displayName"] for t in tops)

    def plural(k, n):
        k = k.lower()
        return k if n == 1 else (k + "es" if k.endswith("s") else k + "s")

    md.append("Top level: " + ", ".join(f"{n} {plural(k, n)}" for k, n in kinds.most_common()) + "\n\n")
    for name in sorted(tops, key=lambda n: (tops[n]["kind"]["displayName"], n)):
        t = tops[name]
        mod_note = f" — `{top_module[name]}`" if len(modules) > 1 and top_module.get(name) else ""
        md.append(f"## {name} ({t['kind']['displayName'].lower()}{mod_note})\n\n`{decl(t)}`  {where(t)}\n\n")
        if doc(t):
            md.append(doc(t) + "\n\n")
        members = sorted(by_type.get(name, []), key=lambda s: (s["kind"]["displayName"], s["pathComponents"]))
        if members:
            md.append("| Member | Kind | Declaration | Summary |\n| --- | --- | --- | --- |\n")
            for m in members:
                md.append(f"| {where(m) or '`' + '.'.join(m['pathComponents'][1:]) + '`'} "
                          f"| {m['kind']['displayName']} | `{decl(m).replace('|', chr(92) + '|')}` | {doc(m, True)} |\n")
            md.append("\n")
    print(common.write_gen("swift-api-inventory.md", "".join(md)))


if __name__ == "__main__":
    main()
