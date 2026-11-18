#!/usr/bin/env python3
"""Make a citation for docs/agello/reference: a Markdown link and a ledger source record.

Untouched files link to Google's repository at the upstream base (so the link can be pasted into
an upstream issue); files our commits change link to our repository at the pin, LiteRT and ML Drift
under their agello/ folders. Paths are relative to the component (litert/c/..., ml_drift/metal/...).

Usage:
  cite.py <repo> <path> <start> [<end>]          # lines
  cite.py <repo> <path> --symbol "<text>" [--context N]   # first line containing the text
  add --json for the ledger source record (repo, sha, path, lines, symbol, sha1 of the cited lines)
  add --at-base to cite Google's base for a file our tree no longer has
Repos: litert-lm, litert, ml-drift (the pin, main), litert-lm-shipped (the release Agello ships); see pins.json.
"""
import argparse
import hashlib
import json
import sys

import common


def source_record(key, path, start, end, symbol=None, at_base=False):
    text = common.show(key, path, common.repo(key)["base"] if at_base else None)
    lines = text.splitlines()[start - 1:end]
    url = common.permalink(key, path, start, end)
    return {
        "type": "code", "repo": key, "sha": url.split("/blob/")[1].split("/")[0], "pin": common.repo(key)["pin"],
        "path": path, "lines": [start, end], "symbol": symbol,
        "lines_sha1": hashlib.sha1("\n".join(lines).encode()).hexdigest(), "url": url,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("path")
    p.add_argument("start", nargs="?", type=int)
    p.add_argument("end", nargs="?", type=int)
    p.add_argument("--symbol")
    p.add_argument("--context", type=int, default=0, help="extra lines after the symbol line")
    p.add_argument("--json", action="store_true")
    p.add_argument("--at-base", action="store_true", help="a file our tree no longer has, at Google's base")
    a = p.parse_args()

    sha = common.repo(a.repo)["base"] if a.at_base else None
    if common.show(a.repo, a.path, sha) is None:
        where = "Google's base" if a.at_base else f"the pin {common.repo(a.repo)['pin'][:10]}"
        hint = "" if a.at_base or common.show(a.repo, a.path, common.repo(a.repo)["base"]) is None else \
            " (it's at Google's base: our tree removed it; --at-base cites it there)"
        sys.exit(f"{a.path} not in {a.repo} at {where}{hint}")
    if a.at_base and not common.at_upstream(a.repo, a.path):
        sys.exit(f"{a.path} is in our tree: cite it at the pin")
    if a.symbol:
        hits = common.find_symbol(a.repo, a.path, a.symbol, sha)
        if not hits:
            sys.exit(f"'{a.symbol}' not found in {a.path}")
        if len(hits) > 1:
            print(f"note: '{a.symbol}' occurs on lines {hits}; using {hits[0]}", file=sys.stderr)
        start, end = hits[0], hits[0] + a.context
    else:
        start, end = a.start, a.end or a.start
    if a.json:
        print(json.dumps(source_record(a.repo, a.path, start, end, a.symbol, a.at_base)))
    else:
        print(common.md_link(a.repo, a.path, start, end))


if __name__ == "__main__":
    main()
