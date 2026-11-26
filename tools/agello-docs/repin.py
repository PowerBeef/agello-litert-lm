#!/usr/bin/env python3
"""Move the reference's citations to the pins in pins.json (maintenance.md, "After a re-pin or rebase").

  repin.py status            per group: code citations whose cited lines are unchanged at the new pins
                             (in place or moved), and the claims to re-verify, with the reason
  repin.py apply             re-cites every unchanged code citation where the merge reads it (the claims
                             in _work/claims, or the verdict whose corrected_sources the merge keeps),
                             writes _work/repin.json (the claims to re-verify), then merges every group
  repin.py links [--apply]   the same for the pages' permalinks (generated/ is regenerated instead);
                             lists the links it can't move
  repin.py labels [--apply]  link texts that name lines (`path:12-20`) whose numbers differ from their
                             permalink's #L anchors, as moving a link leaves them; --apply rewrites them
  repin.py show <group> [--ids A,B]   for verifiers: each listed claim as merged, why it's listed, and
                             per code source the old cited lines next to the file at the new pin

A citation is unchanged when its exact lines (lines_sha1) are found in the file at the new pin, or at
Google's base for a file we haven't changed (common.at_upstream). The search starts at the old line
and takes the nearest match. Anything else needs a new verdict: the lines changed, the file is gone, or
the old commit can't be read. Claims are also listed when a cited file is gone from our tree (the
citation then points to Google's base, where the file still is), when their text names an old pin or
branch, when they describe the shipped build (they cite litert-lm-shipped, or applies_to.build is
"shipped"), whose meaning moving that pin changes, and when they cite a table the pins regenerate.
Sources already at a current pin or base count as current; a claim whose sources are all current and
that was verified in the last re-pin's batches (previous_pins.repin_batch in pins.json) isn't listed.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import common
import claims as C

BLOB = re.compile(r"https://github\.com/([^/]+/[^/]+)/blob/([0-9a-f]{40})/([^#)\s]+)(?:#L(\d+)(?:-L(\d+))?)?")


def build_old_text_pattern():
    """Build a regex matching old pins, branches, releases, and model revisions from pins.json."""
    prev = common.PINS.get("previous_pins", {})
    tokens = []
    seen = set()

    def add_token(tok):
        if not isinstance(tok, str):
            return
        tok = tok.strip()
        if not tok or tok in seen:
            return
        seen.add(tok)
        if re.fullmatch(r"[0-9a-f]{7,40}", tok):
            tokens.append(rf"\b{re.escape(tok[:7])}\w*\b")
        else:
            tokens.append(rf"\b{re.escape(tok)}\b")

    for key, val in prev.items():
        if key in ("comment", "repin_batch", "stale_text_patterns"):
            continue
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and " " not in item:
                    add_token(item)
        elif isinstance(val, str) and " " not in val:
            add_token(val)

    parts = list(tokens)
    for pat in prev.get("stale_text_patterns", []):
        if isinstance(pat, str) and pat:
            parts.append(pat)
    if not parts:
        return re.compile(r"(?!)")
    return re.compile("|".join(parts))


OLD_TEXT = build_old_text_pattern()


def sha1(lines):
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def old_url_key(slug, sha, path):
    """(key, path) for a permalink to a current pin or base, or to a pin of before the last re-pin."""
    hit = common.url_key(slug, sha, path)
    if hit:
        return hit
    prev = common.PINS.get("previous_pins", {})
    for key, r in common.PINS["repos"].items():
        old = prev.get(key) or []
        imp = r.get("imported")
        if imp and slug == imp["from"] and sha == imp["from_pin"]:
            return key, path
        old_shas = [x for x in old if isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}", x)]
        if old_shas and sha in old_shas:
            p = common.prefix(key)
            if slug == r["fork"]:
                if p and path.startswith(p):
                    return key, path[len(p):]
                if not p:
                    return key, path
            if slug == r["upstream"]:
                return key, path
    return None


def relocate(key, path, sha, a, b, want_sha1=None, symbol=None):
    """(status, new source record or None) for lines a..b of path at sha."""
    old = common.show(key, path, sha)
    if old is None:
        return "unreadable", None
    block = old.splitlines()[a - 1:b]
    if want_sha1 and sha1(block) != want_sha1:
        return "unreadable", None
    r = common.repo(key)
    upstream = common.at_upstream(key, path)
    target = common.show(key, path, r["base"] if upstream else None)
    if target is None:
        return "file gone", None
    lines = target.splitlines()
    n = len(block)
    want = sha1(block)
    hits = [i for i in range(1, len(lines) - n + 2) if sha1(lines[i - 1:i - 1 + n]) == want]
    if not hits:
        return "changed", None
    start = min(hits, key=lambda i: abs(i - a))
    end = start + n - 1
    url = common.permalink(key, path, start, end if end != start else None)
    new_sha = url.split("/blob/")[1].split("/")[0]
    rec = {"type": "code", "repo": key, "sha": new_sha, "pin": r["pin"], "path": path, "lines": [start, end],
           "symbol": symbol, "lines_sha1": want, "url": url}
    status = "in place" if (start == a and new_sha == sha) else "moved"
    return status, rec


GENERATED_BY_PIN = ("generated/patch-log.md", "generated/binaries.md", "generated/kernels-by-family.md",
                    "generated/engine-run.md", "generated/version-matrix.md", "generated/c-api-inventory.md",
                    "generated/swift-api-inventory.md", "generated/proto-fields.md", "generated/cli-flags.md")


def current(key, sha):
    r = common.repo(key)
    return sha in (r["pin"], r["base"])


def effective_locations(name):
    """For each claim: (file, line index) of the record whose sources the merge keeps."""
    loc = {}
    claims_file = os.path.join(C.CLAIMS_DIR, f"{name}.jsonl")
    for i, line in enumerate(open(claims_file)):
        if line.strip():
            loc[json.loads(line)["id"]] = (claims_file, i, "sources")
    for f in sorted(glob.glob(os.path.join(C.VERDICTS_DIR, f"{name}.*.jsonl"))):
        for i, line in enumerate(open(f)):
            if line.strip():
                v = json.loads(line)
                if v.get("verdict") == "corrected" and v.get("corrected_sources") and v["id"] in loc:
                    loc[v["id"]] = (f, i, "corrected_sources")
    return loc


def survey():
    """Per group: {claim id: [reasons]} and per-source statuses, from the merged claims."""
    todo, counts, moves = {}, {}, {}
    for name in C.GROUPS:
        todo[name], counts[name] = {}, {}
        for c in C.merged(name):
            if c.get("verdict") == "wrong":
                continue  # off the pages
            reasons, shipped, stale = [], False, False
            for j, s in enumerate(c.get("sources", [])):
                if s.get("type") != "code" or s.get("repo") not in common.PINS["repos"]:
                    continue
                key, sha = s["repo"], s.get("sha") or common.repo(s["repo"])["pin"]
                a, b = s["lines"]
                if key == "litert-lm-shipped":
                    shipped = True
                if current(key, sha):
                    counts[name]["current"] = counts[name].get("current", 0) + 1
                    continue
                stale = True
                st, rec = relocate(key, s["path"], sha, a, b, s.get("lines_sha1"), s.get("symbol"))
                counts[name][st] = counts[name].get(st, 0) + 1
                where = f"{key} {s['path']}:{a}-{b}"
                if rec:
                    moves[(c["id"], j)] = rec
                    if common.show(key, s["path"]) is None:
                        reasons.append(f"our tree no longer has the file: {where}")
                else:
                    reasons.append(f"{st}: {where}")
            prev = common.PINS.get("previous_pins", {})
            batch = prev.get("repin_batch")
            verified_in = c.get("verified_in", "")
            in_current_repin = bool(batch and re.search(rf"\.{re.escape(batch)}(?:-|$)", verified_in))
            if not stale and in_current_repin:
                if reasons:
                    todo[name][c["id"]] = reasons
                continue  # verified at these pins in the current re-pin batch
            for s in c.get("sources", []):
                sp = s.get("path", "")
                if s.get("type") == "empirical" and (sp in GENERATED_BY_PIN or sp.startswith("generated/models/")):
                    reasons.append(f"cites {sp}, regenerated at the new pin")
            if OLD_TEXT.search(c.get("text", "")):
                reasons.append("text names an old pin, branch, release, or superseded condition")
            if shipped or (c.get("applies_to") or {}).get("build") == "shipped":
                reasons.append("describes the shipped build, now our release")
            if reasons:
                todo[name][c["id"]] = reasons
    return todo, counts, moves


def cmd_status():
    todo, counts, _ = survey()
    total = {}
    print(f"{'group':16} {'claims to re-verify':>20}  sources: current / in place / moved / changed / file gone / unreadable")
    for name in C.GROUPS:
        k = counts[name]
        for st, n in k.items():
            total[st] = total.get(st, 0) + n
        print(f"{name:16} {len(todo[name]):>20}  " + " / ".join(str(k.get(x, 0)) for x in
              ("current", "in place", "moved", "changed", "file gone", "unreadable")))
    print(f"total: {sum(len(t) for t in todo.values())} claims to re-verify; sources {total}")


def cmd_apply():
    todo, _, moves = survey()
    rewritten = 0
    for name in C.GROUPS:
        loc = effective_locations(name)
        by_file = {}
        for c in C.merged(name):
            if c["id"] in loc:
                f, i, field = loc[c["id"]]
                by_file.setdefault(f, {})[i] = (c, field)
        for f, rows in by_file.items():
            lines = open(f).read().splitlines()
            for i, (c, field) in rows.items():
                rec = json.loads(lines[i])
                srcs = rec[field]
                for j, s in enumerate(srcs):
                    new = moves.get((c["id"], j))
                    if new and s.get("type") == "code" and s != new:
                        srcs[j] = new
                        rewritten += 1
                lines[i] = json.dumps(rec, ensure_ascii=False)
            open(f, "w").write("\n".join(lines) + "\n")
        C.cmd_merge(name)
    out = {"pins": {k: r["pin"] for k, r in common.PINS["repos"].items()},
           "groups": {g: [{"id": cid, "reasons": rs} for cid, rs in t.items()] for g, t in todo.items() if t}}
    path = os.path.join(C.WORK, "repin.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"re-cited {rewritten} sources; {sum(len(t) for t in todo.values())} claims to re-verify: "
          f"{os.path.relpath(path, common.REF_DIR)}")


def cmd_links(apply):
    stuck, moved, kept = [], 0, 0
    for page in sorted(glob.glob(os.path.join(common.REF_DIR, "**", "*.md"), recursive=True)):
        rel = os.path.relpath(page, common.REF_DIR)
        if rel.startswith(("generated", "_work")):
            continue
        text = open(page).read()

        def fix(m):
            nonlocal moved, kept
            slug, sha, path, a, b = m.groups()
            hit = old_url_key(slug, sha, path)
            if hit is None:
                stuck.append(f"{rel}: {m.group(0)} (not a pin, base or previous pin)")
                return m.group(0)
            key, p = hit
            r = common.repo(key)
            fresh = common.permalink(key, p, int(a) if a else None, int(b) if b else None)
            if not a:
                if common.show(key, p, r["base"] if common.at_upstream(key, p) else None) is None:
                    stuck.append(f"{rel}: {m.group(0)} (file gone)")
                    return m.group(0)
                kept += fresh == m.group(0)
                moved += fresh != m.group(0)
                return fresh
            st, rec = relocate(key, p, sha, int(a), int(b or a))
            if rec is None:
                stuck.append(f"{rel}: {m.group(0)} ({st})")
                return m.group(0)
            url = rec["url"]
            kept += url == m.group(0)
            moved += url != m.group(0)
            return url

        new = BLOB.sub(fix, text)
        if apply and new != text:
            open(page, "w").write(new)
    for s in stuck:
        print("stuck", s)
    print(f"links: {kept} current, {moved} {'moved' if apply else 'to move'}, {len(stuck)} to re-cite by hand")


LABELLED = re.compile(r"\[`([^`\]]+?):(\d+)(?:-(\d+))?`\]\((https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/([^#)\s]+)#L(\d+)(?:-L(\d+))?)\)")


def stale_labels(text):
    """(match, new label text) for each link whose text names other lines than its permalink."""
    out = []
    for m in LABELLED.finditer(text):
        name, a, b, url, path, ua, ub = m.groups()
        if not path.endswith(name):
            continue
        a2, b2 = int(ua), int(ub or ua)
        if (int(a), int(b or a)) != (a2, b2):
            out.append((m, f"[`{name}:{a2}" + (f"-{b2}" if b2 != a2 else "") + f"`]({url})"))
    return out


def cmd_labels(apply):
    total = 0
    for page in sorted(glob.glob(os.path.join(common.REF_DIR, "**", "*.md"), recursive=True)):
        rel = os.path.relpath(page, common.REF_DIR)
        if rel.startswith("_work"):
            continue
        text = open(page).read()
        fixes = stale_labels(text)
        if not fixes:
            continue
        total += len(fixes)
        print(f"{rel}: {len(fixes)}")
        if apply:
            for m, new in reversed(fixes):
                text = text[:m.start()] + new + text[m.end():]
            open(page, "w").write(text)
    print(f"{total} link texts {'rewritten' if apply else 'name other lines than their permalink'}")


def nearest(old_block, lines):
    """Start (1-based) of the window of `lines` most like old_block."""
    import difflib
    n = max(1, len(old_block))
    best, where = -1.0, 1
    first = old_block[0].strip() if old_block else ""
    cands = [i for i, l in enumerate(lines, 1) if first and l.strip() == first] or range(1, max(2, len(lines) - n + 2))
    for i in cands:
        r = difflib.SequenceMatcher(None, "\n".join(old_block), "\n".join(lines[i - 1:i - 1 + n])).quick_ratio()
        if r > best:
            best, where = r, i
    return where, best


def cmd_show(name, ids):
    todo = json.load(open(os.path.join(C.WORK, "repin.json")))["groups"].get(name, [])
    reasons = {x["id"]: x["reasons"] for x in todo}
    ids = ids or list(reasons)
    by_id = {c["id"]: c for c in C.merged(name)}
    for cid in ids:
        c = by_id[cid]
        print("=" * 100)
        print(json.dumps({k: c[k] for k in ("id", "page", "anchor", "tier", "kind", "verdict", "text", "applies_to")
                          if k in c}, indent=1, ensure_ascii=False))
        for r in reasons.get(cid, []):
            print("listed:", r)
        for s in c.get("sources", []):
            if s.get("type") != "code":
                print("source:", json.dumps({k: v for k, v in s.items() if k != "quote"}, ensure_ascii=False)[:300])
                continue
            key, path, (a, b) = s["repo"], s["path"], s["lines"]
            sha = s.get("sha") or common.repo(key)["pin"]
            old = (common.show(key, path, sha) or "").splitlines()
            block = old[a - 1:b]
            print(f"--- {key}@{sha[:10]} {path}:{a}-{b}" + ("" if current(key, sha) else "  (old commit)"))
            for n, line in enumerate(block[:40], a):
                print(f"{n:6} {line}")
            if len(block) > 40:
                print(f"       ... ({len(block) - 40} more lines)")
            if current(key, sha):
                continue
            st, rec = relocate(key, path, sha, a, b, s.get("lines_sha1"), s.get("symbol"))
            if rec:
                print(f"+++ unchanged at {rec['url']}")
                continue
            upstream = common.at_upstream(key, path)
            r = common.repo(key)
            new = common.show(key, path, r["base"] if upstream else None)
            if new is None:
                print(f"+++ {st}: {path} is in neither our tree at {r['pin'][:10]} nor Google's base")
                continue
            lines = new.splitlines()
            at, ratio = nearest(block, lines)
            lo, hi = max(1, at - 5), min(len(lines), at + len(block) + 5)
            where = "Google's base (file gone from our tree)" if upstream and path not in common.pin_files(key) else "the new pin"
            print(f"+++ {st}; nearest lines at {where}: {path}:{lo}-{hi} (similarity {ratio:.2f})")
            for n in range(lo, min(hi, lo + 60) + 1):
                print(f"{n:6} {lines[n - 1]}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("apply")
    sub.add_parser("links").add_argument("--apply", action="store_true")
    sub.add_parser("labels").add_argument("--apply", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("group")
    s.add_argument("--ids", default="")
    a = p.parse_args()
    if a.cmd == "status":
        cmd_status()
    elif a.cmd == "apply":
        cmd_apply()
    elif a.cmd == "labels":
        cmd_labels(a.apply)
    elif a.cmd == "show":
        cmd_show(a.group, [x for x in a.ids.replace(" ", "").split(",") if x])
    else:
        cmd_links(a.apply)


if __name__ == "__main__":
    sys.exit(main())
