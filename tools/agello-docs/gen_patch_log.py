#!/usr/bin/env python3
"""generated/patch-log.md: every commit of ours between the upstream bases and the pin, with files.

One repository since 2026-11-15: LiteRT-LM's commits on Google's base, then for LiteRT and ML Drift
the old forks' commits (imported unchanged; listed from pins.json, the forks are archived) and the
commits since that change their folders. A commit the shipped release contains is marked.
"""
import collections

import common

COMPONENTS = ["litert-lm", "litert", "ml-drift"]
SHIPPED = "litert-lm-shipped"


def esc(s):
    return s.replace("|", "\\|")


def files_cell(files):
    return ", ".join(f"`{f}`" for f in files[:6]) + (f" (+{len(files) - 6})" if len(files) > 6 else "")


def log(key, rng, *paths):
    out = common.git(key, "log", "--reverse", "--format=%H%x09%ad%x09%s", "--date=short", rng, "--", *paths)
    return [row.split("\t", 2) for row in out.strip().splitlines()]


def differ(key):
    """Files that differ from Google's base: those our tree has, and the removed ones by folder."""
    have, gone = [], collections.Counter()
    at_pin = set(common.ls_files(key))
    for f in sorted(common.touched(key)):
        if f in at_pin:
            have.append(f)
        else:
            gone["/".join(f.split("/")[:2]) if "/" in f else f] += 1
    md = [f"\n<details><summary>Files that differ from upstream: {len(have)} changed or added, "
          f"{sum(gone.values())} removed</summary>\n\n"]
    md += [f"- `{f}`\n" for f in have]
    if gone:
        md.append("\nRemoved, by folder: " + ", ".join(f"`{k}` {n}" for k, n in sorted(gone.items())) + "\n")
    md.append("\n</details>\n\n")
    return "".join(md)


def main():
    shipped = common.repo(SHIPPED)
    released = set(common.git(SHIPPED, "rev-list", f"{shipped['base']}..{shipped['pin']}").split())
    md = [common.gen_header("gen_patch_log.py", COMPONENTS + [SHIPPED]), "# Our commits on top of upstream\n\n",
          f"One repository, `{common.repo('litert-lm')['fork']}` `{common.repo('litert-lm')['branch']}`, since "
          "2026-11-15: LiteRT and ML Drift are its `agello/litert` and `agello/ml_drift` folders. "
          f"✓ marks the commits in the release Agello ships, `{shipped['release']}` (`{shipped['pin'][:10]}`).\n\n"]
    head = "| Commit | Date | Subject | Files | Shipped |\n| --- | --- | --- | --- | --- |\n"
    for key in COMPONENTS:
        r = common.repo(key)
        p = common.prefix(key)
        if not r.get("imported"):
            others = [f":(exclude){x['subdir']}" for x in common.PINS["repos"].values() if x.get("subdir")]
            rows = log(key, f"{r['base']}..{r['pin']}")
            md.append(f"## {key}\n\n{len(rows)} commits from {r['upstream']} `{r['base'][:10]}` ({r['base_label']}) "
                      f"to `{r['pin'][:10]}`; {len(common.touched(key))} files differ outside the LiteRT and ML Drift "
                      f"folders.\n\n{head}")
            for sha, date, subject in rows:
                files = common.git(key, "show", "--name-only", "--format=", sha, "--", ".", *others).split()
                md.append(f"| [`{sha[:10]}`](https://github.com/{r['fork']}/commit/{sha}) | {date} | {esc(subject)} "
                          f"| {files_cell(files)} | {'✓' if sha in released else ''} |\n")
            md.append(differ(key))
            continue
        imp = r["imported"]
        md.append(f"## {key}\n\nIn `{r['subdir']}`. Imported unchanged at [`{imp['commit'][:10]}`]"
                  f"(https://github.com/{r['fork']}/commit/{imp['commit']}) from `{imp['from']}` "
                  f"`{imp['from_pin'][:10]}` (archived): {r['upstream']} `{r['base'][:10]}` ({r['base_label']}) plus "
                  f"{len(imp['commits'])} commits of ours.\n\n| Commit | Date | Subject | Files |\n| --- | --- | --- | --- |\n")
        for sha, date, subject, nfiles in imp["commits"]:
            md.append(f"| [`{sha[:10]}`](https://github.com/{imp['from']}/commit/{sha}) | {date} | {esc(subject)} "
                      f"| {nfiles} |\n")
        rows = log(key, f"{imp['commit']}..{r['pin']}", r["subdir"])
        md.append(f"\n{len(rows)} commits since the import change `{r['subdir']}`; {len(common.touched(key))} files "
                  f"differ from {r['upstream']} `{r['base'][:10]}`.\n\n{head}")
        for sha, date, subject in rows:
            files = [f[len(p):] for f in common.git(key, "show", "--name-only", "--format=", sha, "--", r["subdir"]).split()]
            md.append(f"| [`{sha[:10]}`](https://github.com/{r['fork']}/commit/{sha}) | {date} | {esc(subject)} "
                      f"| {files_cell(files)} | {'✓' if sha in released else ''} |\n")
        md.append(differ(key))
    print(common.write_gen("patch-log.md", "".join(md)))


if __name__ == "__main__":
    main()
