"""Shared helpers for the reference generators (pins, git, permalinks, output paths).

Since 2026-11-15 the runtime is one repository: LiteRT-LM at its root, LiteRT and ML Drift in its
`agello/litert` and `agello/ml_drift` folders (a repo's `subdir` in pins.json). Paths given to these
helpers, and paths in the ledger, are relative to each component's root, as in Google's repositories.

LiteRT's and ML Drift's upstream bases (Google's commits) aren't in this repository. Their trees are
known through the import: the folders came in unchanged at `imported.commit` from our old forks' pins
(`imported.from_pin`), and the old forks changed only the files in `base_changed`, so every other file
is the same at the base as at the import.
"""
import functools
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = json.load(open(os.path.join(HERE, "pins.json")))
REF_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "agello", "reference"))
GEN_DIR = os.path.join(REF_DIR, "generated")
LEDGER_DIR = os.path.join(REF_DIR, "ledger")


def repo(key):
    return PINS["repos"][key]


def prefix(key):
    """The component's folder in the repository, with a trailing slash ('' at the root)."""
    sub = repo(key).get("subdir")
    return sub + "/" if sub else ""


def git(key, *args, check=True):
    r = subprocess.run(["git", "-C", repo(key)["path"], *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {key}: {r.stderr.strip()}")
    return r.stdout


def local_commit(key, sha):
    """The local commit that holds `key`'s tree at `sha`, or None when only Google has it."""
    r = repo(key)
    imp = r.get("imported")
    if not imp:
        return sha
    if sha == imp["from_pin"]:
        return imp["commit"]
    if sha == r["base"]:
        return "base"
    return sha


@functools.lru_cache(maxsize=None)
def touched(key):
    """Files (relative to the component) that differ between Google's base and the pin."""
    r = repo(key)
    if r.get("imported"):
        p = prefix(key)
        ours = git(key, "diff", "--no-renames", "--name-only", r["imported"]["commit"], r["pin"], "--", r["subdir"])
        return frozenset(set(r["base_changed"]) | {f[len(p):] for f in ours.split()})
    others = [f":(exclude){x['subdir']}" for x in PINS["repos"].values() if x.get("subdir") and x["path"] == r["path"]]
    return frozenset(git(key, "diff", "--no-renames", "--name-only", r["base"], r["pin"], "--", ".", *others).split())


@functools.lru_cache(maxsize=None)
def show(key, path, sha=None):
    """File contents at a commit (the pin by default), or None if absent or unknown locally."""
    r = repo(key)
    local = local_commit(key, sha or r["pin"])
    if local == "base":
        if path in r["base_changed"]:
            return None
        local = r["imported"]["commit"]
    out = subprocess.run(["git", "-C", r["path"], "show", f"{local}:{prefix(key)}{path}"], capture_output=True)
    return out.stdout.decode("utf-8", "replace") if out.returncode == 0 else None


def ls_files(key, *patterns, sha=None):
    p = prefix(key)
    out = git(key, "ls-tree", "-r", "--name-only", sha or repo(key)["pin"], "--",
              *([p + x for x in patterns] or ([p] if p else [])))
    return [f[len(p):] for f in out.split()]


@functools.lru_cache(maxsize=None)
def pin_files(key):
    return frozenset(ls_files(key))


def at_upstream(key, path):
    """Whether a citation of `path` goes to Google's base: files we haven't changed (identical there,
    so line numbers hold for both), and files our tree no longer has."""
    r = repo(key)
    if show(key, path, r["base"]) is None:
        return False
    return path not in touched(key) or path not in pin_files(key)


def permalink(key, path, start=None, end=None):
    """GitHub permalink: Google's repository at the base (see at_upstream), our repository at the pin otherwise."""
    r = repo(key)
    if at_upstream(key, path):
        url = f"https://github.com/{r['upstream']}/blob/{r['base']}/{path}"
    else:
        url = f"https://github.com/{r['fork']}/blob/{r['pin']}/{prefix(key)}{path}"
    if start:
        url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
    return url


def url_key(slug, sha, path):
    """(key, path relative to the component) for a permalink to a pin or base, or None."""
    best = None
    for key, r in PINS["repos"].items():
        if slug == r["upstream"] and sha == r["base"]:
            return key, path
        p = prefix(key)
        if slug == r["fork"] and sha == r["pin"] and path.startswith(p):
            if best is None or len(p) > len(prefix(best[0])):
                best = (key, path[len(p):])
    return best


def label(path, start=None, end=None):
    if not start:
        return path
    return f"{path}:{start}" + (f"-{end}" if end and end != start else "")


def md_link(key, path, start=None, end=None):
    return f"[`{label(path, start, end)}`]({permalink(key, path, start, end)})"


def find_symbol(key, path, symbol, sha=None):
    """1-based line numbers where `symbol` occurs in the file."""
    text = show(key, path, sha)
    if text is None:
        return []
    return [i for i, line in enumerate(text.splitlines(), 1) if symbol in line]


def gen_header(script, sources):
    """Banner for generated pages: what made them, from what."""
    pins = ", ".join(f"{k} `{repo(k)['pin'][:10]}`" for k in sources if k in PINS["repos"])
    extra = ", ".join(s for s in sources if s not in PINS["repos"])
    src = "; ".join(x for x in (pins, extra) if x)
    return (f"<!-- Generated by tools/agello-docs/{script}; do not edit. Regenerate with tools/agello-docs/regen.sh. -->\n"
            f"> Generated by `tools/agello-docs/{script}` from {src}. Do not edit by hand.\n\n")


def write_gen(name, text):
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    return path
