# Re-pin of 2026-11-17: the verification brief

The reference moved from the old pins (LiteRT-LM `agello-next` `7b872544`, LiteRT `362759838` and ML
Drift `15fa9e2` in their own forks, the shipped wrapper `85b1a776` on v0.18.0) to the one repository:

| Key in pins.json | Pin | What it is |
| --- | --- | --- |
| `litert-lm` | `main` `b7b856a0` | LiteRT-LM at the repository's root; Google's base `df6df5b8` (0.19.0-nightly.20261013) |
| `litert` | `main` `b7b856a0`, folder `agello/litert` | imported unchanged at `e3c93749` from `PowerBeef/LiteRT` `362759838`; Google's base `ca21dc99` |
| `ml-drift` | `main` `b7b856a0`, folder `agello/ml_drift` | imported the same way from `PowerBeef/ml-drift` `15fa9e2`; Google's base `1b18771f` |
| `litert-lm-shipped` | `6da75396` | what Agello ships: our release `agello-one-20261115` (its Swift wrapper and framework) |

Between the old pins and `b7b856a0` there are about 116 commits: LiteRT and ML Drift moved into the
tree; three prunes removed what the Apple build never reads (other platforms' delegates and SDKs,
the Python, Kotlin and JS bindings, CI, upstream docs, most testdata; 5,514 LiteRT files, 347 ML Drift
files, 944 at the root); train 1's speedups, the sparse KV caches and the window read from the model
file (released in `agello-one-20261115`); then the fixes of 2026-11-17 (softmax at every window, a
speculative round's last token, pending tokens' embeddings, the Gemma 4 tool boundary, JSON with
speculative decoding), G1 (unknown Apple chips), the drafter chain on by default, and B5's long
rounds on `verify_8`. `generated/patch-log.md` lists every commit with its files.

The mechanical pass is done: every claim source and page link whose cited lines are byte-identical at
the new pin (or at Google's base, for files we haven't changed) was re-cited. What's left is in your
task file, `_work/repin/<you>.json`: the claims to re-verify, each with why it's listed, and the page
links whose cited lines changed.

## Read first

- `_work/style.md`: the rules for claims, evidence and wording. They all still apply.
- `maintenance.md`, "After a re-pin or rebase".
- Your task file. Work only on its pages and its verdict batches.

## Tools (run from `tools/agello-docs`)

- `python3 repin.py show <group> --ids A,B`: each claim as the ledger has it, why it's listed, and
  per code source the old cited lines next to the nearest lines at the new pin.
- `python3 cite.py <repo> <path> --symbol "<text>" [--context N] [--json]` or `cite.py <repo> <path> <start> <end>`:
  a Markdown link (pages) or, with `--json`, a ledger source record. Paths are relative to the
  component (`litert/c/...`, `ml_drift/metal/...`). `--at-base` cites Google's base for a file our
  tree no longer has.
- Read code at the pin with `git -C "/Users/patricedery/Coding projects/LiteRT-LM" show b7b856a0:<path>`
  (LiteRT and ML Drift paths start with `agello/litert/` and `agello/ml_drift/`), `git grep`, and
  `git diff <old> b7b856a0 -- <path>` to see what changed.
- `python3 claims.py check-verdicts <group> <batch> --ids ...` validates a verdict file.
- `python3 check_refs.py --only <page> ... --no-tags` checks your pages' links and anchors.

## Per claim

Decide the verdict at the new pin, and write one JSON line per claim to
`_work/verdicts/<group>.z-repin-<you>.jsonl` (the batch names are in your task file). Append as you
go, so a long run keeps its work.

- `verified`: the claim's current text holds. When any code source changed, give
  `corrected_sources`: the full list, each code source re-cited with `cite.py --json`, the others
  (web, empirical, agello, results, and code sources that didn't move) copied as the ledger has them.
- `corrected`: `corrected_text` is the full new claim text, not a diff; add `corrected_sources` as above.
- `wrong`: it no longer holds and no corrected version belongs on the page; its sentence leaves the page.
- `unverifiable`: only when nothing at the pin can settle it.
- `note`: always: what you checked, at which lines.

The merge applies verdicts batch by batch, so a `verified` keeps the text earlier corrections gave.

## Wording at the new pin

- The page header names the pin; claims don't name commits for the current state. Say "our build"
  (the documented build, `main`) and "the release Agello ships" (`agello-one-20261115`); replace
  "agello-next 7b872544" and the like. Name an old commit, branch or Google's v0.18.0 only for
  history, in the past tense with its date.
- Agello has shipped our runtime since 2026-10-28 (`agello-next-20261024`, then `agello-one-20261115`
  from 2026-11-15). A claim about the shipped build that described Google's v0.18.0 binary or the
  v0.18.0 wrapper now describes our release. A fact only about Google's binary stays only as dated
  history the page needs (the record at the end of `generated/binaries.md`); otherwise it's wrong.
  Don't add comparisons with Google's runtime: the backend is a hard fork (`docs/agello/BACKEND.md`),
  and Google's code is a read-only source.
- Code our tree removed: a claim about it says the tree removed it (with the prune), or goes.
- Numbers from regenerated tables (`generated/engine-run.md`, `kernels-by-family.md`,
  `patch-log.md`, `binaries.md`, `version-matrix.md`, the API inventories): check against the new
  table and correct them. The engine run is one run in benchmark mode, for orientation only.
- Mark our changes "(ours: `<commit>`)" as before, with the commit at the new pin's history.

## Pages

- For each verdict, edit the sentence that carries `<!-- c:ID -->` (an untagged sentence for tier-B
  claims that were unsampled): apply the corrected wording in the page's own phrasing, re-cite its
  links with `cite.py`, and for `wrong` delete the sentence and its tag. A tier-A claim that is
  verified or corrected keeps its tag.
- Fix every link listed for your pages: re-cite it at the new pin (or `--at-base` for a removed file,
  and say in the sentence that our tree removed it).
- Update each page's header: the **Pinned to** line (for example: "`main` `b7b856a0`: LiteRT-LM at
  the root, LiteRT in `agello/litert`, ML Drift in `agello/ml_drift`; Agello ships release
  `agello-one-20261115` (`6da75396`)", naming only the components the page covers), the **Applies to**
  line if it names the old branches, and **Last verified:** 2026-11-17.
- Edit only your pages and your verdict files. Don't run `claims.py merge` (the main loop merges
  every group at the end), don't commit, don't build, don't run the engine or measure anything,
  don't post anything. Delete any scratch you create.

## Done

`check_refs.py --only <your pages> --no-tags` reports no errors for your pages, and
`check-verdicts` passes for each of your batches with every listed claim present. Report: verdict
counts per group, the corrections where the code changed meaning, links or claims you couldn't
settle, and anything another page should change.
