# agello-docs: tooling for docs/agello/reference

The reference documents LiteRT-LM, LiteRT (`agello/litert`), and ML Drift (`agello/ml_drift`) at the
commits in `pins.json` (`PowerBeef/agello-litert-lm` `main` and Google's upstream bases), the models
Agello uses, and what Agello ships. These scripts generate its tables from sources and files, manage
claims and re-pins, make citations, and check the result.

| Script / Dir | Does |
| --- | --- |
| `pins.json` | Repositories (local path, fork, pin, upstream, base, `previous_pins`), local model files, remote model files |
| `common.py` | Shared helpers: git, files our commits touch, permalinks |
| `claims.py` | Claim ledger helpers: merge `_work/claims/` + `_work/verdicts/` into `docs/agello/reference/ledger/*.jsonl` |
| `cite.py` | A citation: `cite.py litert-lm runtime/core/tasks.cc --symbol "Decode("` prints a Markdown link; `--json` prints the ledger source record |
| `repin.py` | Moves unchanged code citations and permalinks to new `pins.json` commits (`status`, `apply`, `links [--apply]`, `labels [--apply]`, `show <group>`) and writes `_work/repin.json` for claims needing re-verification |
| `check_refs.py` | Checks permalinks (commit, file, line range), relative links and anchors, ledger citations (hash of the cited lines), claim tags; `--numbers` flags untagged numbers; `--repin` runs `repin.py status` |
| `frozen/` | Immutable snapshots (e.g. historical pre-hard-fork binary inspection records) preserved across regenerations |
| `gen_c_api.py` | `generated/c-api-inventory.md`, from clang's AST of the public C headers |
| `gen_swift_api.sh` / `gen_swift_api.py` | `generated/swift-api-inventory.md`, from `LiteRTLM` and `LiteRTLMTools` Swift symbol graphs built in a scratch folder |
| `gen_protos.py` | `generated/proto-fields.md` |
| `gen_flags.py` | `generated/cli-flags.md` |
| `gen_patch_log.py` | `generated/patch-log.md`: our commits on top of upstream across the root, `agello/litert`, and `agello/ml_drift` |
| `gen_model_tables.py` | `generated/models/*.{md,json}`: sections, signatures, ops, composites, weights by type, per-layer attention for `.litertlm` files (tools venv); non-`.litertlm` Core AI ANE bundles are summarized from metadata |
| `gen_versions.py` | `generated/version-matrix.md` |
| `gen_binaries.py` | `generated/binaries.md`: our `dist/CLiteRTLM.xcframework` slices beside the frozen historical record |
| `gen_hf_configs.py` | `generated/hf/` and `generated/hf-configs.md`: official model configs at pinned revisions (`--update` moves the pins) |
| `capture_empirical.sh` / `gen_empirical.py` | `generated/kernels-by-family.md` and `generated/engine-run.md`, from engine runs per Apple GPU family (`--no-benchmark-mode`) |
| `regen.sh` | Runs the generators above (`--code` or `--all`), then `check_refs.py` |

Citations link Google's repository at the upstream base for files our commits don't change (so
they paste into upstream issues), and our fork at the pin otherwise. Don't type line numbers by hand.

After moving a pin: update `pins.json` (and `previous_pins`), run `regen.sh --all`, then
`repin.py apply`, `repin.py links --apply`, `repin.py labels --apply`, re-verify the claims in
`_work/repin.json`, and finish with `check_refs.py` (see `docs/agello/reference/maintenance.md`).
