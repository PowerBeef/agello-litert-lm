# Building, releasing and vendoring our LiteRT-LM

The source build is this repository's one branch, `main` (checkout `../LiteRT-LM` beside Agello),
with LiteRT in `agello/litert` and ML Drift in `agello/ml_drift`. Until 2026-11-15 it lived on
`agello-next` and then `agello-one`, with LiteRT and ML Drift pinned from `PowerBeef/LiteRT` and
`PowerBeef/ml-drift` (now archived, our changes on their `main`); the reference pages under
`reference/` still describe that three-repository layout where they cover pins and overrides.

## Build

```sh
agello/build-xcframework.sh          # dist/CLiteRTLM.xcframework (iPhone and Mac slices) + dist/BUILD-INFO.txt
```

- `WORKSPACE` reads LiteRT and ML Drift from `agello/litert` and `agello/ml_drift`
  (`local_repository`): edit them in place and rebuild, with no override and no re-pin. The binary
  isn't byte-identical between builds (the build info compiled into it names the build), so check
  behavior by comparing replies, not bytes.
- Push with `git push origin main`.

## Build provenance (P1.1)

`--config=agello` runs `agello/workspace_status.sh`, which stamps the LiteRT-LM, LiteRT and ML
Drift commits (each with `+<hash>` of its uncommitted changes), the build flags, and a 12-digit
build id that changes with any of them. `litert_lm_get_build_info()` returns them as JSON
(`LiteRTLM.BuildInfo` in Swift). agello-bench records it in each machine record
(`runtimeBuild`), Agello's harness in its device record, and Agello's `EngineCache` keeps one
cache folder per build id (`kv-8192-<id>`; Google's binary, without an id, keeps `kv-8192`).

## Release (P1.3)

```sh
agello/release-xcframework.sh <tag>              # dry run: dist/release/<tag>/
agello/release-xcframework.sh <tag> --publish    # creates the GitHub release (needs the user's OK)
```

It refuses local overrides and uncommitted changes, builds, and writes the zip, its SwiftPM
checksum (`checksum.txt`), a release `Package.swift`, `BUILD-INFO.txt`, `THIRD_PARTY_NOTICES.md`
and `NOTES.md`. Agello then vendors it with
`REF=<commit> TAG=<tag> Vendor/use-release-litert.sh` (its header lists the follow-up:
`Vendor/README.md`, `NOTICE`, the README section, the iPhone suites).

Dry run on 2026-10-15 (commit `57376f53`, build id `3b702497f7d5`): a 26 MB zip; its slices hash
to the BUILD-INFO values, and the release `Package.swift` parses with the URL and checksum.

## Third-party licenses (P1.4)

`python3 agello/third_party_notices.py > agello/THIRD_PARTY_NOTICES.md` lists the external
repositories among the framework's link inputs (so build tools aren't listed), with each license
(Cargo `license` fields for Rust crates, a table for C/C++) and its text. As of 2026-10-15: 101
components, all permissive (MIT and/or Apache-2.0 for most, BSD-2/3-Clause, Zlib, Unlicense, and
Ooura's FFT license); nothing copyleft. Regenerate it when dependencies change; the release
script ships it.

## Privacy manifest (P1.5)

Required-reason APIs referenced by the iOS slice (`nm -u`): `stat`, `fstat`, `lstat` (file
timestamp) and `mach_absolute_time` (system boot time). Agello's `PrivacyInfo.xcprivacy` already
declares both categories (`C617.1`, `35F9.1`) for Google's binary, so switching to our build
needs no manifest change. Also referenced, not on Apple's list: `clock_gettime`, `sysctl`,
`sysctlbyname`. Recheck on every release.
