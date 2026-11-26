# Building, releasing and vendoring our LiteRT-LM

The source build is this repository's one branch, `main` (checkout `../LiteRT-LM` beside Agello),
with LiteRT in `agello/litert` and ML Drift in `agello/ml_drift`. Until 2026-11-15 it lived on
`agello-next` and then `agello-one`, with LiteRT and ML Drift pinned from `PowerBeef/LiteRT` and
`PowerBeef/ml-drift` (now archived, our changes on their `main`). The deep reference under
`reference/` was re-pinned to this single-repository layout on 2026-11-17 (`b7b856a0`) and awaits
its next repin pass for post-Phase-2 changes.

## Build

```sh
agello/build-xcframework.sh              # dist/CLiteRTLM.xcframework (iPhone + Mac slices) + dist/BUILD-INFO.txt
agello/build-xcframework.sh --mac-only   # ~2x faster macos-arm64 build for Mac CLI / agello serve iteration
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
(`LiteRTLM.BuildInfo` in Swift). `agello-bench` records it in each machine record
(`runtimeBuild`), Agello's harness in its device record, and Agello's `EngineCache` keeps one
shared cache folder per build id (`kv-all-<id>` and `embedding-<id>`), deleting older builds'
folders automatically.

## Release (P1.3)

```sh
agello/release-xcframework.sh <tag>              # dry run: dist/release/<tag>/
agello/release-xcframework.sh <tag> --publish    # creates the GitHub release (needs the user's OK)
```

It refuses local overrides and uncommitted changes, builds, and writes the zip, its SwiftPM
checksum (`checksum.txt`), a release `Package.swift`, `BUILD-INFO.txt`, `THIRD_PARTY_NOTICES.md`
and `NOTES.md`. Agello then vendors it with
`REF=<commit> TAG=<tag> Vendor/use-release-litert.sh` (its header lists the follow-up:
`Vendor/README.md`, `NOTICE` / `Agello/Resources/NOTICE.txt`, `docs/BACKEND.md`, and the iPhone
suites).

Shipped example (`agello-one-20261124`, commit `c819aecb`, build id `8a7e34594612`): dual-slice
`ios-arm64` + `macos-arm64` Metal XCFramework zip with SwiftPM checksum `336023ee…3de7c`.

## Third-party licenses (P1.4)

`python3 agello/third_party_notices.py > agello/THIRD_PARTY_NOTICES.md` lists the external
repositories among the framework's link inputs (so build tools aren't listed), with each license
(Cargo `license` fields for Rust crates, a table for C/C++) and its text. Currently 101 linked
components, all permissive (MIT and/or Apache-2.0 for most, BSD-2/3-Clause, Zlib, Unlicense, and
Ooura's FFT license); nothing copyleft. Regenerate it whenever dependencies change; the release
script ships it and Agello copies it into `NOTICE` / `Agello/Resources/NOTICE.txt`.

## Privacy manifest (P1.5)

Required-reason APIs referenced by the iOS slice (`nm -u`): `stat`, `fstat`, `lstat` (file
timestamp) and `mach_absolute_time` (system boot time). Agello's `PrivacyInfo.xcprivacy` declares
both categories (`C617.1`, `35F9.1`). Also referenced, not on Apple's list: `clock_gettime`,
`sysctl`, `sysctlbyname`. Recheck with `nm -u` on every release.
