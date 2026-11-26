#!/bin/sh
# Build the LiteRTLM symbol graph in a scratch folder (outside the repos), render
# generated/swift-api-inventory.md, then delete the build (a few hundred MB).
# The checkout's swift/ must be the pin's, with dist/CLiteRTLM.xcframework built from the pin.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(python3 -c "import json,sys; print(json.load(open('$HERE/pins.json'))['repos']['litert-lm']['path'])")"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/agello-docs-symgraph}"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
(cd "$ROOT" && swift package --scratch-path "$SCRATCH/build" dump-symbol-graph --minimum-access-level public \
  > "$SCRATCH/dump.log" 2>&1)
MAIN_GRAPH="$(find "$SCRATCH/build" -name 'LiteRTLM.symbols.json' | head -1)"
TOOLS_GRAPH="$(find "$SCRATCH/build" -name 'LiteRTLMTools.symbols.json' | head -1)"
if [ -z "$MAIN_GRAPH" ]; then
  echo "no LiteRTLM symbol graph; see $SCRATCH/dump.log" >&2
  exit 1
fi
if [ -n "$TOOLS_GRAPH" ]; then
  python3 "$HERE/gen_swift_api.py" --log "$SCRATCH/dump.log" "$MAIN_GRAPH" "$TOOLS_GRAPH"
else
  python3 "$HERE/gen_swift_api.py" --log "$SCRATCH/dump.log" "$MAIN_GRAPH"
fi
status=$?
rm -rf "$SCRATCH"
exit $status
