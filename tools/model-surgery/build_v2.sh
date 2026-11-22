#!/bin/sh
# Build the Gemma 4 E4B v2 surgical .litertlm artifact (Agello P3.1):
#   1. ring_kv.py        (1,536-slot local-attention KV ring buffers)
#   2. window_128k.py    (131,072-token global window + unshifted Dual-GATHER RoPE tables)
#   3. verify_n.py       (verify_8 speculative signature for 7-draft ContextDrafts rounds)
#
# Input must already have `patch_rope.py --split --block 512` applied (e.g., the published
# PowerBeef02/gemma-4-E4B-it-litert-lm-fp16-rope file at a1693e22).
#
# Usage:
#   tools/model-surgery/build_v2.sh <in.litertlm> <out.litertlm>
set -eu

if [ $# -ne 2 ]; then
  echo "Usage: $0 <in.litertlm> <out.litertlm>" >&2
  exit 2
fi

SRC="$1"
DST="$2"
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${AGELLO_TOOLS_PYTHON:-$HOME/.venvs/agello-tools/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "Missing tools python: $PYTHON (set AGELLO_TOOLS_PYTHON)" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agello-v2.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

RING="$TMP_DIR/step1-ring.litertlm"
W128K="$TMP_DIR/step2-128k.litertlm"

echo "== Step 1/3: ring_kv.py (1,536-slot local KV rings)"
"$PYTHON" "$DIR/ring_kv.py" "$SRC" "$RING"

echo "== Step 2/3: window_128k.py (128K window + unshifted Dual-GATHER RoPE)"
"$PYTHON" "$DIR/window_128k.py" "$RING" "$W128K"
rm -f "$RING"

echo "== Step 3/3: verify_n.py --positions 8"
mkdir -p "$(dirname "$DST")"
"$PYTHON" "$DIR/verify_n.py" "$W128K" "$DST" --positions 8
rm -f "$W128K"

BYTES="$(stat -f '%z' "$DST" 2>/dev/null || stat -c '%s' "$DST")"
SHA="$(shasum -a 256 "$DST" | awk '{print $1}')"
echo "== Built v2 artifact: $DST"
echo "   size:   $BYTES bytes"
echo "   sha256: $SHA"
