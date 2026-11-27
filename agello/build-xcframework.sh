#!/bin/sh
# Build CLiteRTLM.xcframework (iPhone and Mac slices, Metal accelerator linked statically) and
# unpack it into dist/, where Package.swift's binary target points.
# Usage: agello/build-xcframework.sh [--mac-only] [extra bazel flags...]
#   --mac-only  Build only the macos-arm64 slice (//agello/mac:CLiteRTLM) for fast Mac CLI
#               iteration (~2x faster incremental builds). Omit for PLATFORM=mac / iPhone builds.
# Writes dist/CLiteRTLM.xcframework (with bundled dSYMs), dist/dSYMs/, and dist/BUILD-INFO.txt.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAC_ONLY=0
if [ "${1:-}" = "--mac-only" ]; then
  MAC_ONLY=1
  shift
fi
if [ "$MAC_ONLY" -eq 1 ]; then
  TARGET="//agello/mac:CLiteRTLM"
  ZIP="bazel-bin/agello/mac/CLiteRTLM.xcframework.zip"
  DSYM_ROOT="bazel-bin/agello/mac/CLiteRTLM_dsyms"
  SLICES_NOTE="macos-arm64 only (--mac-only dev build)"
else
  TARGET="//agello:CLiteRTLM"
  ZIP="bazel-bin/agello/CLiteRTLM.xcframework.zip"
  DSYM_ROOT="bazel-bin/agello/CLiteRTLM_dsyms"
  SLICES_NOTE="ios-arm64 + macos-arm64"
fi
COMMON_FLAGS="--config=agello -c opt --apple_generate_dsym --output_groups=+dsyms --copt=-gline-tables-only --objccopt=-gline-tables-only --copt=-DMA_NO_DEVICE_IO --objccopt=-DMA_NO_DEVICE_IO"
export AGELLO_BUILD_FLAGS="$COMMON_FLAGS${*:+ $*}"
bazelisk build \
  --config=agello \
  -c opt \
  --apple_generate_dsym \
  --output_groups=+dsyms \
  --copt=-gline-tables-only \
  --objccopt=-gline-tables-only \
  --copt=-DMA_NO_DEVICE_IO \
  --objccopt=-DMA_NO_DEVICE_IO \
  "$@" \
  "$TARGET"

STAGE="$(cd "$(mktemp -d "${TMPDIR:-/tmp}/agello-xcframework.XXXXXX")" && pwd)"
trap 'rm -rf "$STAGE"' EXIT
unzip -q "$ZIP" -d "$STAGE/raw"

set_plist_string() {
  plist="$1"
  key="$2"
  val="$3"
  /usr/libexec/PlistBuddy -c "Set :$key $val" "$plist" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Add :$key string $val" "$plist"
}

prepare_dsym() {
  bazel_name="$1"
  out_dsym="$2"
  slice_binary="$3"
  src_dsym="$DSYM_ROOT/${bazel_name}.framework.dSYM"
  src_dwarf="$src_dsym/Contents/Resources/DWARF/$bazel_name"
  src_plist="$src_dsym/Contents/Info.plist"
  if [ ! -f "$src_dwarf" ] || [ ! -f "$src_plist" ]; then
    echo "Missing generated dSYM for $bazel_name under $DSYM_ROOT" >&2
    exit 1
  fi
  mkdir -p "$out_dsym/Contents/Resources/DWARF"
  cp "$src_dwarf" "$out_dsym/Contents/Resources/DWARF/CLiteRTLM"
  cp "$src_plist" "$out_dsym/Contents/Info.plist"
  chmod 0644 "$out_dsym/Contents/Info.plist" "$out_dsym/Contents/Resources/DWARF/CLiteRTLM"
  set_plist_string "$out_dsym/Contents/Info.plist" CFBundleIdentifier com.apple.xcode.dsym.com.patricedery.agello.CLiteRTLM
  set_plist_string "$out_dsym/Contents/Info.plist" CFBundleShortVersionString 0.19.0
  set_plist_string "$out_dsym/Contents/Info.plist" CFBundleVersion 1
  bin_uuid="$(xcrun dwarfdump --uuid "$slice_binary" | awk 'NR==1 {print $2}')"
  dsym_uuid="$(xcrun dwarfdump --uuid "$out_dsym" | awk 'NR==1 {print $2}')"
  if [ -z "$bin_uuid" ] || [ "$bin_uuid" != "$dsym_uuid" ]; then
    echo "dSYM UUID mismatch for $slice_binary: binary=$bin_uuid dsym=$dsym_uuid" >&2
    exit 1
  fi
}

if [ "$MAC_ONLY" -eq 0 ]; then
  # rules_apple 3.22.0 builds the iOS environment plist in the default iOS configuration, which is
  # the simulator on Apple silicon: the device slice said iPhoneSimulator. Give it the device SDK's.
  ios_fw="$STAGE/raw/CLiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework"
  ios_plist="$ios_fw/Info.plist"
  ios_sdk_version="$(xcrun --sdk iphoneos --show-sdk-version)"
  ios_sdk_build="$(xcrun --sdk iphoneos --show-sdk-build-version)"
  chmod 0644 "$ios_plist" "$ios_fw/PrivacyInfo.xcprivacy" \
    "$ios_fw/Headers/"*.h "$ios_fw/Modules/module.modulemap"
  /usr/libexec/PlistBuddy \
    -c "Set :CFBundleSupportedPlatforms:0 iPhoneOS" \
    -c "Set :DTPlatformName iphoneos" \
    -c "Set :DTPlatformVersion $ios_sdk_version" \
    -c "Set :DTPlatformBuild $ios_sdk_build" \
    -c "Set :DTSDKName iphoneos$ios_sdk_version" \
    -c "Set :DTSDKBuild $ios_sdk_build" \
    "$ios_plist"
  chmod 0755 "$ios_fw/CLiteRTLM"
  install_name_tool -delete_rpath @executable_path/Frameworks "$ios_fw/CLiteRTLM" 2>/dev/null || true
  xcrun strip -S -x "$ios_fw/CLiteRTLM"
  ios_dsym="$STAGE/dsyms/ios-arm64/CLiteRTLM.framework.dSYM"
  prepare_dsym CLiteRTLM_ios_device "$ios_dsym" "$ios_fw/CLiteRTLM"
fi

# macOS frameworks are versioned bundles, and rules_apple writes the iOS-style shallow one:
# Versions/A holds the binary, headers, module map and Resources/{Info.plist,PrivacyInfo.xcprivacy},
# and the top level links through Versions/Current. The install name,
# @rpath/CLiteRTLM.framework/CLiteRTLM, resolves through the top-level link.
mac_fw="$STAGE/raw/CLiteRTLM.xcframework/macos-arm64/CLiteRTLM.framework"
mkdir -p "$mac_fw/Versions/A/Resources"
mv "$mac_fw/CLiteRTLM" "$mac_fw/Headers" "$mac_fw/Modules" "$mac_fw/Versions/A/"
mv "$mac_fw/Info.plist" "$mac_fw/PrivacyInfo.xcprivacy" "$mac_fw/Versions/A/Resources/"
ln -s A "$mac_fw/Versions/Current"
for item in CLiteRTLM Headers Modules Resources; do
  ln -s "Versions/Current/$item" "$mac_fw/$item"
done
chmod 0644 "$mac_fw/Versions/A/Resources/Info.plist" \
  "$mac_fw/Versions/A/Resources/PrivacyInfo.xcprivacy" \
  "$mac_fw/Versions/A/Headers/"*.h \
  "$mac_fw/Versions/A/Modules/module.modulemap"
chmod 0755 "$mac_fw/Versions/A/CLiteRTLM"
install_name_tool -delete_rpath @executable_path/Frameworks "$mac_fw/Versions/A/CLiteRTLM" 2>/dev/null || true
xcrun strip -S -x "$mac_fw/Versions/A/CLiteRTLM"
codesign --force --sign - "$mac_fw/Versions/A/CLiteRTLM"
mac_dsym="$STAGE/dsyms/macos-arm64/CLiteRTLM.framework.dSYM"
prepare_dsym CLiteRTLM_macos_device "$mac_dsym" "$mac_fw/Versions/A/CLiteRTLM"

rm -rf dist/CLiteRTLM.xcframework dist/dSYMs
mkdir -p dist dist/dSYMs
if [ "$MAC_ONLY" -eq 0 ]; then
  xcodebuild -create-xcframework \
    -framework "$ios_fw" -debug-symbols "$ios_dsym" \
    -framework "$mac_fw" -debug-symbols "$mac_dsym" \
    -output "$ROOT/dist/CLiteRTLM.xcframework"
  cp -R "$STAGE/dsyms/ios-arm64" "$STAGE/dsyms/macos-arm64" dist/dSYMs/
else
  xcodebuild -create-xcframework \
    -framework "$mac_fw" -debug-symbols "$mac_dsym" \
    -output "$ROOT/dist/CLiteRTLM.xcframework"
  cp -R "$STAGE/dsyms/macos-arm64" dist/dSYMs/
fi
chmod 0644 dist/CLiteRTLM.xcframework/Info.plist
# A repository overridden in .bazelrc.user (a local checkout) reports that checkout's commit.
source_of() {
  dir="$(sed -n "s|^build \"*--override_repository=$1=\([^\"]*\)\"*\$|\1|p" .bazelrc.user 2>/dev/null | tail -n 1)"
  if [ -n "$dir" ]; then
    echo "$(git -C "$dir" rev-parse HEAD)$(git -C "$dir" diff --quiet HEAD || echo ' (modified)') (local: $dir)"
  else
    # In this tree (agello/litert, agello/ml_drift) since the one repository: this repository's
    # commit covers it, and the WORKSPACE keeps the commit it was imported at.
    echo "in tree, imported at $(sed -n "s/^$2 = \"\(.*\)\"/\1/p" WORKSPACE)"
  fi
}
{
  echo "LiteRT-LM: $(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (modified)')"
  echo "LiteRT: $(source_of litert AGELLO_LITERT_COMMIT)"
  echo "ML Drift: $(source_of ml_drift ML_DRIFT_COMMIT)"
  echo "Bazel: $(bazelisk version 2>/dev/null | sed -n 's/^Build label: //p')"
  echo "Xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "Flags: $AGELLO_BUILD_FLAGS"
  echo "Slices: $SLICES_NOTE"
  echo "Build id: $(sed -n 's/^STABLE_AGELLO_BUILD_ID //p' bazel-out/stable-status.txt)"
  echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for slice in dist/CLiteRTLM.xcframework/*/CLiteRTLM.framework; do
    binary="$(find "$slice" -name CLiteRTLM -type f | head -n 1)"
    echo "$(shasum -a 256 "$binary" | cut -d' ' -f1)  ${binary#dist/}"
  done
  for slice in dist/CLiteRTLM.xcframework/*/CLiteRTLM.framework; do
    binary="$(find "$slice" -name CLiteRTLM -type f | head -n 1)"
    slice_name="$(basename "$(dirname "$slice")")"
    echo "UUID $slice_name: $(xcrun dwarfdump --uuid "$binary" | awk 'NR==1 {print $2}')"
  done
} > dist/BUILD-INFO.txt
cat dist/BUILD-INFO.txt
