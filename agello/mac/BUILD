# Mac-only XCFramework slice for fast M6 inner-loop iteration with the Agello CLI (`cli-test.sh`,
# `cli-bench.sh`). Full dual-slice builds (iPhone + Mac) live in //agello:CLiteRTLM.
#
#   agello/build-xcframework.sh --mac-only

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_xcframework")

apple_xcframework(
    name = "CLiteRTLM",
    bundle_id = "com.patricedery.agello.CLiteRTLM",
    features = ["exported_symbols"],
    infoplists = ["//agello:Info.plist"],
    macos = ["arm64"],
    minimum_os_versions = {
        "macos": "27.0",
    },
    public_hdrs = [
        "//c:api_export.h",
        "//c:conversation.h",
        "//c:embedding_engine.h",
        "//c:engine.h",
        "//c:error_reporter.h",
        "//c:experimental.h",
        "//c:model_info.h",
        "//c:telemetry.h",
        "//c:build_info.h",
    ],
    deps = ["//swift:CLiteRTLM_objc"],
)
