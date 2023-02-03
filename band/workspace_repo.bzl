"""Initialize Band workspace"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/android:android.bzl", android = "repo")

def workspace():
    """Workspace initialization for dependencies."""

    ######## Remote repositories ########
    # TODO(widiba03304): Uncomment when abseil comes
    # http_archive(
    #     name = "com_google_absl",
    #     url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.0.tar.gz",
    #     sha256 = "3ea49a7d97421b88a8c48a0de16c16048e17725c7ec0f1d3ea2683a2a75adc21",
    #     strip_prefix = "abseil-cpp-20230125.0",
    # )

    http_archive(
        name = "build_bazel_rules_android",
        url = "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
    )

    http_archive(
        name = "jsoncpp",
        url = "https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
    )

    ######## Android repositories ########
    android(name = "android_repo")

workspace_repo = workspace