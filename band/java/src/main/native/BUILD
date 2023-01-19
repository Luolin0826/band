load("//band:band.bzl", "band_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "native_hdrs",
    srcs = [
        "jni_utils.h",
    ],
)

filegroup(
    name = "native_srcs",
    srcs = [
        "jni_utils.cc",
        "band_jni.cc",
        "config_jni.cc",
        "nativeconfigbuilderwrapper_jni.cc",
        "nativeenginewrapper_jni.cc",
        "nativemodelwrapper_jni.cc",
        "nativetensorwrapper_jni.cc",
    ],
)

cc_library(
    name = "native",
    srcs = [":native_srcs"],
    hdrs = [":native_hdrs"],
    copts = band_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//band/java/jni",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "exported_symbols.lds",
        "version_scripts.lds",
    ]
)