import os
import platform
import argparse


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_cygwin():
    return platform.system().startswith("CYGWIN_NT")


def get_home():
    global env
    if is_windows() or is_cygwin():
        home_path = env['APPDATA'].replace('\\', '/')
    elif is_linux():
        home_path = env['HOME']
    else:
        raise EnvironmentError("Only Linux or Windows is supported to build.")
    return home_path


def get_var(var, default):
    global env
    if env.get(var) is not None:
        return env.get(var)
    else:
        print(f"{var} is not set. {default} is set by default.")
        return default


def validate_android_tools(sdk_path, ndk_path):
    if not os.path.exists(sdk_path):
        return False
    if not os.path.exists(ndk_path):
        return False
    return True

def validate_android_build_tools(sdk_path, version):
    build_tools_path = os.path.exists(os.path.join(sdk_path, 'build-tools'))
    if not build_tools_path:
        return False
    versions = sorted(os.listdir(build_tools_path))
    if version not in versions:
        return False
    return True

def validate_android_api_level(sdk_path, api_level):
    if not os.path.exists(os.path.join(sdk_path, 'android-' + api_level)):
        return False
    return True

def main():
    global env
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str,
                        default=os.path.abspath(os.path.dirname(__file__)))
    args = parser.parse_args()

    ROOT_DIR = args.workspace

    env = dict(os.environ)

    android_build_tools_version = get_var(
        "ANDROID_BUILD_TOOLS_VERSION", "30.0.0")

    android_sdk_path = get_var(
        "ANDROID_SDK_HOME", os.path.join(get_home(), "Android/Sdk"))
    android_ndk_path = get_var("ANDROID_NDK_HOME", os.path.join(
        get_home(), "Android/Sdk/ndk-bundle"))

    if not validate_android_tools(android_sdk_path, android_ndk_path):
        raise EnvironmentError(
            "Cannot find android sdk & ndk. Please check ANDROID_SDK_HOME & ANDROID_NDK_HOME.")
    platforms = os.path.join(android_sdk_path, 'platforms')
    api_levels = sorted(os.listdir(platforms))
    api_levels = [level.replace('android-', '') for level in api_levels]

    # ANDROID_API_LEVEL
    android_api_level = get_var("ANDROID_API_LEVEL", "28")

    # ANDROID_NDK_API_LEVEL
    android_ndk_api_level = get_var("ANDROID_NDK_API_LEVEL", "21")


if __name__ == "__main__":
    main()
