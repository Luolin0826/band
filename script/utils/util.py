# Copyright 2023 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import platform as plf
import shlex
import shutil
import subprocess
from typing import Literal, overload
from .docker import run_cmd_docker

PLATFORM = {
    "linux": "linux_x86_64",
    "windows": "windows",
    "android": "android_arm64",
}


def is_windows():
    return plf.system() == "Windows"


def is_linux():
    return plf.system() == "Linux"


def is_cygwin():
    return plf.system().startswith("CYGWIN_NT")


def get_platform():
    return plf.system().lower()


def canon_path(path):
    if is_windows() or is_cygwin():
        return path.replace('\\', '/')
    return path


def run_cmd(cmd, capture_output=False):
    args = shlex.split(cmd)
    proc = subprocess.run(args, cwd=os.getcwd(), check=True,
                          capture_output=capture_output, text=True)
    if capture_output:
        return proc.stdout


def copy(src, dst):
    subprocess.check_call(['mkdir', '-p', f'{os.path.normpath(dst)}'])
    # append filename to dst directory

    dst = canon_path(os.path.join(dst, os.path.basename(src)))

    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore_dangling_symlinks=True, symlinks=True)
    else:
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            print(dst)
            shutil.copy(src + ".exe", dst + ".exe")


def get_bazel_options(
        debug: bool,
        trace: bool, 
        platform: str,
        backend: str):
    opt = ""
    opt += "-c " + ("dbg" if debug else "opt") + " "
    opt += "--strip " + ("never" if debug else "always") + " "
    opt += f"--config {PLATFORM[platform]}" + ("" if backend == "none" else f" --config {backend}") + ("" if trace == False else " --config trace") + " "
    return opt


def make_cmd(
        build_only: bool, 
        debug: bool,
        trace: bool, 
        platform: str, 
        backend: str, 
        target: str,
    ):
    cmd = "bazel" + " "
    cmd += ("build" if build_only else "test") + " "
    cmd += get_bazel_options(debug, trace, platform, backend)
    cmd += target
    return cmd


def get_dst_path(base_dir, target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(base_dir, target_platform, build)
    return canon_path(path)


ANDROID_BASE = '/data/local/tmp/'


def push_to_android(src, dst):
    run_cmd(f'adb -d push {src} {ANDROID_BASE}{dst}')


def run_binary_android(basepath, path, option='', run_as_su=False):
    run_on_android(f'chmod 777 {ANDROID_BASE}{basepath}{path}', run_as_su=run_as_su)
    run_on_android(f'./{path} {option}', cwd=f'{ANDROID_BASE}{basepath}', run_as_su=run_as_su)


@overload
def run_on_android(cmd, cwd='.', run_as_su=False, capture_output: Literal[False] = ...) -> None:
    ...


@overload
def run_on_android(cmd, cwd='.', run_as_su=False, capture_output: Literal[True] = ...) -> str:
    ...


def run_on_android(cmd, cwd='.', run_as_su=False, capture_output: bool = False):
    if "'" in cmd or '"' in cmd:
        raise ValueError('cmd should not contain " or \'')
    if run_as_su:
        cmd = f'su -c "{cmd}"'
    try:
        return run_cmd(f'adb -d shell \'cd {cwd} && {cmd}\'', capture_output=capture_output)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e


def get_argument_parser(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-B', '--backend', type=str,
                        default='tflite', help='Backend <tflite|none>')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Test on Android (with adb)')
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries for android from docker (assuming that the current docker engine has devcontainer built with a /.devcontainer')
    parser.add_argument('-s', '--ssh', required=False, default=None, help="SSH host name (e.g. dev@ssh.band.org)")
    parser.add_argument('-t', '--trace', action="store_true", default=True, help='Build with trace (default = True)')
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android ')
    parser.add_argument('-b', '--build', action="store_true", default=False,
                        help='Build only, only affects to local (default = false)')
    parser.add_argument('-su', '--run_as_su', action=argparse.BooleanOptionalAction, default=True)
    return parser


def clean_bazel(docker):
    if docker:
        run_cmd_docker('bazel clean')
    else:
        run_cmd('bazel clean')
