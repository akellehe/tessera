#!/usr/bin/env bash
# Syntax-check the translation units that build only against libtorch.
#
# CMakeLists.txt adds src/rl/ (the RL harness and its _tessera_rl module) only
# when the configuring Python can import torch, and the test jobs build without
# torch, so nothing else in CI compiles these files. A rename elsewhere in the
# tree can then break them unnoticed: CobordismObjectiveEnv.cpp stopped
# compiling when MultiCobordism's enums moved namespace (fixed in
# https://github.com/akellehe/tessera/pull/1530).
#
# Usage: check-torch-sources.sh <python with torch> [build dir]
#
# Configures the project with that Python, which makes CMake find libtorch and
# export a compile command for every src/rl/ unit, then runs each command with
# -fsyntax-only: the full parse and template instantiation, without code
# generation, so a unit takes seconds and no object file is written.
set -euo pipefail
python="$1"
build_dir="${2:-build-torch-sources}"

"$python" -c "import torch; print('torch', torch.__version__)"
cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPython_EXECUTABLE="$python" \
  -DTESSERA_CUDA=OFF \
  -DTESSERA_BUILD_TESTS=OFF > "$build_dir.configure.log" 2>&1 \
  || { cat "$build_dir.configure.log"; exit 1; }
grep -q "Building RL subsystem" "$build_dir.configure.log" \
  || { echo "::error::CMake did not find libtorch; nothing was checked"; cat "$build_dir.configure.log"; exit 1; }

"$python" - "$build_dir/compile_commands.json" <<'PY'
import json
import shlex
import subprocess
import sys

entries = [e for e in json.load(open(sys.argv[1])) if "/src/rl/" in e["file"]]
if not entries:
    sys.exit("::error::the compile database has no src/rl/ unit")
failed = []
for entry in sorted(entries, key=lambda e: e["file"]):
    args = shlex.split(entry["command"])
    # Drop the object output and the dependency-file options; check the parse only.
    keep, skip = [], 0
    for arg in args:
        if skip:
            skip -= 1
            continue
        if arg in ("-o", "-MF", "-MT", "-MQ"):
            skip = 1
            continue
        if arg in ("-MD", "-MMD"):
            continue
        keep.append(arg)
    result = subprocess.run(keep + ["-fsyntax-only"], cwd=entry["directory"],
                            capture_output=True, text=True)
    unit = entry["file"].split("/src/rl/", 1)[1]
    if result.returncode == 0:
        print(f"ok src/rl/{unit}")
    else:
        failed.append(unit)
        print(f"FAIL src/rl/{unit}")
        print(result.stderr)
print(f"{len(entries) - len(failed)} of {len(entries)} Torch-only units parse")
sys.exit(1 if failed else 0)
PY
