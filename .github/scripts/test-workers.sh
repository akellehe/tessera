#!/usr/bin/env bash
# Print the number of pytest-xdist workers the test lanes should run with.
#
# The count is TESSERA_TEST_CPUS / TESSERA_TEST_THREADS_PER_WORKER, clamped to
# at least one worker and to at most the machine's real CPU count.
#
# This lives in a script rather than inline in build.yml because the obvious
# inline form was silently wrong. GNU coreutils' `nproc` reports the OpenMP
# thread limit, not the CPU count, whenever OMP_NUM_THREADS is set:
#
#     $ nproc                      # 32
#     $ OMP_NUM_THREADS=4 nproc    # 4
#     $ OMP_NUM_THREADS=4 nproc --all  # 32
#
# The test steps set OMP_NUM_THREADS in their own `env:` block, so a
# `$(nproc) / OMP_NUM_THREADS` expression evaluated to 1 and the suite ran
# single-worker. `nproc --all` ignores the OpenMP variables and is the only
# correct spelling here.
set -euo pipefail

budget="${TESSERA_TEST_CPUS:-16}"
per_worker="${TESSERA_TEST_THREADS_PER_WORKER:-4}"
cores="$(nproc --all)"

if [ "${budget}" -gt "${cores}" ]; then
    budget="${cores}"
fi

workers=$(( budget / per_worker ))
if [ "${workers}" -lt 1 ]; then
    workers=1
fi

echo "${workers}"
