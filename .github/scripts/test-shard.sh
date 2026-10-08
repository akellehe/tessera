#!/usr/bin/env bash
# Print the pytest-split options that select this job's shard of the test
# suite, or nothing when the suite is not sharded.
#
# TESSERA_TEST_SPLITS is the number of shards the suite is divided into and
# TESSERA_TEST_GROUP is which of them (counted from 1) this job runs. build.yml
# sets both from the matrix its `shards` job builds. With one shard nothing is
# printed: the plugin stays dormant and the pytest command line is the one an
# unsharded job runs.
#
# .test_durations at the repository root is pytest-split's JSON record of each
# test's duration, from which it forms shards of about equal expected runtime.
# When the file is absent the plugin loads no durations (pytest_split/plugin.py
# reads a missing file as an empty record), gives every test the same weight
# and divides the collected tests evenly by count (pytest_split/algorithms.py
# weights a test without a recorded duration at the average of the recorded
# ones, or at 1 when there are none). A test the file does not list is
# weighted at that average.
set -euo pipefail

splits="${TESSERA_TEST_SPLITS:-1}"
group="${TESSERA_TEST_GROUP:-1}"

if [ "${splits}" -le 1 ]; then
    exit 0
fi

if [ ! -f .test_durations ]; then
    echo "test-shard.sh: .test_durations is absent; pytest-split divides the tests evenly by count" >&2
fi

echo "--splits ${splits} --group ${group} --durations-path .test_durations"
