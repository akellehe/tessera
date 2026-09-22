# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The libtorch reinforcement-learning harness — PPO over ``MultiCobordism.buildStep``.

Optional: requires the ``rl`` extra (libtorch), which builds the ``_tessera_rl`` extension.
Everything here is a HARNESS that DRIVES the canonical ``MultiCobordism`` + ``ProtonSynthesis`` engine
(the sole source of truth for proton construction) — it never reimplements construction.

Import as ``tessera.rl``::

    import tessera
    cfg = tessera.rl.carry_profile_env()
    res = tessera.rl.benchmark(cfg, tessera.rl.carry_profile_train(), formation=True,
                               checkpoint_path="proton_policy.pt")
"""
import os

# tessera_core (loaded by `import tessera`) links Homebrew's LLVM libomp; libtorch ships its
# own copy of the *same* LLVM OpenMP runtime. Loading `_tessera_rl` pulls libtorch's libomp
# into a process that already has the core's, so TWO OpenMP runtimes coexist. That breaks in
# two stages: the second runtime's init aborts the process ("OMP: Error #15 ... multiple
# copies of the OpenMP runtime"), and merely silencing that abort is not enough — the two
# runtimes then fight over a shared thread pool and DEADLOCK inside any threaded region (e.g.
# libtorch's log_softmax during PPO.update parks forever in __kmp_join_barrier).
#
# Both fixes below are needed and both use setdefault so an explicit environment override
# still wins. KMP_DUPLICATE_LIB_OK tolerates the duplicate init (benign: same implementation).
# OMP_NUM_THREADS=1 keeps every OpenMP region single-threaded, which sidesteps the cross-
# runtime barrier deadlock entirely. This is tessera's default scan policy anyway (the core's
# heat-kernel loop is bit-identical to serial at 1 thread), and this RL workload is bound by
# MultiCobordism.buildStep in the core, not by libtorch ops on the tiny actor-critic — so the
# single-thread cap costs effectively nothing here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load libtorch into the process BEFORE the extension. `_tessera_rl` links libtorch and
# carries an RPATH to it, but when built under standard build isolation that RPATH points at
# the (now-deleted) isolated build environment — so it cannot be relied on. Importing torch
# first pulls libtorch_cpu.so / libc10.so into the process by their sonames; the dynamic
# loader then reuses those already-loaded libraries to satisfy `_tessera_rl`'s DT_NEEDED
# entries regardless of RPATH. This must come AFTER the OMP setdefaults above so they take
# effect before libtorch initializes its OpenMP runtime.
try:
    import torch  # noqa: E402,F401
except ModuleNotFoundError as exc:  # pragma: no cover - guidance for a half-set-up env
    raise ModuleNotFoundError(
        "tessera.rl requires the optional 'rl' extra (libtorch). Build it with:\n"
        '    TESSERA_RL=1 pip install -e ".[rl]"'
    ) from exc

from tessera._tessera_rl import *  # noqa: E402,F401,F403
