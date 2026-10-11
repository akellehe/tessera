# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Shared refined closed-S⁴ test host.

The bare ∂Δ⁵ sphere refined by ``n_refine`` PreGeometric stellar Pachner adds (so
surgery has room to act — the minimal triangulation is too small), then given a mild
deterministic non-uniform metric. Built via the public bindings — a local fixture
replacing the retired ``emergent_optimizer.build_closed_s4``.

One copy on purpose (the ``_holed_surface`` pattern): the golden constants in
``test_multi_cobordism_python.py`` are pinned to ``closed_s4(20, 3)`` and the
causal-guard suite builds smaller instances of the same host — keep this fixture
byte-stable so the two suites cannot silently drift apart.
"""

import tessera as T
import cmath


def closed_s4(n_refine=20, seed=3):
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT, 1.0, 1.0,
                     T.PREFERRED, T.SimplexBoundarySphere(4))
    st.build()
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(1.0)))
    applied = 0
    for s in range(seed, seed + n_refine * 4):
        mv = T.AddMove(st, s, False, T.PachnerMode.PreGeometric, False)
        if mv.propose() and mv.apply():
            applied += 1
        if applied >= n_refine:
            break
    for i, e in enumerate(st.get_edge_list().to_vector()):
        e.set_length(cmath.sqrt(complex(1.0 + 0.01 * (i % 6))))
    return st
