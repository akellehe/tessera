# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Drift-free holed 2-surface fixture for the k=1 period-realizability tests.

The retired ``merge_cobordism`` substrate (deleted with ``MergeCobordism`` /
``TransportCobordism`` in #491) supplied a holed complex with a b1 register so the
analytic ``r_U`` / ``r_psi`` gradients could be checked against finite differences.
This fixture supplies the same thing from first principles: a standard icosahedron
(closed S^2) with three mutually vertex-disjoint triangular windows removed.

Removing ``n`` disjoint triangles from S^2 leaves ``b1 = n - 1``; with three
windows the carried representative is 2-dimensional while there are three
hole-circles, so the period system is overdetermined and a generic target is
*non-realizable* (``r_U > 0``) -- exactly the regime the gradient correctness
checks need, while a carried period row stays realizable (``r_U ~ 0``).
"""

import numpy as np

import tessera
import cmath

cob = tessera.cobordism

# Standard icosahedron: 12 vertices, 20 triangular faces.
_ICOSA_FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
# Three mutually vertex-disjoint faces -> b1 = 2, three hole-circles.
_WINDOWS = [[0, 11, 5], [3, 2, 6], [9, 8, 1]]


def holed_surface(degree=1, jitter=True, metric_source=None):
    """Return ``(st, es, holes, P)``: the holed icosahedron, an
    ``EigenstateSynthesis`` at register ``degree`` (on ``metric_source``, the
    process default when None), the removed window triangles (the
    hole-circles), and the carried period matrix of shape ``(b_k, n_holes)``."""
    rm = {tuple(sorted(t)) for t in _WINDOWS}
    holed = [list(f) for f in _ICOSA_FACES if tuple(sorted(f)) not in rm]
    st = tessera.Spacetime.fromVertexTuples(2, holed, 1.0, 0.0)
    if jitter:
        for i, e in enumerate(st.getEdgeList().toVector()):
            e.setLength(cmath.sqrt(complex(1.0 + 0.013 * (i % 6))))
    st.materializeFacets()
    es = (cob.EigenstateSynthesis(st, degree) if metric_source is None
          else cob.EigenstateSynthesis(st, degree, metric_source))
    holes = [list(t) for t in _WINDOWS]
    periods = np.asarray(es.cyclePeriods(holes), complex)
    n = len(holes)
    return st, es, holes, periods.reshape(len(periods) // n, n)
