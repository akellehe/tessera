# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""run_stage2's trials are unbounded in complex z=l² — no clamp or causal guard
(#565, #589) — plus signature-change verification of every objective reader.

Semantics: `runStage2` in MultiCobordism.h is THE authoritative statement. The
configuration space is the full complex squared interval z=l². A trial may land
spacelike, timelike, lightlike, or off the real Lorentzian locus. The line search
backs off its real step scale but never projects a coordinate onto a floor, cap,
real axis, or cone side, so the old clamp pins (`0.05`, `20`) must never reappear
on an edge.

Epic #559's rule: NO timelike initialization — causal content may only EMERGE. The
timelike edge hand-set below is a verification of the READERS (the complex Sorkin
deficit, the dual Regge action, the exact-gradient objective) across a signature
change — not an initialization policy.
"""
import cmath
import math
import os
import sys
import unittest

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _closed_s4 import closed_s4 as _closed_s4  # noqa: E402  (the shared host fixture)

cob = T.cobordism

# The RETIRED clamp's pins: with the old projection every accepted line-search sweep
# rewrote every out-of-range edge to exactly one of these. Their exact absence after
# an accepted step is the sharpest executable witness that no clamping happens.
_OLD_FLOOR = 0.05
_OLD_CAP = 20.0

#: The rotation epsilon of the Lorentzian family the seeded hosts are built at
#: (integration specification, Requirement 2). A real Lorentzian seed sits on
#: the boundary of the Kontsevich-Segal allowable domain, the set of complex
#: metrics whose Gram eigenvalues on every top simplex satisfy
#: sum_i |arg lambda_i| < pi; the default Whitney metric's configuration space
#: is the closure of that domain.
_EPSILON = 0.1


def _seed_rotated(host, seeds, epsilon=_EPSILON):
    """Write the squared length s of each seeded edge, with its declared
    timelike part tau, as the member at rotation epsilon of its Lorentzian
    family: s(epsilon) = (s - tau) + exp(-2 i epsilon) tau. `seeds` maps an
    edge index to (s, tau); every other edge keeps its length and has no
    timelike part. Returns the host's Kontsevich-Segal margin, the minimum
    over top simplices of pi - sum_i |arg lambda_i|."""
    edges = host.getEdgeList().toVector()
    squared = [complex(e.getLength()) ** 2 for e in edges]
    timelike = [0j] * len(edges)
    for index, (value, part) in seeds.items():
        squared[index] = complex(value)
        timelike[index] = complex(part)
    rotated = T.chainhodge.LorentzianFamily.rotate(squared, timelike, epsilon)
    for edge, value in zip(edges, rotated):
        edge.setLength(cmath.sqrt(complex(value)))
    return cob.HodgeLaplacian.kontsevichSegalMargin(host)


def _sphere4(jitter=True):
    """A minimal triangulated S⁴ (boundary of a 5-simplex), unit spacelike, lightly
    jittered so the dual Regge action is nontrivial."""
    sig = T.Signature(4, T.Lorentzian)
    st = T.Spacetime(T.Metric(True, sig), T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for i, e in enumerate(st.getEdgeList().toVector()):
        e.setLength(cmath.sqrt(complex(1.0 + (0.013 * (i % 5) if jitter else 0.0))))
    return st


def _hinges(st):
    """The (d-2)-simplices, exactly as ``ReggeSolver::collectHinges``: top simplices
    have d+1 vertices, so a hinge has (top_verts - 2) vertices (triangles in 4D)."""
    sims = list(st.getSimplices())
    hinge_nverts = max(len(s.getVertices()) for s in sims) - 2
    return [s for s in sims if len(s.getVertices()) == hinge_nverts]


class SignatureChangeReadersTest(unittest.TestCase):
    """The load-bearing physics check: every reader the stage-2 objective is built
    from stays sane when one edge is hand-set timelike (Re l^2 < 0)."""

    def test_readers_survive_one_timelike_edge(self):
        w = cmath.exp(2j * math.pi / 3)
        st = _sphere4()
        rs = T.ReggeSolver(st, T.MatterConfiguration())
        action_before = complex(rs.dualReggeAction())
        self.assertTrue(cmath.isfinite(action_before))

        # Hand-set ONE edge timelike — a reader verification, not initialization.
        st.getEdgeList().toVector()[3].setLength(cmath.sqrt(complex(complex(-0.8, 0.0))))

        # (a) The complex Sorkin/Asante–Dittrich deficit is sane on EVERY hinge.
        for h in _hinges(st):
            eps = complex(h.deficitAngle())
            self.assertTrue(cmath.isfinite(eps),
                            f"non-finite deficit {eps} on hinge "
                            f"{[v.getId() for v in h.getVertices()]}")

        # (b) The dual Regge action is finite and its Im part RESPONDS to the
        # signature change (the boost branch wakes up; Lorentzian action is complex —
        # never Re-only).
        action_after = complex(rs.dualReggeAction())
        self.assertTrue(cmath.isfinite(action_after))
        self.assertGreater(abs(action_after.imag - action_before.imag), 1e-9,
                           "Im(dual Regge action) did not respond to the "
                           f"signature change: {action_before} -> {action_after}")

        # (c) The exact-gradient term and the full objective stay finite.
        grad_sq = cob.MultiCobordism.regge_action_gradient(st)
        self.assertTrue(math.isfinite(grad_sq))
        opt = cob.MultiCobordism(st, [[1, w, w * w], [1, w * w, w]], [[1, w, w * w]],
                                 degrees=[3], gamma=1.0, seed=0)
        opt.seed_inputs([v.getId() for v in st.getVertexList().toVector()][:2])
        self.assertTrue(math.isfinite(opt.objective()))


class Stage2UnclampedTest(unittest.TestCase):
    """No projection exists: neither the API surface nor the dynamics may clamp."""

    @classmethod
    def setUpClass(cls):
        cls.w = cmath.exp(2j * math.pi / 3)

    def _node(self, host, seed=3, **source):
        w = self.w
        opt = cob.MultiCobordism(host, [[1, w, w * w], [1, w * w, w]],
                                 [[1, w, w * w]], degrees=[3], gamma=1.0, seed=seed,
                                 **source)
        opt.seed_inputs([v.getId() for v in host.getVertexList().toVector()][:2])
        return opt

    def test_projection_api_is_gone(self):
        # The clamp/guard machinery must not exist at all — reintroducing any of it
        # (a flag, a projection helper, a band) fails here first.
        for name in ("set_causal_guard", "causal_guard_epsilon",
                     "bounded_trial_real_part"):
            self.assertFalse(hasattr(cob.MultiCobordism, name),
                             f"clamping machinery reappeared: {name}")

    def test_accepted_steps_never_pin_to_the_old_clamp(self):
        # Seed edges on the wrong side of every retired bound: timelike (the reader-
        # verification allowance), timelike beyond the old -20 cap, and inside the
        # old 0.05 floor band. The retired clamp rewrote ALL of them to exactly a pin
        # value on the FIRST accepted sweep; unclamped descent may move them, but
        # never to a pin. Requires an accepted step (len(trace) >= 2) to be
        # meaningful.
        #
        # The two timelike edges are all timelike part and are rotated at
        # _EPSILON, which puts the host inside the allowable domain (margin
        # 0.00959; 0 unrotated). A spacelike edge beyond the +20 cap cannot be
        # seeded there: at s = 25 among edges near 1 it breaks the triangle
        # inequality, its twelve top simplices are Lorentzian with no declared
        # timelike part, and they stay at margin 0 at every rotation.
        host = _closed_s4(n_refine=8, seed=3)
        margin = _seed_rotated(host, {5: (-0.8, -0.8),     # timelike
                                      7: (-25.0, -25.0),   # beyond the old cap
                                      9: (0.01, 0.0)})     # inside the old floor band
        self.assertAlmostEqual(margin, 0.009592433696182567, places=9)
        opt = self._node(host)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05, tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        self.assertGreaterEqual(len(trace), 2, "no accepted step — vacuous run")
        # Measured on the default Whitney metric: three accepted steps,
        # 1745.9421 -> 143.1963 -> 111.9970 -> 97.2480, ending inside the
        # domain (margin 1.27e-4).
        self.assertGreater(cob.HodgeLaplacian.kontsevichSegalMargin(opt.st), 0.0)
        for e in opt.st.getEdgeList().toVector():
            sq = complex(e.getLength()**2)
            self.assertTrue(cmath.isfinite(sq))
            for pin in (_OLD_FLOOR, _OLD_CAP, -_OLD_CAP):
                self.assertGreater(abs(sq.real - pin), 1e-12,
                                   f"edge pinned to the retired clamp value {pin}")

    def test_lightlike_band_is_admissible(self):
        # An edge INSIDE any would-be degeneracy band (|Re l^2| = 1e-6): stage 2 must
        # run it as-is — no push-out, no floor, no exception. Whether descent moves it
        # is dynamics; snapping it to a pin would be a projection.
        host = _closed_s4(n_refine=8, seed=3)
        host.getEdgeList().toVector()[3].setLength(cmath.sqrt(complex(complex(1e-6, 0.0))))
        opt = self._node(host)
        trace = opt.run_stage2(beta=1.0, max_iters=2, alpha0=0.05, tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        sq = complex(opt.st.getEdgeList().toVector()[3].getLength()**2)
        self.assertTrue(cmath.isfinite(sq))
        self.assertGreater(abs(sq.real - _OLD_FLOOR), 1e-12,
                           "the lightlike-band edge was snapped to the old floor")

    def test_unclamped_step_with_timelike_edge_no_nan(self):
        # A stage-2 run on a host carrying one hand-set timelike edge neither NaNs
        # nor collapses: finite F trace, every edge finite. Cone-side changes and
        # magnitudes and complex phases are dynamics; only the line search's
        # variational acceptance decides (#589). The timelike edge is rotated
        # at _EPSILON, inside the allowable domain (margin 0.154).
        host = _closed_s4(n_refine=8, seed=3)
        self.assertGreater(_seed_rotated(host, {5: (-0.8, -0.8)}), 0.0)
        opt = self._node(host)
        trace = opt.run_stage2(beta=1.0, max_iters=3, alpha0=0.05, tolerance=1e-9)
        self.assertTrue(all(math.isfinite(f) for f in trace))
        for e in opt.st.getEdgeList().toVector():
            self.assertTrue(cmath.isfinite(complex(e.getLength()**2)))


if __name__ == "__main__":
    unittest.main()
