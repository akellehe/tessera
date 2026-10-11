# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.

"""OpenMP parallelization of the Regge hinge loop.

``ReggeSolver::actionGradientExact`` and ``action_hessian_exact`` sum independent
per-hinge contributions into a shared accumulator. The C++ implementation
parallelizes that hinge loop with per-thread partial accumulators reduced
deterministically into the shared result (see ``src/simulations/ReggeSolver.cpp``).

The optimization lives entirely in C++; this module only *tests* it. The serial
reference oracles below re-accumulate the per-hinge contributions in a plain
Python loop using the exposed per-hinge methods (``deficit_angle``,
``dual_volume``, and their gradients/Hessians). The per-hinge *values* come from
C++; only the *reduction order* differs between the serial reference and the
parallel reduction — which is exactly what parallelization changes. So agreement
to floating-point round-off is the correctness criterion ("identical values vs
serial"), and bit-identical results from repeated/concurrent runs is the
determinism criterion (the per-thread split must not leak thread-timing into the
answer).

Coverage:
  * gradient and Hessian match an independent serial Python reduction (Lorentzian
    CDT meshes — both complex/boost and real triangle hinges);
  * the parallel Hessian stays symmetric;
  * repeated calls in one process are bit-identical (determinism);
  * results are invariant across OMP_NUM_THREADS and bit-identical across
    processes at a fixed thread count, including oversubscription (subprocess);
  * empty/zero-hinge meshes return correctly (every thread idle);
  * near-linear gradient speedup with thread count (reported benchmark).
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import tessera

_HERE = os.path.dirname(os.path.abspath(__file__))

# Tight enough to catch a real reduction bug (those are O(1), like a dropped or
# double-counted hinge), loose enough to absorb the round-off from reassociating
# the per-edge sum across threads (and any FMA contraction in the C++ build).
_RTOL = 1e-9
_ATOL = 1e-11


# --------------------------------------------------------------------------- #
# Mesh builders (shared by the in-process tests and the subprocess driver)
# --------------------------------------------------------------------------- #
def _new_spacetime():
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                             tessera.Toroid())


def _make_cdt(n_simplices):
    st = _new_spacetime()
    st.build(n_simplices)
    st.materialize_facets()
    return st


def _complex_mesh():
    """A Lorentzian CDT mesh sized so its triangle hinges carry genuinely complex
    (boost) deficits -- exercises the complex branch of the parallel reduction."""
    return _make_cdt(200)


# --------------------------------------------------------------------------- #
# Serial reference oracles — plain Python re-accumulation of the per-hinge terms
# --------------------------------------------------------------------------- #
def _edge_index(st):
    edges = st.get_edge_list().to_vector()
    eidx = {}
    for i, e in enumerate(edges):
        a, b = e.get_source().get_id(), e.get_target().get_id()
        eidx[(min(a, b), max(a, b))] = i
    return edges, eidx


def _collect_hinges(st):
    """The (d-2)-simplices, exactly as ``ReggeSolver::collectHinges``. Metric-free:
    top simplices have d+1 vertices, so a hinge has (top_verts - 2) vertices
    (= d - 1), i.e. edges in 3D and triangles in 4D."""
    sims = list(st.get_simplices())
    if not sims:
        return []
    hinge_nverts = max(len(s.get_vertices()) for s in sims) - 2
    return [s for s in sims if len(s.get_vertices()) == hinge_nverts]


def _serial_gradient(st):
    """dS/dl^2_e = sum_h [ d|*h|/dl^2_e * eps_h + |*h| * d eps_h/dl^2_e ],
    accumulated serially in Python."""
    edges, eidx = _edge_index(st)
    g = [0j] * len(edges)
    for h in _collect_hinges(st):
        eps = complex(h.deficit_angle())
        dv = complex(h.dual_volume())
        for e, d_eps in h.deficit_angle_gradient().items():
            i = eidx.get(e)
            if i is not None:
                g[i] += dv * complex(d_eps)
        for e, d_dv in h.dual_volume_gradient().items():
            i = eidx.get(e)
            if i is not None:
                g[i] += complex(d_dv) * eps
    return np.array(g, dtype=complex)


def _serial_hessian(st):
    """d^2 S/dl^2_e dl^2_f = sum_h [ d2V*eps + dV_e*dEps_f + dV_f*dEps_e
    + V*d2Eps ], accumulated serially in Python. Dense E x E; small meshes only."""
    edges, eidx = _edge_index(st)
    E = len(edges)
    H = np.zeros((E, E), dtype=complex)
    for h in _collect_hinges(st):
        eps = complex(h.deficit_angle())
        V = complex(h.dual_volume())
        d_eps = {k: complex(v) for k, v in h.deficit_angle_gradient().items()}
        d_v = {k: complex(v) for k, v in h.dual_volume_gradient().items()}
        d2_eps = {k: complex(v) for k, v in h.deficit_angle_hessian().items()}
        d2_v = {k: complex(v) for k, v in h.dual_volume_hessian().items()}
        for e, dVe in d_v.items():
            ie = eidx.get(e)
            if ie is None:
                continue
            dEe = d_eps.get(e, 0j)
            for f, dVf in d_v.items():
                jf = eidx.get(f)
                if jf is None:
                    continue
                dEf = d_eps.get(f, 0j)
                term = dVe * dEf + dVf * dEe
                term += d2_v.get((e, f), 0.0) * eps
                term += V * d2_eps.get((e, f), 0j)
                H[ie, jf] += term
    return H


def _parallel_gradient(st):
    rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
    return np.array([complex(z) for z in rs.action_gradient_exact()], dtype=complex)


def _parallel_hessian(st):
    rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
    H = rs.action_hessian_exact()
    return np.array([[complex(z) for z in row] for row in H], dtype=complex)


# --------------------------------------------------------------------------- #
# In-process correctness: parallel C++ == independent serial Python reduction
# --------------------------------------------------------------------------- #
class GradientCorrectnessTest(unittest.TestCase):
    def _assert_matches_serial(self, st):
        par = _parallel_gradient(st)
        ser = _serial_gradient(st)
        self.assertEqual(par.shape, ser.shape)
        worst = float(np.max(np.abs(par - ser))) if par.size else 0.0
        self.assertTrue(
            np.allclose(par, ser, rtol=_RTOL, atol=_ATOL),
            f"parallel gradient disagrees with serial reference: max|d|={worst:.3e}")
        return par

    def test_matches_serial_reference_complex(self):
        # Lorentzian CDT: triangle hinges with genuinely complex (boost) deficits.
        par = self._assert_matches_serial(_complex_mesh())
        self.assertGreater(
            float(np.max(np.abs(par.imag))), 0.1,
            "gradient should be materially complex (boost hinges)")

    def test_matches_serial_reference_cdt(self):
        # A second, larger CDT mesh — more hinges across the parallel reduction.
        self._assert_matches_serial(_make_cdt(600))

    def test_length_matches_edge_count(self):
        st = _make_cdt(60)
        edges, _ = _edge_index(st)
        self.assertEqual(_parallel_gradient(st).shape[0], len(edges))


class HessianCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = _complex_mesh()
        cls.par = _parallel_hessian(cls.st)

    def test_matches_serial_reference_complex(self):
        ser = _serial_hessian(self.st)
        self.assertEqual(self.par.shape, ser.shape)
        worst = float(np.max(np.abs(self.par - ser)))
        self.assertTrue(
            np.allclose(self.par, ser, rtol=_RTOL, atol=_ATOL),
            f"parallel Hessian disagrees with serial reference: max|d|={worst:.3e}")

    def test_symmetric(self):
        # The analytic Hessian is symmetric; the per-thread reduction must keep it.
        worst = float(np.max(np.abs(self.par - self.par.T)))
        self.assertLess(worst, 1e-9, f"Hessian not symmetric: max|H-H^T|={worst:.3e}")

    def test_shape_is_square_edges(self):
        edges, _ = _edge_index(self.st)
        self.assertEqual(self.par.shape, (len(edges), len(edges)))


# --------------------------------------------------------------------------- #
# Determinism in one process — repeated calls are bit-identical
# --------------------------------------------------------------------------- #
class DeterminismWithinProcessTest(unittest.TestCase):
    def test_gradient_bit_identical_across_calls(self):
        st = _complex_mesh()
        rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        a = np.array([complex(z) for z in rs.action_gradient_exact()])
        b = np.array([complex(z) for z in rs.action_gradient_exact()])
        self.assertTrue(np.array_equal(a, b),
                        "gradient not bit-identical across repeated calls")

    def test_hessian_bit_identical_across_calls(self):
        st = _complex_mesh()
        rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        a = np.array([[complex(z) for z in r] for r in rs.action_hessian_exact()])
        b = np.array([[complex(z) for z in r] for r in rs.action_hessian_exact()])
        self.assertTrue(np.array_equal(a, b),
                        "Hessian not bit-identical across repeated calls")


# --------------------------------------------------------------------------- #
# Thread-count behavior — runs the subprocess driver under different
# OMP_NUM_THREADS. This is the genuine "identical values vs serial" test:
# OMP_NUM_THREADS=1 IS the serial reduction.
# --------------------------------------------------------------------------- #
_DRIVER_CACHE = {}


def _run_driver(mesh, n, threads, want_hess, tag=0):
    """Run this file as a subprocess driver under a fixed OMP_NUM_THREADS and
    return (grad, hess_or_None, compute_seconds). ``tag`` forces a distinct run
    of an otherwise-identical configuration (used to check cross-process
    determinism). Results are memoized per (mesh, n, threads, want_hess, tag)."""
    key = (mesh, n, threads, want_hess, tag)
    if key in _DRIVER_CACHE:
        return _DRIVER_CACHE[key]
    out = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = "1"  # don't let BLAS oversubscribe the box
    env["MKL_NUM_THREADS"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--dump", mesh, str(n),
           "1" if want_hess else "0", out]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(
            f"driver failed (mesh={mesh} n={n} threads={threads}):\n{res.stderr[-3000:]}")
    secs = None
    for line in res.stdout.splitlines():
        if line.startswith("COMPUTE_SECONDS="):
            secs = float(line.split("=", 1)[1])
    data = np.load(out)
    grad = data["grad"]
    hess = data["hess"] if data["hess"].size else None
    os.unlink(out)
    result = (grad, hess, secs)
    _DRIVER_CACHE[key] = result
    return result


class ThreadCountInvarianceTest(unittest.TestCase):
    def test_gradient_invariant_serial_vs_parallel(self):
        g1, _, _ = _run_driver("complex", 200, threads=1, want_hess=False)
        g8, _, _ = _run_driver("complex", 200, threads=8, want_hess=False)
        worst = float(np.max(np.abs(g1 - g8)))
        self.assertTrue(np.allclose(g1, g8, rtol=_RTOL, atol=_ATOL),
                        f"gradient changed with thread count: max|d|={worst:.3e}")

    def test_hessian_invariant_serial_vs_parallel(self):
        _, h1, _ = _run_driver("complex", 200, threads=1, want_hess=True)
        _, h8, _ = _run_driver("complex", 200, threads=8, want_hess=True)
        worst = float(np.max(np.abs(h1 - h8)))
        self.assertTrue(np.allclose(h1, h8, rtol=_RTOL, atol=_ATOL),
                        f"Hessian changed with thread count: max|d|={worst:.3e}")

    def test_gradient_deterministic_across_processes(self):
        # Two independent 8-thread processes must agree bit-for-bit. A
        # nondeterministic (e.g. critical-section) reduction would fail this.
        a, _, _ = _run_driver("complex", 200, threads=8, want_hess=False, tag=0)
        b, _, _ = _run_driver("complex", 200, threads=8, want_hess=False, tag=1)
        self.assertTrue(np.array_equal(a, b),
                        "8-thread gradient not bit-identical across processes")

    def test_gradient_robust_under_oversubscription(self):
        # More requested threads than cores (and more than some chunks have
        # hinges): every thread that gets zero iterations contributes a zero
        # partial. Result must still match the serial reduction and be
        # deterministic across processes.
        ref, _, _ = _run_driver("cdt", 60, threads=1, want_hess=False)
        a, _, _ = _run_driver("cdt", 60, threads=128, want_hess=False, tag=0)
        b, _, _ = _run_driver("cdt", 60, threads=128, want_hess=False, tag=1)
        self.assertTrue(np.allclose(ref, a, rtol=_RTOL, atol=_ATOL),
                        "oversubscribed gradient disagrees with serial")
        self.assertTrue(np.array_equal(a, b),
                        "oversubscribed gradient not deterministic across processes")


# --------------------------------------------------------------------------- #
# Edge cases — the boundary of the parallel region
# --------------------------------------------------------------------------- #
class EdgeCaseTest(unittest.TestCase):
    def test_empty_mesh_returns_empty(self):
        # No edges, no hinges: every thread's loop is zero-trip. Must not crash;
        # gradient is empty, Hessian is 0x0.
        st = _new_spacetime()
        rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
        self.assertEqual(len(rs.action_gradient_exact()), 0)
        self.assertEqual(len(rs.action_hessian_exact()), 0)


# --------------------------------------------------------------------------- #
# Scaling benchmark — reported, with a loose floor so it can't flake the suite.
# --------------------------------------------------------------------------- #
@unittest.skipUnless((os.cpu_count() or 1) >= 4, "needs >= 4 cores to show scaling")
class ScalingBenchmarkTest(unittest.TestCase):
    def test_gradient_speedup_with_threads(self):
        n = 4000  # big enough that per-hinge work dominates fork/join overhead
        ladder = [1, 2, 4]
        times = {}
        for t in ladder:
            _, _, secs = _run_driver("cdt", n, threads=t, want_hess=False, tag=10 + t)
            times[t] = secs
        base = times[1]
        ratios = {t: base / times[t] for t in ladder}
        print(f"\n[scaling] actionGradientExact, CDT n={n}:")
        for t in ladder:
            print(f"[scaling]   {t:>2d} threads: {times[t]:.3f}s  speedup x{ratios[t]:.2f}")
        # Loose floor: parallelism must clearly engage. Near-linear would be ~4x
        # at 4 threads; 1.5x is a robust floor against shared-box / memory-BW noise.
        self.assertGreaterEqual(
            ratios[4], 1.5,
            f"expected gradient to speed up with threads; got x{ratios[4]:.2f} at 4 "
            f"threads ({times})")


# --------------------------------------------------------------------------- #
# Subprocess driver entry point
# --------------------------------------------------------------------------- #
def _dump_main(argv):
    import time
    mesh, n, want_hess, out = argv[2], int(argv[3]), argv[4] == "1", argv[5]
    st = _complex_mesh() if mesh == "complex" else _make_cdt(n)
    rs = tessera.ReggeSolver(st, tessera.MatterConfiguration())
    t0 = time.perf_counter()
    grad = np.array([complex(z) for z in rs.action_gradient_exact()], dtype=complex)
    secs = time.perf_counter() - t0
    if want_hess:
        hess = np.array([[complex(z) for z in row]
                         for row in rs.action_hessian_exact()], dtype=complex)
    else:
        hess = np.array([], dtype=complex)
    np.savez(out, grad=grad, hess=hess)
    print(f"COMPUTE_SECONDS={secs}")


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--dump":
        _dump_main(sys.argv)
    else:
        unittest.main()
