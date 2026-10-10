# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The entanglement-complex drivers (`tessera.drivers.entanglement_complex`,
`entanglement_regions`, `entanglement_schedules`).

Covers the instrument:

* the gates are unitary, sqrt(SWAP) squares to SWAP, and the round-robin
  schedule meets every pair exactly once;
* the qubit network preserves trace and spectrum, and its reduced states,
  entropies and mutual information agree with an independent numpy
  computation;
* the information identities the analysis leans on hold on the network's
  entropies (the decomposition of I(a:b) through a third qubit, strong
  subadditivity, the variation-of-information triangle slack);
* every edge-length mode builds the documented lengths; the shortest-path
  option is a true metric that joins pairs below the floor through paths
  and changes nothing that is already a metric; the triangle report counts
  what it claims to;
* the Vietoris-Rips filtration gives the known homology of hand-built
  configurations (a square's loop; separate components);
* the time slices keep the entropy budget: C changes by exactly the change
  of the interacting pair's I, bystanders keep I(k:rest), and the temporal
  residuals and event time are as documented;
* the region analysis counts disjoint pairs and triples correctly, gives the
  known ledger and monogamy values of a product of pairs, the classical GHZ
  state and the XOR state, writes every block mutual information as the
  signed sum of the co-informations crossing its cut, and ignores per-qubit
  entropy shifts;
* the command lines run end to end, write what they promise, and refuse
  unsupported combinations.

Skips cleanly when tessera was built without the quantum subsystem.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import os
import random
import tempfile
import unittest
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

try:
    from tessera.quantum import MutualInformation  # noqa: F401
    from tessera.drivers import entanglement_complex as ec
    from tessera.drivers import entanglement_regions as er
    from tessera.drivers import entanglement_schedules as es
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False

LN2 = math.log(2.0)


def quiet(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def ginibre(rng, n, rank=None):
    d = 2 ** n
    g = rng.normal(size=(d, rank or d)) + 1j * rng.normal(size=(d, rank or d))
    rho = g @ g.conj().T
    return rho / np.trace(rho).real


def network(rho):
    n = int(round(math.log2(len(rho))))
    net = ec.QubitNetwork(n)
    net.rho = np.asarray(rho, dtype=complex).reshape((2,) * (2 * n))
    return net


def entropies(rho):
    net = network(rho)
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return er.region_entropies(net, float(-(w * np.log(w)).sum()))


def bits(*qubits):
    return sum(1 << q for q in qubits)


def I(S, A, B):
    return S[A] + S[B] - S[A | B]


def cmi(S, a, b, c):
    return S[a | c] + S[b | c] - S[c] - S[a | b | c]


def classical(probs):
    return np.diag(np.asarray(probs, dtype=complex))


def ghz_classical():
    p = np.zeros(8)
    p[0b000] = p[0b111] = 0.5
    return classical(p)


def xor_state():
    p = np.zeros(8)
    for a, b in itertools.product((0, 1), repeat=2):
        p[4 * a + 2 * b + (a ^ b)] = 0.25
    return classical(p)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestGatesAndSchedule(unittest.TestCase):

    def test_sqrt_swap_squares_to_swap_and_alpha_one_is_swap(self):
        U = ec.swap_power(0.5)
        np.testing.assert_allclose(U @ U, ec.SWAP, atol=1e-15)
        np.testing.assert_allclose(ec.swap_power(1), ec.SWAP, atol=1e-15)

    def test_every_gate_is_unitary(self):
        for name, U in ec.EXPLICIT_UNITARIES + (("SWAP^0.3", ec.swap_power(0.3)),):
            ec._require_unitary(name, U)                    # raises if not
        with self.assertRaises(ValueError):
            ec._require_unitary("not", np.diag([1, 1, 1, 2]))

    def test_explicit_states_are_density_matrices(self):
        for k, rho in enumerate(ec.EXPLICIT_STATES):
            ec._require_density_matrix("explicit %d" % k, rho)
        for bad in (np.eye(4) / 2, np.diag([0.5, 0.5, 0.5, -0.5]),
                    np.eye(4) / 4 + np.triu(np.full((4, 4), 0.1), 1)):
            with self.assertRaises(ValueError):
                ec._require_density_matrix("bad", bad)

    def test_round_robin_meets_every_pair_once(self):
        for n in (2, 3, 4, 5, 6, 7, 8, 10):
            rounds = ec.round_robin(n)
            self.assertEqual(sorted(p for r in rounds for p in r),
                             list(itertools.combinations(range(n), 2)))
            for r in rounds:
                used = [q for p in r for q in p]
                self.assertEqual(len(used), len(set(used)))
                self.assertEqual(len(used), n - n % 2)
            self.assertEqual(rounds[0], [(2 * k, 2 * k + 1) for k in range(n // 2)])

    def test_candidate_pairs(self):
        self.assertEqual(ec.candidate_pairs(4, "chain"), [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(len(ec.candidate_pairs(5, "all")), 10)
        with self.assertRaises(ValueError):
            ec.candidate_pairs(4, "ring")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestQubitNetwork(unittest.TestCase):

    def test_reduced_states_of_a_product(self):
        rng = np.random.default_rng(1)
        a, b = ginibre(rng, 2), ginibre(rng, 1)
        net = ec.QubitNetwork(3)
        net.set_product([((0, 1), a), ((2,), b)])
        np.testing.assert_allclose(net.reduced((0, 1)), a, atol=1e-14)
        np.testing.assert_allclose(net.reduced((2,)), b, atol=1e-14)
        self.assertAlmostEqual(net.mutual_information((0, 2)), 0.0, places=12)
        with self.assertRaises(ValueError):
            net.set_product([((1,), b), ((0, 2), a)])

    def test_gates_preserve_trace_and_spectrum(self):
        rho = ginibre(np.random.default_rng(2), 4)
        net = network(rho)
        before = np.sort(np.linalg.eigvalsh(rho))
        rng = np.random.default_rng(3)
        for pair in ((0, 1), (2, 3), (0, 3), (1, 2)):
            q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
            net.apply_gate(q, pair)
        self.assertAlmostEqual(net.trace(), 1.0, places=12)
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(net.matrix())), before, atol=1e-12)

    def test_gate_on_a_reversed_pair(self):
        a = ginibre(np.random.default_rng(4), 2)
        net = network(a)
        net.apply_gate(ec.CNOT, (1, 0))
        P = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])   # control 1
        np.testing.assert_allclose(net.matrix(), P @ a @ P.T, atol=1e-14)

    def test_entropy_and_mutual_information_in_nats(self):
        psi = np.zeros(4)
        psi[0] = psi[3] = 1 / math.sqrt(2)
        net = network(np.outer(psi, psi))
        self.assertAlmostEqual(net.entropy((0,)), LN2, places=12)
        self.assertAlmostEqual(net.mutual_information((0, 1)), 2 * LN2, places=12)

    def test_replace_by_marginals_keeps_the_marginals(self):
        net = network(ginibre(np.random.default_rng(5), 3))
        marg = [net.reduced((q,)) for q in range(3)]
        net.replace_by_marginals()
        for q in range(3):
            np.testing.assert_allclose(net.reduced((q,)), marg[q], atol=1e-14)
        self.assertAlmostEqual(net.mutual_information((0, 1)), 0.0, places=12)

    def test_refuses_a_bad_qubit_count(self):
        for n in (1, ec.MAX_QUBITS + 1):
            with self.assertRaises(ValueError):
                ec.QubitNetwork(n)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestInformationIdentities(unittest.TestCase):
    a, b, c, d = bits(0), bits(1), bits(2), bits(3)

    def states(self, n=3):
        rng = np.random.default_rng(11)
        for rank in (None, 1, 2):
            for _ in range(3):
                yield ginibre(rng, n, rank)

    def test_mi_through_a_third_party(self):
        a, b, c = self.a, self.b, self.c
        for rho in self.states():
            S = entropies(rho)
            rhs = I(S, a, c) + I(S, b, c) - I(S, c, a | b) + cmi(S, a, b, c)
            self.assertAlmostEqual(I(S, a, b), rhs, places=12)

    def test_strong_subadditivity(self):
        for rho in self.states(4):
            S = entropies(rho)
            for x, y, z in itertools.permutations((self.a, self.b, self.c, self.d), 3):
                self.assertGreaterEqual(cmi(S, x, y, z), -1e-12)

    def test_variation_of_information_slack(self):
        a, b, c = self.a, self.b, self.c

        def vi(S, x, y):
            return 2 * S[x | y] - S[x] - S[y]

        for rho in self.states():
            S = entropies(rho)
            slack = vi(S, a, c) + vi(S, c, b) - vi(S, a, b)
            self.assertAlmostEqual(slack, 2 * (cmi(S, a, b, c) + S[a | b | c] - S[a | b]),
                                   places=12)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestLengths(unittest.TestCase):
    MI = np.array([[0, 0.5, 0.1], [0.5, 0, 0.2], [0.1, 0.2, 0]])
    S = np.array([0.69, 0.6, 0.55])

    def test_each_mode_uses_its_formula(self):
        formulas = {"inverse": lambda i, j: 1 / self.MI[i, j],
                    "log": lambda i, j: -math.log(self.MI[i, j] / ec.I_MAX),
                    "mi": lambda i, j: self.MI[i, j],
                    "vi": lambda i, j: self.S[i] + self.S[j] - 2 * self.MI[i, j]}
        for mode, f in formulas.items():
            D, W = ec.target_lengths(self.MI, mode, 1e-12, self.S)
            self.assertAlmostEqual(D[0, 1] / D[0, 2], f(0, 1) / f(0, 2), places=12, msg=mode)
            self.assertAlmostEqual(D[1, 2] / D[0, 2], f(1, 2) / f(0, 2), places=12, msg=mode)
            self.assertAlmostEqual(D[W > 0].mean(), 1.0, places=12)

    def test_log1p_uses_its_formula_and_can_stay_unnormalised(self):
        a, I0 = 2.5, 0.7
        D, W = ec.target_lengths(self.MI, "log1p", 1e-12, scale=a, reference=I0,
                                 normalise=False)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(D[i, j], a * math.log(1 + I0 / self.MI[i, j]), places=12)
        self.assertEqual(W.sum(), 6)
        D1, _ = ec.target_lengths(self.MI, "log1p", 1e-12, scale=a, reference=I0)
        self.assertAlmostEqual(D1[W > 0].mean(), 1.0, places=12)
        np.testing.assert_allclose(D1[W > 0], D[W > 0] / D[W > 0].mean())
        D2, _ = ec.target_lengths(self.MI, "log1p", 1e-12, reference=I0)
        np.testing.assert_allclose(D2, D1)           # the scale cancels when normalised
        D3, _ = ec.target_lengths(self.MI, "log1p", 1e-12, normalise=False)
        self.assertAlmostEqual(D3[0, 1], math.log(1 + ec.I_MAX / self.MI[0, 1]), places=12)
        self.assertEqual(ec.length_label("log1p"), "a ln(1 + I_0/I(X:Y))")

    def test_floor_and_errors(self):
        _, W = ec.target_lengths(self.MI, "inverse", 0.15)
        self.assertEqual((W[0, 2], W[0, 1]), (0, 1))
        D, W = ec.target_lengths(self.MI, "inverse", 1.0)
        self.assertIsNone(D)
        with self.assertRaises(ValueError):
            ec.target_lengths(self.MI, "vi", 1e-12)
        with self.assertRaises(ValueError):
            ec.target_lengths(self.MI, "cubic", 1e-12)

    def test_negative_vi_is_clipped_with_a_warning(self):
        MI = np.array([[0, 2 * LN2, 0.1], [2 * LN2, 0, 0.1], [0.1, 0.1, 0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            D, _ = ec.target_lengths(MI, "vi", 1e-12, [LN2, LN2, LN2])
        self.assertEqual(D[0, 1], 0.0)
        self.assertTrue(any("clipped to 0" in str(w.message) for w in caught))

    def test_geodesic_matches_brute_force_paths(self):
        rng = np.random.default_rng(6)
        n = 5
        MI = np.triu(rng.uniform(0.001, 1, (n, n)) * (rng.uniform(size=(n, n)) < 0.7), 1)
        MI = MI + MI.T
        D0, W0 = ec.target_lengths(MI, "inverse", 1e-12)
        raw = np.where(W0 > 0, 1 / np.where(MI > 0, MI, 1), np.inf)
        D, W = ec.target_lengths(MI, "inverse", 1e-12, geodesic=True)
        scale = None
        for a, b in itertools.combinations(range(n), 2):
            best = np.inf
            others = [v for v in range(n) if v not in (a, b)]
            for k in range(len(others) + 1):
                for mid in itertools.permutations(others, k):
                    path = (a,) + mid + (b,)
                    best = min(best, sum(raw[u, v] for u, v in zip(path, path[1:])))
            self.assertEqual(W[a, b] > 0, np.isfinite(best))
            if np.isfinite(best):
                scale = scale or best / D[a, b]
                self.assertAlmostEqual(D[a, b] * scale, best, places=9)

    def test_geodesic_joins_through_paths_and_keeps_components_apart(self):
        MI = np.zeros((5, 5))
        for (i, j), v in {(0, 1): 0.5, (1, 2): 0.4, (3, 4): 0.3}.items():
            MI[i, j] = MI[j, i] = v
        D, W = ec.target_lengths(MI, "inverse", 1e-12, geodesic=True)
        self.assertEqual(W[0, 2], 1)
        self.assertAlmostEqual(D[0, 2], D[0, 1] + D[1, 2], places=12)
        self.assertTrue(np.all(W[np.ix_([0, 1, 2], [3, 4])] == 0))

    def test_geodesic_is_a_metric_and_changes_a_metric_not_at_all(self):
        rng = np.random.default_rng(7)
        MI = rng.uniform(0.001, 1.0, (7, 7))
        MI = (MI + MI.T) / 2
        np.fill_diagonal(MI, 0)
        names = list("ABCDEFG")
        D, W = ec.target_lengths(MI, "inverse", 1e-12)
        self.assertGreater(ec.triangle_inequality_report(D, W, names)[1], 0)
        for mode in ec.LENGTH_MODES:
            with warnings.catch_warnings(record=True):        # vi clips negative lengths here
                warnings.simplefilter("always")
                D, W = ec.target_lengths(MI, mode, 1e-12, np.full(7, LN2), geodesic=True)
            self.assertEqual(ec.triangle_inequality_report(D, W, names)[1], 0, mode)
        flat = np.full((5, 5), 0.3)
        np.fill_diagonal(flat, 0)
        D0, W0 = ec.target_lengths(flat, "log", 1e-12)
        D1, W1 = ec.target_lengths(flat, "log", 1e-12, geodesic=True)
        np.testing.assert_allclose(D1, D0, atol=1e-12)

    def test_triangle_report(self):
        D = np.array([[0, 1, 1], [1, 0, 3], [1, 3, 0]], float)
        checks, violations, bad, worst = ec.triangle_inequality_report(D, (D > 0) * 1.0,
                                                                       list("ABC"))
        self.assertEqual((checks, violations, bad), (3, 1, 1))
        self.assertEqual(worst[0][1], "BC")
        zero = np.zeros((3, 3))                       # a zero edge and zero detour
        self.assertEqual(ec.triangle_inequality_report(zero, 1.0 - np.eye(3), list("ABC"))[1], 0)
        flat = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])   # edge 1, detour 0
        report = ec.triangle_inequality_report(flat, 1.0 - np.eye(3), list("ABC"))
        self.assertEqual(report[1], 1)
        self.assertEqual(report[3][0][0], np.inf)
        D = np.array([[0, 1.1, 0.7], [1.1, 0, 0.4], [0.7, 0.4, 0]]) / 3.7
        D[0, 1] = D[1, 0] = D[0, 2] + D[2, 1]
        D = D / D[D > 0].mean()
        self.assertEqual(ec.triangle_inequality_report(D, (D > 0) * 1.0, list("ABC"))[1], 0)

    def test_labels(self):
        self.assertEqual(ec.length_label("inverse"), "1/I(X:Y)")
        self.assertEqual(ec.length_label("vi", True), "shortest path over S(X|Y)+S(Y|X)")


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRipsFiltration(unittest.TestCase):

    def test_a_square_has_a_loop_until_its_diagonals_appear(self):
        # 0-1-2-3 a unit square: sides 1, diagonals sqrt(2)
        D = np.array([[0, 1, 2 ** 0.5, 1], [1, 0, 1, 2 ** 0.5],
                      [2 ** 0.5, 1, 0, 1], [1, 2 ** 0.5, 1, 0]])
        f = ec.rips_filtration(D, (D > 0) * 1.0)
        levels = [(lv["scale"], lv["f_vector"], lv["betti"]) for lv in f["levels"]]
        self.assertEqual(levels[0], (0.0, [4], [4]))
        self.assertEqual(levels[1][1:], ([4, 4], [1, 1]))
        self.assertAlmostEqual(levels[2][0], 2 ** 0.5)
        self.assertEqual(levels[2][1], [4, 6, 4, 1])
        self.assertEqual(levels[2][2][:2], [1, 0])
        self.assertTrue(all(b == 0 for b in levels[2][2][1:]))

    def test_separate_components_and_births(self):
        D = np.zeros((4, 4))
        D[0, 1] = D[1, 0] = 0.7
        D[2, 3] = D[3, 2] = 1.3
        f = ec.rips_filtration(D, (D > 0) * 1.0)
        self.assertEqual(f["levels"][-1]["betti"][0], 2)
        births = dict(zip(f["cliques"], f["births"]))
        self.assertEqual(births[(0, 1)], 0.7)
        self.assertEqual(births[(2, 3)], 1.3)
        self.assertEqual(births[(0,)], 0.0)

    def test_complete_graph_gives_every_subset(self):
        rng = np.random.default_rng(8)
        D = rng.uniform(0.5, 2, (5, 5))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        f = ec.rips_filtration(D, 1.0 - np.eye(5))
        self.assertEqual(len(f["cliques"]), 2 ** 5 - 1)
        for c, b in zip(f["cliques"], f["births"]):
            self.assertEqual(b, max((D[x, y] for x, y in itertools.combinations(c, 2)),
                                    default=0.0))
        self.assertEqual(f["levels"][-1]["f_vector"], [5, 10, 10, 5, 1])


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestTimeSlices(unittest.TestCase):

    def run_slices(self, n=5, timesteps=8, pair_mode="all", regions=False):
        return ec.simulate(n, 0, "1/2", 0, "global", timesteps, pair_mode, regions)

    def test_total_correlation_changes_by_the_pair_mi_change(self):
        sl = self.run_slices()["slices"]
        for prev, cur in zip(sl, sl[1:]):
            self.assertAlmostEqual(cur["C"] - prev["C"], cur["dI"], places=10)
            self.assertLess(cur["identity_err"], ec.STEP_IDENTITY_TOL)

    def test_residuals_and_event_time(self):
        sl = self.run_slices()["slices"]
        self.assertEqual(sl[0]["tau"], 0.0)
        for prev, cur in zip(sl, sl[1:]):
            i, j = cur["pair"]
            I_new = cur["I_dict"][cur["pair"]]
            for q in range(5):
                expect = prev["S_dict"][q] - (I_new if q in (i, j) else 0.0)
                self.assertAlmostEqual(cur["residual"][q], expect, places=13)
            self.assertAlmostEqual(cur["dtau"], (cur["residual"][i] + cur["residual"][j]) / 2,
                                   places=13)
            self.assertAlmostEqual(cur["tau"], prev["tau"] + cur["dtau"], places=13)

    def test_chain_events_are_nearest_neighbours(self):
        for s in self.run_slices(n=6, timesteps=10, pair_mode="chain")["slices"][1:]:
            self.assertEqual(s["pair"][1] - s["pair"][0], 1)

    def test_bystanders_keep_their_whole_system_mi(self):
        net = network(ginibre(np.random.default_rng(9), 4))
        w = np.linalg.eigvalsh(net.matrix())
        Sg = float(-(w * np.log(w)).sum())

        def vertex_mi(S):
            return np.array([S[1 << k] + S[0b1111 ^ (1 << k)] - Sg for k in range(4)])

        before = vertex_mi(er.region_entropies(net, Sg))
        q, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(4, 4)) + 0j)
        net.apply_gate(q, (0, 2))
        after = vertex_mi(er.region_entropies(net, Sg))
        np.testing.assert_allclose(after[[1, 3]], before[[1, 3]], atol=1e-12)

    def test_slice_ledger_matches_the_final_state(self):
        res = self.run_slices(n=4, timesteps=5, regions=True)
        rep = er.analyse(res["net"], res["S_global"])
        last = res["slices"][-1]["regions"]
        np.testing.assert_allclose(last["shares"], rep["shares"], atol=1e-12)
        self.assertAlmostEqual(last["C"], res["slices"][-1]["C"], places=10)

    def test_marginals_carry_refused_with_slices(self):
        with self.assertRaises(ValueError):
            ec.simulate(4, 0, "1/2", 0, "marginals", 3)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRegions(unittest.TestCase):

    def test_counts_of_disjoint_pairs_and_triples(self):
        for n in range(3, 8):
            A, B = er.disjoint_pairs(n)
            self.assertEqual(len(A), (3 ** n - 2 ** (n + 1) + 1) // 2)
            self.assertTrue(np.all(A & B == 0))
            self.assertEqual(er.mmi_report(np.zeros(2 ** n), n)["triples"],
                             (4 ** n - 3 * 3 ** n + 3 * 2 ** n - 1) // 6)

    def test_product_of_pairs_is_all_pairwise(self):
        rng = np.random.default_rng(12)
        S = entropies(np.kron(ginibre(rng, 2), ginibre(rng, 2)))
        shares, C = er.ledger(S, 4)
        self.assertAlmostEqual(shares[0], C, places=12)
        np.testing.assert_allclose(shares[1:], 0, atol=1e-12)
        self.assertEqual(er.mmi_report(S, 4)["violations"], 0)

    def test_classical_ghz(self):
        S = entropies(ghz_classical())
        shares, C = er.ledger(S, 3)
        np.testing.assert_allclose(shares, [3 * LN2, -LN2], atol=1e-12)
        self.assertAlmostEqual(C, 2 * LN2, places=12)
        r = er.mmi_report(S, 3)
        self.assertEqual(r["violations"], 1)
        self.assertAlmostEqual(r["worst"], LN2, places=12)

    def test_xor_is_purely_three_body_and_monogamous(self):
        S = entropies(xor_state())
        shares, C = er.ledger(S, 3)
        np.testing.assert_allclose(shares, [0.0, LN2], atol=1e-12)
        r = er.mmi_report(S, 3)
        self.assertEqual(r["violations"], 0)
        self.assertAlmostEqual(r["worst"], -LN2, places=12)

    def test_block_mi_is_the_signed_sum_of_crossing_coinformations(self):
        n = 4
        S = entropies(ginibre(np.random.default_rng(13), n))
        I_T = er.coinformation(S, n)
        pc = er.popcounts(n)
        for a, b in zip(*er.disjoint_pairs(n)):
            u = a | b
            total = sum((-1) ** pc[T] * I_T[T] for T in range(1, 2 ** n)
                        if T & ~u == 0 and T & a and T & b)
            self.assertAlmostEqual(I(S, a, b), total, places=12)

    def test_local_shifts_change_nothing(self):
        S0 = entropies(ginibre(np.random.default_rng(14), 4))
        c = np.array([0.3, 0.1, 0.7, 0.2])
        S1 = S0 + np.array([c[list(er.members(A, 4))].sum() for A in range(16)])
        A, B = er.disjoint_pairs(4)
        np.testing.assert_allclose(er.block_mi(S0, A, B), er.block_mi(S1, A, B), atol=1e-12)
        np.testing.assert_allclose(er.ledger(S0, 4)[0], er.ledger(S1, 4)[0], atol=1e-12)
        np.testing.assert_allclose(er.mmi_report(S0, 4)["I3"], er.mmi_report(S1, 4)["I3"],
                                   atol=1e-12)

    def test_entropies_from_coinformations(self):
        S = entropies(ginibre(np.random.default_rng(15), 4))
        I_T = er.coinformation(S, 4)
        pc = er.popcounts(4)
        for A in range(1, 16):
            total = sum((-1) ** (pc[T] + 1) * I_T[T] for T in range(1, 16) if T & ~A == 0)
            self.assertAlmostEqual(total, S[A], places=12)

    def test_too_many_qubits_refused(self):
        net = ec.QubitNetwork(er.MAX_QUBITS + 1)
        with self.assertRaises(ValueError):
            er.analyse(net, 0.0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestCommandLine(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_text_report(self):
        code, out = quiet(ec.main, ["--qubits", "4", "--no-show"])
        self.assertEqual(code, 0)
        for text in ("INPUT STATES", "FINAL STATE", "VIETORIS-RIPS FILTRATION",
                     "triangle inequality"):
            self.assertIn(text, out)

    def test_json_record(self):
        code, out = quiet(ec.main, ["--qubits", "5", "--regions", "--json"])
        record = json.loads(out)
        self.assertEqual(record["entropy_unit"], "nats")
        self.assertEqual(len(record["MI"]), 5)
        levels = record["filtration"]
        self.assertEqual(levels[0]["f_vector"], [5])
        self.assertEqual(levels[-1]["betti"][0], 1)
        self.assertIn("regions", record)

    def test_geodesic_has_no_violations(self):
        _, out = quiet(ec.main, ["--json", "--geodesic"])
        self.assertEqual(json.loads(out)["triangle_inequality"]["violations"], 0)

    def test_slices_regions_and_figures(self):
        _, out = quiet(ec.main, ["--qubits", "4", "--timesteps", "3", "--regions",
                                 "--no-show", "--save", self.path("t.png")])
        self.assertEqual(out.count("C by order"), 4)
        for name in ("t.png", "t_regions.png"):
            self.assertTrue(os.path.exists(self.path(name)), name)

    def test_refusals(self):
        for argv in (["--timesteps", "2", "--carry", "marginals", "--no-show"],
                     ["--qubits", str(er.MAX_QUBITS + 1), "--regions", "--no-show"],
                     ["--swap-power", "half", "--no-show"]):
            with self.assertRaises(SystemExit, msg=argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                quiet(ec.main, argv)

    def test_schedule_comparison(self):
        code, out = quiet(es.main, ["--qubits", "4", "--timesteps", "6", "--seeds", "0",
                                    "--json"])
        payload = json.loads(out)
        self.assertEqual(len(payload["all"]["sumI"][0]), 7)
        self.assertEqual(payload["entropy_unit"], "nats")

    def test_relaxation_fit(self):
        t = np.arange(40)
        floor, A, tr = es.fit_relaxation(0.1 + np.exp(-t / 5.0), 4)
        self.assertGreater(tr, 0)
        self.assertTrue(np.isfinite(tr))
        self.assertTrue(np.isinf(es.fit_relaxation(np.linspace(1, 2, 20), 2)[2]))


if __name__ == "__main__":
    unittest.main()
