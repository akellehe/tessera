# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The exchange statistic from occupation parity, and the mode order from
oriented component lineage (issue #1202): the occupancy channel of
:class:`tessera.observables.ExchangeHolonomy`, the compilation-order key of
:class:`tessera.observables.ClusterLineage`, and the lineage assignment of
:class:`tessera.quantum.EdgeModeRegistry`.

The whitepaper is the specification these tests hold the code to.

* The graded interchange law is ``tau(a (x) b) = (-1)^{F_a F_b} b (x) a``, so
  cluster parity is OCCUPATION parity: the sign of an exchange follows the
  number of occupied one-particle modes each cluster carries.
* The odd-rank determinant ``det pi_AB = (-1)^{r_A r_B}`` -- the determinant of
  exchanging two complete fibre frames -- is exact as an identity about frames,
  but promoting it to particle statistics is a falsifiable hypothesis the
  construction does not adopt. It is reported as an independent cross-check,
  and the two minus signs are never counted in the same exchange.
* On a sheeted support the cross-check is retired: the colour-spin fibre
  ``E-bar (x) C^3`` has even rank, so exchanging whole frames gives ``+1``,
  while a quark is a single occupied mode of that fibre and its exchange sign
  is the occupation parity ``-1``.
* Writing the exterior algebra as an ordered tensor product needs a chosen mode
  order, and a deterministic order is fixed by oriented component lineage, with
  the parity of every reordering applied.

Acceptance coverage (ticket #1202):

* the statistic of a two-cluster exchange follows the occupations when they
  differ from the ranks -- two rank-two blocks carrying one occupied mode each
  exchange with statistic ``-1`` while the rank-parity cross-check reads
  ``+1``, and the read says so rather than reporting one of them as the other;
* the mode order is invariant under a relabelling of cells: the modes of one
  cluster are compiled together and the clusters follow their lineage numbers,
  whatever order the cells were registered in.

Every quantity below is an exact integer, an exact string comparison, or a
certificate verdict: parities are exact integers given the verified matching
premise, and the compilation order is a sort on exact strings.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import tessera

obs = tessera.observables
quantum = tessera.quantum

EH = obs.ExchangeHolonomy
ClusterLineage = obs.ClusterLineage
EdgeModeRegistry = quantum.EdgeModeRegistry
LineageAssignment = quantum.LineageAssignment

OMEGA = 0.7          # a common metric phase, so nothing is accidentally real


# ─── the block-tracking fixture ────────────────────────────────────────────

def _mode(x, n):
    """A localized unit mode at ring position x: a cosine/sine interpolation
    between cells floor(x) and floor(x) + 1."""
    k = int(math.floor(x)) % n
    f = x - math.floor(x)
    v = np.zeros(n, complex)
    v[k] += math.cos(f * math.pi / 2.0)
    v[(k + 1) % n] += math.sin(f * math.pi / 2.0)
    return v


def _ring_cells(n):
    return [[c] for c in range(n)]


def _weights(n):
    return np.exp(1j * OMEGA) * np.ones(n, complex)


_UNMEASURED = float("nan")


def _fiber(frame, cells, weights, accepted=True):
    """A synthetic SpectralFiber over explicit cell tuples, built through the
    public record schema, which is the documented replay path."""
    frame = np.asarray(frame, complex)
    if frame.ndim == 1:
        frame = frame[:, None]
    rows, rank = frame.shape
    weights = np.asarray(weights, complex)
    gram = frame.conj().T @ np.diag(weights) @ frame
    left = np.linalg.solve(
        np.diag(weights).conj(),
        np.linalg.solve(gram, frame.conj().T @ np.diag(weights)).conj().T)

    def flat(matrix):
        return [complex(matrix[i, j])
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])]

    inner = {
        "grade": "certified-numerical" if accepted else "heuristic-discovery",
        "domain": "band-window",
        "regime": "non-normal",
        "residual": 0.0 if accepted else _UNMEASURED,
        "conditioning": 1.0,
        "dense_reference_error": _UNMEASURED,
        "tolerance": 1e-9,
    }
    certificate = {
        "degree": 0, "rank": rank,
        "lower_gap": 1.0, "upper_gap": 1.0,
        "localization": 1.0 / rows,
        "projector_residual": 0.0, "eigen_residual": 0.0,
        "left_residual": 0.0, "gram_defect": 0.0,
        "condition_number": 1.0,
        "positive_signature": rank, "negative_signature": 0,
        "frequency_lower": 0.0, "frequency_upper": 0.0,
        "self_adjoint": False, "accepted": bool(accepted),
        "certificate": inner,
    }
    record = {
        "schema_version": 1, "record_type": "spectral_fiber",
        "cells": [[int(v) for v in cell] for cell in cells],
        "rows": rows, "rank": rank,
        "eigenvalues_re": [0.0] * rank, "eigenvalues_im": [0.0] * rank,
        "right_frame_re": [float(z.real) for z in flat(frame)],
        "right_frame_im": [float(z.imag) for z in flat(frame)],
        "left_frame_re": [float(z.real) for z in flat(left)],
        "left_frame_im": [float(z.imag) for z in flat(left)],
        "weights_re": [float(w.real) for w in weights],
        "weights_im": [float(w.imag) for w in weights],
        "certificate": certificate,
    }
    return obs.SpectralFiber.fromRecord(record)


def _exchange_tracking(positions, n, steps, distance, ranks):
    """A T x B per-block tracking of the translation fixture: block b holds
    `ranks[b]` adjacent localized modes starting at positions[b], and the whole
    configuration advances `distance` cells over `steps` cyclic steps."""
    out = []
    for t in range(steps):
        x = distance * t / steps
        row = []
        for p, r in zip(positions, ranks):
            columns = [_mode((p + k + x) % n, n) for k in range(r)]
            row.append(_fiber(np.stack(columns, axis=1), _ring_cells(n),
                              _weights(n)))
        out.append(row)
    return out


def _occupancy(occupation, sheets=1):
    return obs.ClusterOccupancy(occupation=occupation, sheetCount=sheets)


# ─── the exchange statistic ────────────────────────────────────────────────

class TestOccupationParityIsTheStatistic(unittest.TestCase):
    """Cluster parity is occupation parity; fibre rank enters only the
    separately reported cross-check."""

    def test_the_statistic_follows_the_occupations_not_the_ranks(self) -> None:
        """The acceptance case: two rank-two blocks each carrying ONE occupied
        mode. The occupations give (-1)^{1*1} = -1 and the ranks give
        (-1)^{2*2} = +1, and the read reports both as what they are."""
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [2, 2])
        read = EH.blockPermutation(tracking,
                                   occupancies=[_occupancy(1), _occupancy(1)])
        self.assertEqual(list(read.blockPermutation), [1, 0])
        self.assertEqual(list(read.blockRanks), [2, 2])
        self.assertEqual(list(read.blockOccupations), [1, 1])
        self.assertEqual(read.occupationParity, -1)   # the statistic
        self.assertEqual(read.rankParity, +1)         # the cross-check
        self.assertFalse(read.rankParityAgrees)
        self.assertFalse(read.rankParityRetired)
        self.assertTrue(read.certificate.holds())

    def test_the_default_declaration_is_one_occupied_mode(self) -> None:
        """An empty occupancy list declares the default cluster of the
        construction -- one occupied mode on an unsheeted support -- for every
        block, which is what a quark is."""
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [2, 2])
        read = EH.blockPermutation(tracking)
        self.assertEqual(list(read.blockOccupations), [1, 1])
        self.assertEqual(list(read.blockSheetCounts), [1, 1])
        self.assertEqual(read.occupationParity, -1)

    def test_two_doubly_occupied_clusters_exchange_with_plus_one(self) -> None:
        """A meson or a diquark carries two odd constituents, so its composite
        parity is even and two of them exchange with +1."""
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [2, 2])
        read = EH.blockPermutation(tracking,
                                   occupancies=[_occupancy(2), _occupancy(2)])
        self.assertEqual(read.occupationParity, +1)
        self.assertEqual(read.rankParity, +1)
        self.assertTrue(read.rankParityAgrees)

    def test_an_odd_and_an_even_cluster_exchange_with_plus_one(self) -> None:
        """(-1)^{1*2} = +1: a quark and a meson pick up no sign from each
        other."""
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [2, 2])
        read = EH.blockPermutation(tracking,
                                   occupancies=[_occupancy(1), _occupancy(2)])
        self.assertEqual(read.occupationParity, +1)

    def test_two_singly_occupied_rank_one_clusters_agree_with_the_ranks(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [1, 1])
        read = EH.blockPermutation(tracking)
        self.assertEqual(read.occupationParity, -1)
        self.assertEqual(read.rankParity, -1)
        self.assertTrue(read.rankParityAgrees)

    def test_a_double_exchange_is_plus_one_whatever_the_occupations(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 32, 8, [2, 2])
        for occupation in (1, 2):
            read = EH.blockPermutation(
                tracking, occupancies=[_occupancy(occupation)] * 2)
            self.assertEqual(list(read.blockPermutation), [0, 1])
            self.assertEqual(read.occupationParity, +1)

    def test_the_cross_check_is_retired_on_a_sheeted_support(self) -> None:
        """A quark is a single occupied mode of a three-sheeted colour-spin
        fibre of even rank. Exchanging whole frames of even rank gives +1, so
        the cross-check says nothing about the statistic and is retired rather
        than reported as it."""
        tracking = _exchange_tracking([0, 8], 16, 32, 8, [6, 6])
        quark = _occupancy(1, sheets=3)
        read = EH.blockPermutation(tracking, occupancies=[quark, quark])
        self.assertEqual(list(read.blockPermutation), [1, 0])
        self.assertEqual(list(read.blockRanks), [6, 6])
        self.assertEqual(list(read.blockSheetCounts), [3, 3])
        self.assertTrue(read.rankParityRetired)
        self.assertEqual(read.rankParity, 0)
        self.assertFalse(read.rankParityAgrees)
        self.assertEqual(read.occupationParity, -1)
        self.assertTrue(read.certificate.holds())

    def test_an_uncertified_tracking_reports_no_parity_at_all(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [1, 1])
        tracking[8][0] = _fiber(_mode(2.0, 8), _ring_cells(8), _weights(8),
                                accepted=False)
        read = EH.blockPermutation(tracking)
        self.assertFalse(read.certificate.holds())
        self.assertEqual(read.occupationParity, 0)
        self.assertEqual(read.rankParity, 0)
        self.assertFalse(read.rankParityAgrees)
        self.assertEqual(list(read.blockPermutation), [])

    def test_a_wrong_number_of_occupancies_throws(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [1, 1])
        with self.assertRaises(ValueError):
            EH.blockPermutation(tracking, occupancies=[_occupancy(1)])

    def test_occupying_more_modes_than_the_fibre_has_throws(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [1, 1])
        with self.assertRaises(ValueError):
            EH.blockPermutation(
                tracking, occupancies=[_occupancy(2), _occupancy(1)])

    def test_a_support_with_no_sheets_throws(self) -> None:
        tracking = _exchange_tracking([0, 4], 8, 16, 4, [1, 1])
        with self.assertRaises(ValueError):
            EH.blockPermutation(
                tracking,
                occupancies=[_occupancy(1, sheets=0), _occupancy(1)])


class TestFrameExchangeDeterminant(unittest.TestCase):
    """det pi_AB = (-1)^{r_A r_B}, the determinant of exchanging two complete
    fibre frames, offered as the cross-check and never as the statistic."""

    def test_the_determinant_table(self) -> None:
        for rank_a, rank_b, expected in ((1, 1, -1), (3, 3, -1), (1, 2, +1),
                                         (2, 2, +1), (6, 6, +1), (3, 6, +1),
                                         (0, 5, +1)):
            self.assertEqual(
                EH.frameExchangeDeterminant(rank_a, rank_b), expected,
                f"det pi for ranks ({rank_a}, {rank_b})")

    def test_it_matches_an_explicit_permutation_determinant(self) -> None:
        """An independent reference: the determinant of the block interchange
        matrix itself, assembled with numpy."""
        for rank in (1, 2, 3, 4):
            size = 2 * rank
            permutation = np.zeros((size, size))
            for i in range(rank):
                permutation[rank + i, i] = 1.0
                permutation[i, rank + i] = 1.0
            self.assertEqual(EH.frameExchangeDeterminant(rank, rank),
                             round(float(np.linalg.det(permutation))))


# ─── the compilation-order key of a lineage ────────────────────────────────

TETRAHEDRON = [[0, 1, 2, 3]]


def _level(cells, vertices):
    return obs.LevelComplex(cells, vertices)


def _two_level_history():
    """Two copies of the tetrahedron joined by the identity reduction: level
    l's own vertex v is the cobordism's vertex 4 * l + v."""
    return ClusterLineage.history(
        [_level(TETRAHEDRON, 4), _level(TETRAHEDRON, 4)], [[0, 1, 2, 3]])


def _reading(number, fermion_number=1, cluster_id="Q"):
    """A certified LineageNumberRead of the declared lineage number.

    A fibre path out of an incoming vertex crosses the level cut once, so it
    reads ``N_Q = +1``; its reversal reads ``-1``; and a path that stays inside
    the incoming level never crosses the cut and reads ``0``. All three are
    relative cycles, so every reading below is certified.
    """
    if number not in (-1, 0, 1):
        raise AssertionError("the fixture builds only N_Q in {-1, 0, +1}")
    cobordism = _two_level_history()
    cut = ClusterLineage.levelCut(cobordism, 0)
    if number == 0:
        lineage = ClusterLineage.fromVertexPath(cobordism, [0, 1],
                                                fermion_number, cluster_id)
    else:
        lineage = ClusterLineage.fromFiberPath(cobordism, 0, fermion_number,
                                               cluster_id)
        if number == -1:
            lineage = ClusterLineage.reversed(lineage)
    reading = ClusterLineage.read(cobordism, cut, lineage)
    assert reading.number == number, reading.number
    return reading


class TestLineageOrderKey(unittest.TestCase):
    """The primary key of the compilation order is the oriented lineage."""

    def test_the_key_orders_lineage_numbers_numerically(self) -> None:
        """An anti-cluster sorts before a cluster: the sign of N_Q is carried
        by the encoding and not by a leading minus sign, which would sort a
        negative number after every positive one."""
        keys = [ClusterLineage.orderKey(_reading(n)) for n in (-1, 0, 1)]
        self.assertEqual(keys, sorted(keys))

    def test_the_integers_are_written_at_a_fixed_width(self) -> None:
        """Lexicographic order on the keys is numeric order on the integers
        only because the integers are written at one width. A key for a
        negative, a zero and a positive lineage number therefore all have the
        same length, and the numeric field is exactly ten digits."""
        keys = [ClusterLineage.orderKey(_reading(n)) for n in (-1, 0, 1)]
        self.assertEqual({len(k) for k in keys}, {len(keys[0])})
        for key in keys:
            self.assertTrue(key.startswith("lineage:"))
            digits = key[len("lineage:"):].split(":", 1)[0]
            self.assertEqual(len(digits), 10)
            self.assertTrue(digits.isdigit())

    def test_the_key_orders_fermion_numbers_numerically(self) -> None:
        """The width trap: a plain decimal would sort "10" before "2"."""
        keys = [ClusterLineage.orderKey(_reading(1, n, "a"))
                for n in (1, 2, 10)]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 3)

    def test_the_cluster_name_breaks_the_remaining_tie(self) -> None:
        first = ClusterLineage.orderKey(_reading(1, 1, "a"))
        second = ClusterLineage.orderKey(_reading(1, 1, "b"))
        self.assertLess(first, second)

    def test_the_key_is_the_same_for_two_readings_of_one_lineage(self) -> None:
        self.assertEqual(ClusterLineage.orderKey(_reading(1, 1, "q")),
                         ClusterLineage.orderKey(_reading(1, 1, "q")))

    def test_an_uncertified_reading_has_no_order_key(self) -> None:
        """A cut that does not separate leaves N_Q dependent on the cut, so no
        compilation order can be fixed by it."""
        cobordism = _two_level_history()
        broken = ClusterLineage.cutFromSides(cobordism, [0] * 8)
        lineage = ClusterLineage.fromFiberPath(cobordism, 0, 1, "q")
        reading = ClusterLineage.read(cobordism, broken, lineage)
        self.assertFalse(reading.cutSeparates)
        with self.assertRaises(ValueError):
            ClusterLineage.orderKey(reading)


# ─── the mode order from oriented component lineage ────────────────────────

def _registry(edges, key="~unassigned"):
    """A registry built by registering `edges` in the given order."""
    registry = EdgeModeRegistry()
    for a, b in edges:
        registry.addEdge(a, b, +1, key)
    return registry


def _ordered_pairs(registry):
    """The canonical mode order as a list of unordered vertex pairs."""
    return [(min(registry.record(m).vertexA, registry.record(m).vertexB),
             max(registry.record(m).vertexA, registry.record(m).vertexB))
            for m in registry.canonicalModeOrder()]


def _ordered_keys(registry):
    return [registry.record(m).lineageKey
            for m in registry.canonicalModeOrder()]


class TestModeOrderFromLineage(unittest.TestCase):
    """The compilation order's primary key is the oriented component lineage,
    so the modes of one cluster are compiled together and the clusters follow
    relabelling-invariant integers."""

    # Two clusters on disjoint vertex sets. The second carries the SMALLER
    # vertex ids, so an order that ignored the lineage would put it first.
    CLUSTER_LOW = [10, 11, 12]
    CLUSTER_HIGH = [20, 21, 22]
    EDGES = [(10, 11), (11, 12), (10, 12), (20, 21), (21, 22), (20, 22)]

    def _assignments(self, low_number, high_number):
        return [
            LineageAssignment(self.CLUSTER_LOW,
                              ClusterLineage.orderKey(
                                  _reading(low_number, 1, "low"))),
            LineageAssignment(self.CLUSTER_HIGH,
                              ClusterLineage.orderKey(
                                  _reading(high_number, 1, "high"))),
        ]

    def test_the_clusters_are_ordered_by_their_lineage_numbers(self) -> None:
        registry = _registry(self.EDGES)
        assigned = registry.assignLineageKeys(self._assignments(1, 0))
        self.assertEqual(assigned, 6)
        # The cluster with the SMALLER lineage number is compiled first, even
        # though it carries the LARGER vertex ids.
        self.assertEqual(_ordered_pairs(registry),
                         [(20, 21), (20, 22), (21, 22),
                          (10, 11), (10, 12), (11, 12)])

    def test_reversing_the_lineage_numbers_reverses_the_blocks(self) -> None:
        registry = _registry(self.EDGES)
        registry.assignLineageKeys(self._assignments(0, 1))
        self.assertEqual(_ordered_pairs(registry),
                         [(10, 11), (10, 12), (11, 12),
                          (20, 21), (20, 22), (21, 22)])

    def test_an_anti_cluster_is_compiled_before_a_cluster(self) -> None:
        """A negative lineage number sorts before a positive one, which the
        fixed-width offset encoding is there to make true."""
        registry = _registry(self.EDGES)
        registry.assignLineageKeys(self._assignments(1, -1))
        self.assertEqual(_ordered_pairs(registry)[:3],
                         [(20, 21), (20, 22), (21, 22)])

    def test_the_order_is_invariant_under_a_relabelling_of_cells(self) -> None:
        """The acceptance case: registering the same edges in any order gives
        the same compilation order, because the order is read off the lineage
        and the endpoints and never off the registration index."""
        reference = _registry(self.EDGES)
        reference.assignLineageKeys(self._assignments(1, 0))
        expected_pairs = _ordered_pairs(reference)
        expected_keys = _ordered_keys(reference)
        generator = np.random.default_rng(20261202)
        for _ in range(6):
            shuffled = [self.EDGES[i]
                        for i in generator.permutation(len(self.EDGES))]
            registry = _registry(shuffled)
            registry.assignLineageKeys(self._assignments(1, 0))
            self.assertEqual(_ordered_pairs(registry), expected_pairs)
            self.assertEqual(_ordered_keys(registry), expected_keys)

    def test_a_relabelling_of_vertices_preserves_the_cluster_blocks(self) -> None:
        """A vertex relabelling rebuilds the order inside each cluster but
        never moves a mode out of its cluster's block, because the primary key
        is the lineage and the lineage is a relabelling-invariant integer.
        The induced permutation carries the exact exterior-algebra parity."""
        registry = _registry(self.EDGES)
        registry.assignLineageKeys(self._assignments(1, 0))
        relabelled = registry.relabeled({10: 12, 11: 11, 12: 10,
                                         20: 22, 21: 21, 22: 20})
        self.assertEqual(_ordered_keys(relabelled), _ordered_keys(registry))
        permutation = EdgeModeRegistry.orderPermutation(registry, relabelled)
        self.assertEqual(sorted(permutation), list(range(6)))
        parity = quantum.OccupationBitset.fromOccupiedModes(
            6, list(range(6))).permutationParity(list(permutation))
        self.assertIn(parity, (-1, +1))

    def test_an_unclaimed_mode_is_compiled_last(self) -> None:
        registry = _registry(self.EDGES + [(30, 31)])
        assigned = registry.assignLineageKeys(self._assignments(1, 0))
        self.assertEqual(assigned, 6)
        self.assertEqual(_ordered_pairs(registry)[-1], (30, 31))

    def test_a_mode_claimed_by_two_lineages_is_refused(self) -> None:
        registry = _registry(self.EDGES)
        overlapping = self._assignments(1, 0)
        overlapping[1] = LineageAssignment(
            [10, 11, 20, 21],
            ClusterLineage.orderKey(_reading(0, 1, "high")))
        with self.assertRaises(ValueError):
            registry.assignLineageKeys(overlapping)

    def test_setting_one_key_moves_only_that_mode(self) -> None:
        registry = _registry(self.EDGES)
        registry.assignLineageKeys(self._assignments(1, 0))
        before = registry.record(0)
        registry.setLineageKey(0, "lineage:0000000000:fermion:0:first")
        after = registry.record(0)
        self.assertEqual((after.vertexA, after.vertexB, after.orientationSign),
                         (before.vertexA, before.vertexB,
                          before.orientationSign))
        self.assertEqual(_ordered_pairs(registry)[0], (10, 11))

    def test_setting_an_unknown_mode_throws(self) -> None:
        registry = _registry(self.EDGES)
        with self.assertRaises(ValueError):
            registry.setLineageKey(99, "lineage")


class TestRegistryFromSpacetimeWithLineages(unittest.TestCase):
    """The end-to-end route: one mode per edge of a spacetime, each carrying
    the compilation-order key of the cluster whose support holds it."""

    @staticmethod
    def _spacetime(cells):
        return tessera.spacetime.Spacetime.fromVertexTuples(2, cells)

    def test_every_edge_of_a_claimed_cluster_is_assigned(self) -> None:
        spacetime = self._spacetime([[0, 1, 2], [3, 4, 5]])
        first = ClusterLineage.orderKey(_reading(0, 1, "first"))
        second = ClusterLineage.orderKey(_reading(1, 1, "second"))
        registry = EdgeModeRegistry.fromSpacetimeWithLineages(
            spacetime,
            [LineageAssignment([0, 1, 2], first),
             LineageAssignment([3, 4, 5], second)])
        self.assertEqual(registry.modeCount(), 6)
        keys = _ordered_keys(registry)
        self.assertEqual(keys, [first] * 3 + [second] * 3)

    def test_an_unclaimed_complex_keeps_the_declared_default(self) -> None:
        spacetime = self._spacetime([[0, 1, 2]])
        registry = EdgeModeRegistry.fromSpacetimeWithLineages(spacetime, [])
        self.assertEqual(set(_ordered_keys(registry)), {"~unassigned"})

    def test_the_default_key_sorts_after_every_lineage_key(self) -> None:
        spacetime = self._spacetime([[0, 1, 2], [3, 4, 5]])
        claimed = ClusterLineage.orderKey(_reading(1, 1, "z"))
        registry = EdgeModeRegistry.fromSpacetimeWithLineages(
            spacetime, [LineageAssignment([0, 1, 2], claimed)])
        self.assertEqual(_ordered_keys(registry),
                         [claimed] * 3 + ["~unassigned"] * 3)
        # No lineage key can ever reach the default, whatever integers it
        # encodes: every one begins with "lineage:" and the default with a
        # tilde, which sorts after every lower-case letter.
        largest = "lineage:" + "9" * 10 + ":fermion:" + "9" * 10 + ":zzzz"
        self.assertLess(largest, "~unassigned")


if __name__ == "__main__":
    unittest.main()
