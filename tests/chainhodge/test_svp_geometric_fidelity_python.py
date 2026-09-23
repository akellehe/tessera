# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The scaling verification plan's geometric-fidelity suite G1-G6 (#1208).

Each test measures the principal angles (in degrees) between the span of the
geometric images G_1 H_1 of the harmonic chains and the span W of the edge
integrals of the continuum harmonic forms the generator knows, writes the
plan's JSON record for the instance, and derives its verdict from that record.

The criteria are the plan's own:

  G1  F1, both signatures, N in {6, 8, 10}, jitter 0.25: angles <= 1e-8 degrees
      for L2; reported for GRASSMANN_ALL.
  G2  F2, both signatures: the single angle to d(theta) at the same criterion,
      and the boundary circles' images proportional to the signed lengths.
  G3  F3, both signatures, N in {6, 8, 12, 16, 24}: the angles must decrease,
      the estimated order is reported, and the Euclidean order must be >= 1.5;
      Lorentzian: monotone decrease reported, no threshold.
  G4  Sanity of the generators: the deficit angles of F1 and F2 vanish to
      round-off, those of F3 do not.
  G5  The Kontsevich-Segal family of the plan's section 7: the curved torus
      with period ratio 2 at epsilon in {0, 0.03, 0.1, 0.3, 0.6} and its
      Euclidean reference, N in {8, 12, 16, 24}, with the gap of every read.
      At epsilon >= 0.1 the convergence is second order and the gap is
      Euclidean-like; at epsilon = 0 neither holds.
  G6  Complex conformal factor a = 0.3 + 0.2i: convergence on the allowable
      Euclidean base, and the unrelated images on the Lorentzian base.

Every published angle of the plan's tables is carried in `PUBLISHED` and its
deviation is recorded; the rows the plan itself reports as converging are
required to reproduce it to 2%, and the rows it reports as unstable (the real
Lorentzian ones) are recorded only.
"""
import math
import time

import numpy as np
import pytest

from tessera import chainhodge as ch
from tests.chainhodge import _svp as svp
from tests.chainhodge._fixtures import conformal_torus, edges, flat_cylinder, flat_torus

KS = ch.Branch.KontsevichSegal
L2, ALL = ch.Preset.L2, ch.Preset.GRASSMANN_ALL

FLAT_CRITERION = "max principal angle <= 1e-8 degrees"
ORDER_CRITERION = "estimated order of convergence >= 1.5"

# The plan's published angles, for the deviation carried in every record.
PUBLISHED = {
    "G1": {("L2", False): {6: [0.0, 0.0], 8: [0.0, 0.0], 10: [0.0, 0.0]},
           ("L2", True): {6: [0.0, 0.0], 8: [0.0, 0.0], 10: [0.0, 0.0]},
           ("ALL", False): {6: [10.21, 5.78], 8: [9.40, 4.30], 10: [8.83, 5.16]},
           ("ALL", True): {6: [79.75, 63.18], 8: [77.28, 70.10], 10: [78.84, 65.25]}},
    "G2": {("L2", False): {6: [0.0], 8: [0.0], 10: [0.0]},
           ("L2", True): {6: [0.0], 8: [0.0], 10: [0.0]},
           ("ALL", False): {6: [8.44], 8: [7.55], 10: [8.24]},
           ("ALL", True): {6: [68.82], 8: [88.50], 10: [86.76]}},
    "G3": {("L2", False): {6: [4.541, 2.336], 8: [2.708, 1.271], 12: [1.251, 0.702],
                           16: [0.736, 0.406]},
           ("L2", True): {6: [33.85, 11.17], 8: [28.02, 9.17], 12: [16.77, 11.45],
                          16: [12.82, 10.71]},
           ("ALL", False): {6: [14.84, 12.19], 8: [13.84, 12.70], 12: [13.66, 12.56],
                            16: [13.49, 12.56]},
           ("ALL", True): {6: [89.41, 85.54], 8: [87.96, 68.45], 12: [88.18, 68.44],
                           16: [89.41, 79.44]}},
    "G5": {"euclidean": {8: [2.65, 1.82], 12: [1.24, 1.04], 16: [0.75, 0.60], 24: [0.36, 0.34]},
           0.0: {8: [3.86, 2.51], 12: [33.99, 5.07], 16: [2.24, 1.47], 24: [12.39, 3.08]},
           0.03: {8: [3.76, 2.45], 12: [5.92, 2.70], 16: [2.02, 1.42], 24: [2.07, 1.09]},
           0.1: {8: [3.40, 2.16], 12: [2.60, 1.85], 16: [1.55, 1.20], 24: [0.97, 0.67]},
           0.3: {8: [3.14, 1.68], 12: [1.59, 1.39], 16: [1.04, 0.90], 24: [0.58, 0.48]},
           0.6: {8: [3.02, 1.54], 12: [1.44, 1.07], 16: [0.89, 0.69], 24: [0.46, 0.39]}},
    "G6": {"euclidean": {8: [3.19, 2.19], 12: [1.49, 1.27], 16: [0.91, 0.73]},
           "lorentzian": {8: [89.1, 65.0], 12: [89.9, 70.3], 16: [89.3, 68.4]}},
}


def deviation(angles, published):
    """The relative deviation of measured angles from the plan's published
    ones, entry by entry (absolute where the published value is zero)."""
    if published is None:
        return None
    return [float(abs(a - p) / p) if p else float(abs(a - p))
            for a, p in zip(angles, published)]


CROSSOVER = 512            # the library's own default (ChainHodge::kDefaultCrossoverDimension)


def measure(K, s, W, preset, epsilon=math.nan, crossover=CROSSOVER):
    """One instance: the harmonic read at degree 1 and its angles to W. The
    crossover is the library's default, so the kernel comes from the dense SVD
    up to 512 cells and from the sparse rank-revealing QR above it; the two
    agree to five digits on every mesh of these tables."""
    started = time.time()
    hodge = ch.ChainHodge(K, s, preset, KS, crossover, epsilon)
    read = hodge.harmonicChains(1)
    return hodge, read, svp.angles_deg(read.images, W), started


class TestG1FlatJitteredTorus:
    """G1: the flat jittered torus F1 at jitter 0.25, both signatures, at the
    plan's sizes N = 6, 8, 10 (the largest, N = 10, was missing)."""

    @pytest.mark.parametrize("lorentz", [False, True])
    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_whitney_images_are_the_continuum_forms(self, N, lorentz, svp_records):
        K, s, W = flat_torus(N, 0.25, lorentz, seed=1)
        hodge, read, angles, started = measure(K, s, W, L2)
        record = svp_records.instance(
            "G1", "F1", {"N": N, "jitter": 0.25, "seed": 1,
                         "signature": "lorentzian" if lorentz else "euclidean"},
            hodge, read, started=started, rank_report=svp.rank_report(hodge),
            angles_deg=list(angles), criterion=FLAT_CRITERION,
            published=PUBLISHED["G1"][("L2", lorentz)][N],
            deviation=deviation(angles, PUBLISHED["G1"][("L2", lorentz)][N]))
        assert record["nullity"] == record["betti"][1] == 2
        assert record["rank_conditions"]["kernel_is_harmonic"]
        assert max(record["angles_deg"]) <= 1e-8

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_grassmann_angles_are_reported_and_do_not_decrease(self, lorentz, svp_records):
        """The plan reports the GRASSMANN_ALL angles without a threshold; its
        Finding 1 is that they are O(1) and do not decrease from N = 6 to
        N = 10, which is what this asserts (estimated order < 0.5)."""
        sizes, largest = [], []
        for N in (6, 8, 10):
            K, s, W = flat_torus(N, 0.25, lorentz, seed=1)
            hodge, read, angles, started = measure(K, s, W, ALL)
            published = PUBLISHED["G1"][("ALL", lorentz)][N]
            svp_records.instance(
                "G1", "F1", {"N": N, "jitter": 0.25, "seed": 1,
                             "signature": "lorentzian" if lorentz else "euclidean"},
                hodge, read, started=started, rank_report=svp.rank_report(hodge),
                angles_deg=list(angles), criterion="reported (no threshold)",
                published=published, deviation=deviation(angles, published))
            assert read.nullity == 2
            assert max(deviation(angles, published)) < 0.02      # the plan's own row
            sizes.append(N)
            largest.append(float(angles[0]))
        order = svp.convergence(sizes, largest)
        svp_records.write(test="G1", family="F1", preset="GRASSMANN_ALL",
                          params={"N": sizes, "jitter": 0.25, "seed": 1,
                                  "signature": "lorentzian" if lorentz else "euclidean"},
                          angles_deg=largest, convergence=order,
                          criterion="Finding 1: O(1) angles, no decrease (order < 0.5)")
        assert min(largest) > 1.0
        assert order["fit"] < 0.5


class TestG2FlatCylinder:
    """G2: the flat cylinder cobordism F2 with two boundary circles, both
    signatures, at the plan's sizes (N, L) = (6, 4), (8, 6), (10, 8)."""

    @pytest.mark.parametrize("lorentz", [False, True])
    @pytest.mark.parametrize("NL", [(6, 4), (8, 6), (10, 8)])
    def test_whitney_image_is_d_theta(self, NL, lorentz, svp_records):
        N, L = NL
        K, s, W = flat_cylinder(N, L, 0.25, lorentz, seed=2)
        hodge, read, angles, started = measure(K, s, W, L2)
        record = svp_records.instance(
            "G2", "F2", {"N": N, "L": L, "jitter": 0.25, "seed": 2,
                         "signature": "lorentzian" if lorentz else "euclidean"},
            hodge, read, started=started, rank_report=svp.rank_report(hodge),
            angles_deg=list(angles), criterion=FLAT_CRITERION,
            published=PUBLISHED["G2"][("L2", lorentz)][N],
            deviation=deviation(angles, PUBLISHED["G2"][("L2", lorentz)][N]))
        assert record["nullity"] == record["betti"][1] == 1
        assert max(record["angles_deg"]) <= 1e-8

    @pytest.mark.parametrize("lorentz", [False, True])
    @pytest.mark.parametrize("NL", [(6, 4), (8, 6), (10, 8)])
    def test_boundary_circle_images_are_the_signed_lengths(self, NL, lorentz, svp_records):
        """The plan's second G2 clause: on each boundary circle the geometric
        image is proportional to (l_e), the signed edge lengths of CH
        Proposition 7.4. The comparison uses l_e = sqrt(s_e) with the sign of
        the edge's direction around the circle -- the metric alone, not the
        generator's continuum form."""
        N, L = NL
        K, s, W = flat_cylinder(N, L, 0.25, lorentz, seed=2)
        hodge, read, _, started = measure(K, s, W, L2)
        index = {e: i for i, e in enumerate(edges(K))}
        z = read.images[:, 0]
        circles = []
        for layer in (0, L):
            rows, signs = [], []
            for j in range(N):
                a, b = layer * N + j, layer * N + (j + 1) % N
                rows.append(index[(min(a, b), max(a, b))])
                signs.append(1.0 if a < b else -1.0)
            lengths = np.array(signs) * np.sqrt(np.array([s[i] for i in rows], dtype=complex))
            circles.append(float(svp.angles_deg(z[rows].reshape(-1, 1),
                                                lengths.reshape(-1, 1))[0]))
        record = svp_records.instance(
            "G2", "F2", {"N": N, "L": L, "jitter": 0.25, "seed": 2,
                         "signature": "lorentzian" if lorentz else "euclidean"},
            hodge, read, started=started, boundary_circle_angles_deg=circles,
            criterion="angle between the image on each boundary circle and "
                      "(sign_e l_e) <= 1e-8 degrees")
        assert max(record["boundary_circle_angles_deg"]) <= 1e-8

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_grassmann_angles_are_reported_and_do_not_decrease(self, lorentz, svp_records):
        """The plan's GRASSMANN_ALL column of the G2 table, reported: O(1)
        angles to d(theta) that do not decrease with the mesh."""
        sizes, largest = [], []
        for (N, L) in ((6, 4), (8, 6), (10, 8)):
            K, s, W = flat_cylinder(N, L, 0.25, lorentz, seed=2)
            hodge, read, angles, started = measure(K, s, W, ALL)
            published = PUBLISHED["G2"][("ALL", lorentz)][N]
            svp_records.instance(
                "G2", "F2", {"N": N, "L": L, "jitter": 0.25, "seed": 2,
                             "signature": "lorentzian" if lorentz else "euclidean"},
                hodge, read, started=started, rank_report=svp.rank_report(hodge),
                angles_deg=list(angles), criterion="reported (no threshold)",
                published=published, deviation=deviation(angles, published))
            assert read.nullity == 1
            assert max(deviation(angles, published)) < 0.02
            sizes.append(N)
            largest.append(float(angles[0]))
        order = svp.convergence(sizes, largest)
        svp_records.write(test="G2", family="F2", preset="GRASSMANN_ALL",
                          params={"N": sizes, "jitter": 0.25, "seed": 2,
                                  "signature": "lorentzian" if lorentz else "euclidean"},
                          angles_deg=largest, convergence=order,
                          criterion="Finding 1: O(1) angles, no decrease (order < 0.5)")
        assert min(largest) > 1.0
        assert order["fit"] < 0.5


class TestG3ConformallyFlatTorus:
    """G3: the conformally flat (genuinely curved) torus F3 at a = 0.3, jitter
    0.15, over the plan's whole sequence N = 6, 8, 12, 16, 24 -- so that the
    order comes from three successive doublings, not from one pair of meshes."""

    SIZES = (6, 8, 12, 16, 24)

    def sequence(self, preset, lorentz, svp_records, test="G3"):
        largest, rows = [], []
        for N in self.SIZES:
            K, s, W = conformal_torus(N, 0.3, 0.15, lorentz, seed=1)
            hodge, read, angles, started = measure(K, s, W, preset)
            name = "L2" if preset == L2 else "ALL"
            published = PUBLISHED["G3"][(name, lorentz)].get(N)
            svp_records.instance(
                test, "F3", {"N": N, "amp": 0.3, "jitter": 0.15, "seed": 1,
                             "signature": "lorentzian" if lorentz else "euclidean"},
                hodge, read, started=started, rank_report=svp.rank_report(hodge),
                angles_deg=list(angles), criterion=ORDER_CRITERION if not lorentz
                else "reported (no threshold)",
                published=published, deviation=deviation(angles, published))
            assert read.nullity == 2
            rows.append({"N": N, "angles": list(map(float, angles)), "gap": read.gap,
                         "published": published, "deviation": deviation(angles, published)})
            largest.append(float(angles[0]))
        order = svp.convergence(self.SIZES, largest)
        svp_records.write(test=test, family="F3", preset="L2" if preset == L2 else "GRASSMANN_ALL",
                          params={"N": list(self.SIZES), "amp": 0.3, "jitter": 0.15, "seed": 1,
                                  "signature": "lorentzian" if lorentz else "euclidean"},
                          angles_deg=largest, rows=rows, convergence=order)
        return rows, order

    def test_euclidean_converges_at_second_order(self, svp_records):
        rows, order = self.sequence(L2, False, svp_records)
        for row in rows:
            if row["published"]:
                assert max(row["deviation"]) < 0.02
        assert order["monotone"]
        assert order["fit"] >= 1.5
        assert min(order["doublings"].values()) >= 1.5

    def test_lorentzian_is_reported(self, svp_records):
        """The plan sets no threshold here: the monotone decrease and the
        estimated order are reported. Measured: the largest angle decreases
        monotonically, at order 0.8 over the sequence, while the gap of the
        stacked matrix falls by four decades."""
        rows, order = self.sequence(L2, True, svp_records)
        for row in rows:
            if row["published"]:
                assert max(row["deviation"]) < 0.05
        assert order["monotone"]
        assert math.isfinite(order["fit"])
        assert rows[0]["gap"] > 100.0 * rows[-1]["gap"]

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_grassmann_does_not_converge(self, lorentz, svp_records):
        """Finding 1 again, on a curved geometry: the GRASSMANN_ALL angles stay
        at O(1) over the whole sequence (estimated order < 0.5)."""
        rows, order = self.sequence(ALL, lorentz, svp_records)
        assert min(row["angles"][0] for row in rows) > 10.0
        assert order["fit"] < 0.5


class TestG4GeneratorSanity:
    """G4: the generators' own curvature. The deficit angle at a vertex is read
    from the squared lengths alone as the holonomy of the development of its
    star (`_svp.deficit_angles`), which is a ratio of polynomials in s and
    needs no branch choice for an angle; for F1 and F2 it must vanish to
    round-off, for F3 it must not."""

    FLAT = 1e-10
    CURVED = 1e-3

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_f1_is_flat(self, lorentz, svp_records):
        K, s, _ = flat_torus(8, 0.25, lorentz, seed=1)
        table = dict(zip(edges(K), s))
        deficits = svp.deficit_angles(svp._counter_clockwise(8), table)
        largest = max(abs(v) for v in deficits.values())
        record = svp_records.write(
            test="G4", family="F1", preset="none",
            params={"N": 8, "jitter": 0.25, "seed": 1,
                    "signature": "lorentzian" if lorentz else "euclidean"},
            n0=K.numSimplices(0), n1=K.numSimplices(1), n2=K.numSimplices(2),
            betti=list(K.bettiNumbers()), max_abs_deficit=largest,
            vertices=len(deficits), criterion=f"max |deficit| <= {self.FLAT}")
        assert record["vertices"] == 64
        assert record["max_abs_deficit"] <= self.FLAT

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_f2_is_flat(self, lorentz, svp_records):
        """Interior vertices only: a boundary vertex of the cylinder has a
        half-star, whose development measures the geodesic curvature of the
        boundary circle rather than a deficit."""
        K, s, _ = flat_cylinder(6, 4, 0.25, lorentz, seed=2)
        table = dict(zip(edges(K), s))
        deficits = svp.deficit_angles(svp.cylinder_cells(6, 4), table)
        largest = max(abs(v) for v in deficits.values())
        record = svp_records.write(
            test="G4", family="F2", preset="none",
            params={"N": 6, "L": 4, "jitter": 0.25, "seed": 2,
                    "signature": "lorentzian" if lorentz else "euclidean"},
            n0=K.numSimplices(0), n1=K.numSimplices(1), n2=K.numSimplices(2),
            betti=list(K.bettiNumbers()), max_abs_deficit=largest,
            vertices=len(deficits), criterion=f"max |deficit| <= {self.FLAT}")
        assert record["vertices"] == 18                 # the three interior layers
        assert record["max_abs_deficit"] <= self.FLAT

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_f3_is_curved(self, lorentz, svp_records):
        """The conformally flat torus has nonzero deficit angles, and they sum
        to 2 pi chi = 0 (Gauss-Bonnet) in both signatures."""
        K, s, _ = conformal_torus(8, 0.3, 0.15, lorentz, seed=1)
        table = dict(zip(edges(K), s))
        deficits = svp.deficit_angles(svp._counter_clockwise(8), table)
        largest = max(abs(v) for v in deficits.values())
        total = abs(sum(deficits.values()))
        record = svp_records.write(
            test="G4", family="F3", preset="none",
            params={"N": 8, "amp": 0.3, "jitter": 0.15, "seed": 1,
                    "signature": "lorentzian" if lorentz else "euclidean"},
            n0=K.numSimplices(0), n1=K.numSimplices(1), n2=K.numSimplices(2),
            betti=list(K.bettiNumbers()), max_abs_deficit=largest,
            gauss_bonnet=total, vertices=len(deficits),
            criterion=f"max |deficit| >= {self.CURVED} and |sum| <= 1e-10 (2 pi chi = 0)")
        assert record["max_abs_deficit"] >= self.CURVED
        assert record["gauss_bonnet"] <= 1e-10

    def test_the_euclidean_reading_agrees_with_the_angle_defect(self, svp_records):
        """On Euclidean data the holonomy reading is the ordinary angle defect
        2 pi - sum(theta) of the law of cosines, which is the independent
        check that it measures curvature and not a convention."""
        K, s, _ = conformal_torus(8, 0.3, 0.15, False, seed=1)
        table = dict(zip(edges(K), s))
        holonomy = svp.deficit_angles(svp._counter_clockwise(8), table)
        classical = svp.euclidean_deficit_angles(svp._counter_clockwise(8), table)
        difference = max(abs(holonomy[v].real - classical[v]) for v in classical)
        imaginary = max(abs(holonomy[v].imag) for v in classical)
        svp_records.write(test="G4", family="F3", preset="none",
                          params={"N": 8, "amp": 0.3, "jitter": 0.15, "seed": 1,
                                  "signature": "euclidean"},
                          residuals={"holonomy_vs_angle_defect": difference,
                                     "imaginary_part": imaginary},
                          criterion="the two readings agree to 1e-12")
        assert difference <= 1e-12 and imaginary <= 1e-12


@pytest.mark.slow
class TestG5KontsevichSegalFamily:
    """G5: the plan's section 7 table. The curved torus with period ratio 2
    (with equal periods the lattice diagonals are near null), jitter 0.15, at
    epsilon in {0, 0.03, 0.1, 0.3, 0.6} and the Euclidean reference, over
    N = 8, 12, 16, 24. Entries: the principal angles and the gap
    varsigma_r/varsigma_{r+1} of the stacked matrix."""

    SIZES = (8, 12, 16, 24)
    EPSILONS = (0.0, 0.03, 0.1, 0.3, 0.6)

    @pytest.fixture(scope="class")
    def table(self, svp_records):
        rows = {}
        for key in ("euclidean",) + self.EPSILONS:
            epsilon = None if key == "euclidean" else key
            rows[key] = []
            for N in self.SIZES:
                K, s, W, _, _ = svp.ratio_torus(N, epsilon)
                hodge, read, angles, started = measure(
                    K, s, W, L2, math.nan if epsilon is None else epsilon)
                published = PUBLISHED["G5"][key][N]
                record = svp_records.instance(
                    "G5", "F3 (period ratio 2)",
                    {"N": N, "amp": 0.3, "jitter": 0.15, "seed": 1, "Lt": 1.0, "Lx": 2.0,
                     "epsilon": epsilon,
                     "signature": "euclidean" if epsilon is None else "lorentzian"},
                    hodge, read, started=started, angles_deg=list(angles),
                    allowable=hodge.certificate().allowable,
                    margin=hodge.certificate().margin,
                    published=published, deviation=deviation(angles, published),
                    criterion="epsilon >= 0.1: order >= 1.5 and gap >= 0.1 x the "
                              "Euclidean reference; epsilon <= 0.03: reported")
                assert read.nullity == 2
                rows[key].append(record)
        return rows

    def test_the_published_table_is_reproduced(self, table, svp_records):
        """Every allowable row of the plan's table, to 2%. The epsilon = 0 row
        is recorded only: it is the row the plan reports as unstable (the
        harmonic representative is ill-conditioned there), and its angles are
        non-monotone in N."""
        for key, rows in table.items():
            for record in rows:
                if key == 0.0:
                    assert record["deviation"] is not None
                    continue
                assert max(record["deviation"]) < 0.02, (key, record["params"]["N"])

    @pytest.mark.parametrize("epsilon", [0.3, 0.6,
                                         pytest.param(0.1, marks=pytest.mark.xfail(
                                             reason="the plan's own angles at epsilon = 0.1 give "
                                                    "order 1.14 over N = 8..24 (1.14 and 1.42 from "
                                                    "the two doublings), not the second order its "
                                                    "caption claims; reproduced exactly, so this "
                                                    "is a finding about the plan's value",
                                             strict=True))])
    def test_the_rotated_family_converges_at_second_order(self, epsilon, table, svp_records):
        angles = [record["angles_deg"][0] for record in table[epsilon]]
        order = svp.convergence(self.SIZES, angles)
        svp_records.write(test="G5", family="F3 (period ratio 2)", preset="L2",
                          params={"N": list(self.SIZES), "epsilon": epsilon,
                                  "amp": 0.3, "jitter": 0.15, "seed": 1},
                          angles_deg=angles, convergence=order, criterion=ORDER_CRITERION)
        assert order["monotone"]
        assert order["fit"] >= 1.5

    def test_the_allowable_gaps_are_euclidean_like(self, table, svp_records):
        reference = [record["gap"] for record in table["euclidean"]]
        ratios = {}
        for epsilon in self.EPSILONS:
            ratios[epsilon] = [record["gap"] / g for record, g in zip(table[epsilon], reference)]
        svp_records.write(test="G5", family="F3 (period ratio 2)", preset="L2",
                          params={"N": list(self.SIZES), "epsilon": list(self.EPSILONS)},
                          gap_ratio_to_euclidean=ratios,
                          criterion="epsilon >= 0.1: gap >= 0.1 x the Euclidean reference at "
                                    "every N; epsilon = 0: below it under refinement")
        for epsilon in (0.1, 0.3, 0.6):
            assert min(ratios[epsilon]) >= 0.1, (epsilon, ratios[epsilon])
        assert max(ratios[0.0][1:]) < 0.1            # N >= 12 on the real Lorentzian row

    def test_the_real_lorentzian_row_is_reported_with_its_gap(self, table, svp_records):
        """epsilon = 0: neither the convergence nor the gap holds. The angles
        are non-monotone in N and the gap collapses; the plan reports the row
        and sets no threshold on it (Requirement 2 of the integration
        specification: never alone, always beside the family and its gap)."""
        angles = [record["angles_deg"][0] for record in table[0.0]]
        gaps = [record["gap"] for record in table[0.0]]
        order = svp.convergence(self.SIZES, angles)
        svp_records.write(test="G5", family="F3 (period ratio 2)", preset="L2",
                          params={"N": list(self.SIZES), "epsilon": 0.0},
                          angles_deg=angles, gap=gaps, convergence=order,
                          criterion="reported; no threshold")
        assert not order["monotone"]
        assert gaps[0] > 10.0 * gaps[-1]
        assert not any(record["allowable"] for record in table[0.0])
        assert all(record["allowable"] for key in (0.03, 0.1, 0.3, 0.6)
                   for record in table[key])


class TestG6ComplexConformalFactor:
    """G6: the complex conformal factor a = 0.3 + 0.2i on the same period-ratio
    2 torus. On the Euclidean base the deformation is allowable and the
    convergence persists; on the Lorentzian base the argument sum reaches pi,
    the instance is not allowable, and the harmonic images are unrelated to the
    geometry (the row the tests were missing)."""

    SIZES = (8, 12, 16)
    AMPLITUDE = 0.3 + 0.2j

    def sequence(self, lorentz, svp_records):
        rows = []
        for N in self.SIZES:
            K, s, W, _, _ = svp.ratio_torus(N, 0.0 if lorentz else None, amp=self.AMPLITUDE)
            hodge, read, angles, started = measure(K, s, W, L2)
            published = PUBLISHED["G6"]["lorentzian" if lorentz else "euclidean"][N]
            rows.append(svp_records.instance(
                "G6", "F3 (period ratio 2, complex a)",
                {"N": N, "amp": [self.AMPLITUDE.real, self.AMPLITUDE.imag], "jitter": 0.15,
                 "seed": 1, "base": "lorentzian" if lorentz else "euclidean"},
                hodge, read, started=started, angles_deg=list(angles),
                allowable=hodge.certificate().allowable, margin=hodge.certificate().margin,
                published=published, deviation=deviation(angles, published),
                criterion="Euclidean base: order >= 1.5 and allowable; "
                          "Lorentzian base: not allowable, angles reported"))
            assert read.nullity == 2
        return rows

    def test_euclidean_base_is_allowable_and_converges(self, svp_records):
        rows = self.sequence(False, svp_records)
        order = svp.convergence(self.SIZES, [row["angles_deg"][0] for row in rows])
        svp_records.write(test="G6", family="F3 (period ratio 2, complex a)", preset="L2",
                          params={"N": list(self.SIZES), "base": "euclidean"},
                          angles_deg=[row["angles_deg"][0] for row in rows],
                          convergence=order, criterion=ORDER_CRITERION)
        for row in rows:
            assert row["allowable"] and row["margin"] > 0.0
            assert max(row["deviation"]) < 0.02
        assert order["monotone"] and order["fit"] >= 1.5

    def test_lorentzian_base_is_not_allowable_and_is_unrelated_to_the_geometry(self, svp_records):
        """The plan's second G6 row: the argument sum of every top simplex is
        at or above pi, so the instance is outside the Kontsevich-Segal domain,
        and the angles stay at 54-90 degrees with no decrease. The published
        row (89.1, 65.0 / 89.9, 70.3 / 89.3, 68.4) is reproduced in its
        largest angle to a degree; the second angle, off the allowable domain,
        is not stable and is recorded rather than asserted."""
        rows = self.sequence(True, svp_records)
        order = svp.convergence(self.SIZES, [row["angles_deg"][0] for row in rows])
        svp_records.write(test="G6", family="F3 (period ratio 2, complex a)", preset="L2",
                          params={"N": list(self.SIZES), "base": "lorentzian"},
                          angles_deg=[row["angles_deg"] for row in rows], convergence=order,
                          criterion="not allowable; angles O(1) with no decrease")
        for row in rows:
            assert not row["allowable"] and row["margin"] <= 0.0
            assert min(row["angles_deg"]) > 45.0
            assert row["deviation"][0] < 0.02
        assert order["fit"] < 0.5
