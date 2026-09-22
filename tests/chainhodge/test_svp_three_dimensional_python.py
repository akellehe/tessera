# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The scaling verification plan at d >= 3: direct values for M_1 and M_2 and
the F7 prism families (#1208).

The plan lists F7 -- "the tessera staircase prism of an F1/F2 base gives
3-complexes with known b_k" -- as not yet exercised, "listed so the C++ suite
covers d = 3 from the start", and its open item 2 records that M_2 for d = 3
must be implemented (CH section 6, Option A, Gram-determinant rule). The
audit's item 41 adds that M_1 and M_2 have no direct value tests at d >= 3.

This module supplies both:

  * direct values. The Whitney mass matrix of one simplex is integrated from
    its coordinates -- barycentric gradients from the inverse edge matrix, the
    inner product of k-forms as the determinant of pairwise inner products
    under the ambient inverse metric, and a quadrature rule exact for the
    quadratic integrand -- in a Euclidean and in a Minkowski embedding. No
    Gram matrix of squared lengths, branch rule or closed form of the library
    enters it. The dense oracle of `_svp` (the specification's closed form, in
    any dimension) is checked against the library at d = 2, 3 and 4 as well.
  * the F7 families. The staircase prism of a flat jittered torus and of a
    flat cylinder, in both signatures: the Betti numbers of the product, the
    harmonic dimension at every degree, and -- the geometric content -- the
    geometric images of the harmonic 1- and 2-chains against the continuum
    absolute harmonic forms dt, dx and dt ^ dx of the flat product, which is
    the plan's G1 criterion carried to d = 3 and the first test that uses the
    d >= 3 values of both M_1 and M_2.
"""
import itertools
import math
import time

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge import _svp as svp
from tests.chainhodge._fixtures import edges, random_allowable, torus_cells

KS = ch.Branch.KontsevichSegal
L2 = ch.Preset.L2
WM = ch.WhitneyMass
FLAT_CRITERION = "max principal angle <= 1e-8 degrees"

EUCLIDEAN_3 = np.eye(3)
MINKOWSKI_3 = np.diag([-1.0, 1.0, 1.0])

TETRAHEDRA = {
    "unit corner": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "irregular": np.array([[0.1, -0.2, 0.3], [1.3, 0.1, -0.2], [-0.2, 1.1, 0.4], [0.3, 0.2, 1.4]]),
}


def squared_lengths(points, metric):
    """s_e = <x_b - x_a, x_b - x_a> in the ambient metric, in the canonical
    edge order of the single simplex [0 < 1 < ... < d]."""
    out = []
    for a, b in itertools.combinations(range(len(points)), 2):
        d = points[b] - points[a]
        out.append(complex(d @ metric @ d))
    return out


@pytest.mark.parametrize("name", sorted(TETRAHEDRA))
@pytest.mark.parametrize("signature", ["euclidean", "lorentzian"])
@pytest.mark.parametrize("k", [1, 2])
def test_tetrahedron_mass_matrices_have_the_integrated_values(name, signature, k, svp_records):
    """The direct value test at d = 3: M_1 and M_2 of one tetrahedron against
    the integral of the Whitney forms over its coordinates. On the Minkowski
    embedding det g_T < 0 and the library's volume is i times the coordinate
    volume, so the comparison carries that factor."""
    started = time.time()
    points = TETRAHEDRA[name]
    metric = EUCLIDEAN_3 if signature == "euclidean" else MINKOWSKI_3
    s = squared_lengths(points, metric)
    K = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    code = WM.assemble(K, s, k, KS).toarray()
    integrated, faces = svp.simplex_mass_from_coordinates(points, metric, k)
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    order = [cells.index(f) for f in faces]
    factor = 1.0 if signature == "euclidean" else 1j
    expected = factor * integrated
    agreement = float(np.linalg.norm(code[np.ix_(order, order)] - expected)
                      / np.linalg.norm(expected))
    hodge = ch.ChainHodge(K, s, L2, KS)
    record = svp_records.instance(
        "F7 (values)", f"tetrahedron {name}",
        {"d": 3, "k": k, "signature": signature}, hodge, k=k, started=started,
        oracle_agreement={"coordinate_integral": agreement},
        criterion="relative agreement with the coordinate integral <= 1e-13")
    assert record["oracle_agreement"]["coordinate_integral"] <= 1e-13


@pytest.mark.parametrize("signature", ["euclidean", "lorentzian"])
def test_triangle_mass_matrices_have_the_integrated_values(signature, svp_records):
    """The same reading at d = 2, where the closed form is already tested:
    the two agree, so the d = 3 comparison above is the same measurement."""
    points = np.array([[0.0, 0.0], [1.2, 0.1], [-0.3, 0.9]])
    metric = np.eye(2) if signature == "euclidean" else np.diag([-1.0, 1.0])
    s = squared_lengths(points, metric)
    K = cob.ChainComplex.fromTopCells([[0, 1, 2]])
    factor = 1.0 if signature == "euclidean" else 1j
    agreement = {}
    for k in (0, 1, 2):
        code = WM.assemble(K, s, k, KS).toarray()
        integrated, faces = svp.simplex_mass_from_coordinates(points, metric, k)
        cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
        order = [cells.index(f) for f in faces]
        expected = factor * integrated
        agreement[f"M_{k}"] = float(np.linalg.norm(code[np.ix_(order, order)] - expected)
                                    / np.linalg.norm(expected))
    svp_records.write(test="F7 (values)", family="triangle", preset="L2",
                      params={"d": 2, "signature": signature}, oracle_agreement=agreement,
                      criterion="relative agreement with the coordinate integral <= 1e-13")
    assert max(agreement.values()) <= 1e-13


@pytest.mark.parametrize("name,cells", [("3-complex", [[0, 1, 2, 3], [1, 2, 3, 4]]),
                                        ("4-complex", [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])])
def test_the_dense_oracle_reproduces_every_degree_above_two_dimensions(name, cells, svp_records):
    """The dense oracle of `_svp` is the specification's closed form in any
    dimension; at d = 3 and d = 4 it reproduces the library at every degree,
    which extends the d = 2 oracle comparison of the Whitney mass tests."""
    rng = np.random.default_rng(37)
    K = cob.ChainComplex.fromTopCells(cells)
    s = random_allowable(K, rng, 0.2)
    hodge = ch.ChainHodge(K, s, L2, KS)
    agreement = {}
    for k in range(K.dimension() + 1):
        code = WM.assemble(K, s, k).toarray()
        agreement[f"M_{k}"] = float(np.linalg.norm(code - svp.whitney_reference(K, s, k))
                                    / np.linalg.norm(code))
    record = svp_records.instance("F7 (values)", name, {"d": K.dimension()}, hodge,
                                  oracle_agreement=agreement,
                                  criterion="relative agreement <= 1e2 tau")
    assert max(record["oracle_agreement"].values()) <= 100.0 * record["tau"]


def torus_base(N, jitter=0.15, seed=1, Lt=1.0, Lx=2.0):
    """An F1 base for the prism: the jittered N x N torus with its coordinates."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) * Lt / N,
                                   (j + jitter * rng.uniform(-1, 1)) * Lx / N])
              for i in range(N) for j in range(N)}
    return cells, coords, N * N, (Lt, Lx)


def cylinder_base(N, L, jitter=0.15, seed=2, Lx=1.0):
    """An F2 base for the prism: the flat cylinder, jittered on its interior
    layers, with periods (None, Lx) -- it wraps in x only."""
    rng = np.random.default_rng(seed)
    cells = svp.cylinder_cells(N, L)
    coords = {i * N + j: np.array([(i + (jitter * rng.uniform(-1, 1) if 0 < i < L else 0.0)) / N,
                                   (j + jitter * rng.uniform(-1, 1)) * Lx / N])
              for i in range(L + 1) for j in range(N)}
    return cells, coords, (L + 1) * N, (None, Lx)


class TestF7PrismFamilies:
    """F7: the staircase prism of an F1 or F2 base. The product of a flat base
    with an interval is flat, its absolute harmonic 1-forms are dt and dx and
    its absolute harmonic 2-form is dt ^ dx (their normal components on the two
    boundary copies vanish), so the plan's flat criterion applies at both
    degrees."""

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_torus_prism(self, lorentz, svp_records):
        cells, coords, stride, periods = torus_base(4)
        K, tops = svp.prism_complex(cells, 2)
        s, W1, W2 = svp.prism_geometry(K, coords, 4, 2, stride, lorentz=lorentz,
                                       periods=periods)
        started = time.time()
        hodge = ch.ChainHodge(K, s, L2, KS)
        reads = {k: hodge.harmonicChains(k) for k in range(4)}
        angles = {1: list(svp.angles_deg(reads[1].images, W1)),
                  2: list(svp.angles_deg(reads[2].images, W2))}
        record = svp_records.instance(
            "F7", "prism of F1", {"N": 4, "layers": 2, "jitter": 0.15, "seed": 1,
                                  "signature": "lorentzian" if lorentz else "euclidean",
                                  "top_cells": tops},
            hodge, reads[1], started=started,
            n3=K.numSimplices(3),
            nullities=[reads[k].nullity for k in range(4)],
            gaps=[reads[k].gap for k in range(4)],
            angles_deg=angles[1], angles_deg_degree_two=angles[2],
            criterion="b_k = (1, 2, 1, 0), dim H_k = b_k, and " + FLAT_CRITERION
                      + " at degrees 1 and 2")
        assert record["betti"] == [1, 2, 1, 0]
        assert record["nullities"] == record["betti"]
        assert max(record["angles_deg"]) <= 1e-8
        assert max(record["angles_deg_degree_two"]) <= 1e-8

    @pytest.mark.parametrize("lorentz", [False, True])
    def test_cylinder_prism(self, lorentz, svp_records):
        """The prism of the cylinder cobordism is S^1 x I x I, which retracts
        to the circle: b = (1, 1, 0, 0), and the absolute harmonic 1-form is
        d(theta) again."""
        N, L = 6, 3
        cells, coords, stride, periods = cylinder_base(N, L)
        K, tops = svp.prism_complex(cells, 2)
        s, W1, _ = svp.prism_geometry(K, coords, N, 2, stride, lorentz=lorentz,
                                      periods=periods, height=1.0 / N)
        started = time.time()
        hodge = ch.ChainHodge(K, s, L2, KS)
        reads = {k: hodge.harmonicChains(k) for k in range(4)}
        angles = list(svp.angles_deg(reads[1].images, W1[:, 1].reshape(-1, 1)))
        record = svp_records.instance(
            "F7", "prism of F2", {"N": N, "L": L, "layers": 2, "jitter": 0.15, "seed": 2,
                                  "signature": "lorentzian" if lorentz else "euclidean",
                                  "top_cells": tops},
            hodge, reads[1], started=started, n3=K.numSimplices(3),
            nullities=[reads[k].nullity for k in range(4)],
            gaps=[reads[k].gap for k in range(4)], angles_deg=angles,
            criterion="b_k = (1, 1, 0, 0), dim H_k = b_k, and " + FLAT_CRITERION)
        assert record["betti"] == [1, 1, 0, 0]
        assert record["nullities"] == record["betti"]
        assert max(record["angles_deg"]) <= 1e-8
