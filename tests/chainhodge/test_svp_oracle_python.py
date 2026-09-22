# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The scaling verification plan's cross-checks X (#1208).

"Python oracle versus C++ on identical inputs (relative agreement <= 1e2 tau);
exact rational arithmetic for N <= 6 on rational s_e for GRASSMANN_ALL
(polynomial entries) and for the integer topology."

The oracle is the dense one of `_svp`: the Whitney mass matrices in any
dimension, the Grassmann chain metrics, and the two pencils built from them,
each an independent implementation of the specification's formulas. Every
comparison is relative and is measured against the plan's own tolerance
tau = kappa n eps_m cond(G_1) with kappa = 10, instead of the fixed absolute
1e-13 the tests used before; every instance writes its record.

The rational check pins the floating-point values themselves: the generator's
squared lengths are dyadic rationals, so the C++ instance and the exact
arithmetic see bit-identical inputs, and the Grassmann metric -- a polynomial
in s, with no root and no branch -- has an exact value to compare against.
"""
import math
import time
from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import eig

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge import _svp as svp
from tests.chainhodge._fixtures import flat_torus, random_allowable, torus33

KS = ch.Branch.KontsevichSegal
L2, ALL = ch.Preset.L2, ch.Preset.GRASSMANN_ALL
FACTOR = 100.0                      # the plan's 1e2 tau


def relative(a, b):
    """Relative agreement of two matrices in the Frobenius norm."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.linalg.norm(b), np.finfo(float).tiny)
    return float(np.linalg.norm(a - b) / scale)


def hausdorff(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    d = np.abs(a[:, None] - b[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def instances():
    """The identical inputs the oracle and the library are run on: the
    specification's 3x3 tori in both signatures, a jittered flat torus in both
    signatures, and a complex allowable 2-complex."""
    rng = np.random.default_rng(23)
    K, sL = torus33()
    _, sE = torus33(1.0, 1.0, 1.0)
    K6, s6, _ = flat_torus(6, 0.25, False, seed=1)
    K6L, s6L, _ = flat_torus(6, 0.25, True, seed=1)
    K2 = cob.ChainComplex.fromTopCells([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [2, 3, 4]])
    return {"T6 Lorentzian 3x3": (K, sL, "lorentzian"),
            "T7 Euclidean 3x3": (K, sE, "euclidean"),
            "F1 Euclidean N=6": (K6, s6, "euclidean"),
            "F1 Lorentzian N=6": (K6L, s6L, "lorentzian"),
            "complex allowable 2-complex": (K2, random_allowable(K2, rng, 0.3), "complex")}


@pytest.mark.parametrize("name", list(instances()))
@pytest.mark.parametrize("preset", ["L2", "GRASSMANN_ALL"])
def test_metrics_agree_with_the_dense_oracle(name, preset, svp_records):
    """X, the metrics: every M_k (Whitney) or G_k (Grassmann) against the dense
    oracle, relative agreement <= 1e2 tau."""
    K, s, signature = instances()[name]
    started = time.time()
    hodge = ch.ChainHodge(K, s, L2 if preset == "L2" else ALL, KS)
    agreement = {}
    for k in range(K.dimension() + 1):
        if preset == "L2":
            code = ch.WhitneyMass.assemble(K, s, k, KS).toarray()
            oracle = svp.whitney_reference(K, s, k)
        else:
            code = ch.WhitneyMass.assembleGrassmann(K, s, k).toarray()
            oracle = svp.grassmann_reference(K, s, k)
        agreement[f"M_{k}" if preset == "L2" else f"G_{k}"] = relative(code, oracle)
    for k in range(1, K.dimension() + 1):
        # the reference orientation itself: the oracle builds d_k from the cell
        # lists ((-1)^i on the face that drops the i-th vertex).
        agreement[f"d_{k}"] = relative(hodge.boundary(k).toarray(), svp.boundary_matrix(K, k))
    record = svp_records.instance(
        "X", name, {"instance": name, "signature": signature}, hodge,
        started=started, oracle_agreement=agreement,
        criterion=f"relative agreement <= {FACTOR:g} tau")
    assert max(record["oracle_agreement"].values()) <= FACTOR * record["tau"]


@pytest.mark.parametrize("name", list(instances()))
@pytest.mark.parametrize("preset", ["L2", "GRASSMANN_ALL"])
def test_pencil_spectrum_and_harmonic_space_agree_with_the_dense_oracle(name, preset, svp_records):
    """X, the readouts: the pencil (A, B), its spectrum and the harmonic span,
    all at 1e2 tau. The oracle solves the same dense generalized eigenproblem
    with scipy and takes the harmonic space from the singular value
    decomposition of the stacked matrix it builds itself."""
    K, s, signature = instances()[name]
    started = time.time()
    hodge = ch.ChainHodge(K, s, L2 if preset == "L2" else ALL, KS)
    read = hodge.harmonicChains(1)
    pencil = hodge.pencil(1)
    if preset == "L2":
        A, B = svp.whitney_pencil_reference(K, s, 1)
        stacked = np.vstack([svp.boundary_matrix(K, 2).T, svp.boundary_matrix(K, 1) @ B])
    else:
        A, B = svp.grassmann_pencil_reference(K, s, 1)
        stacked = np.vstack([svp.boundary_matrix(K, 1), svp.boundary_matrix(K, 2).T @ B])
    _, sv, vh = np.linalg.svd(stacked)
    nullity = int(np.sum(sv <= 1e-10 * sv[0]) + max(0, stacked.shape[1] - sv.size))
    kernel = vh[vh.shape[0] - nullity:].conj().T
    values = eig(A, B, right=False)
    spectrum = hodge.spectrum(1)
    scale = float(np.max(np.abs(np.array(spectrum.eigenvalues))))
    agreement = {
        "A": relative(pencil.A, A), "B": relative(pencil.B, B),
        "spectrum": hausdorff(spectrum.eigenvalues, values) / scale,
        "harmonic_span_rad": float(np.max(np.radians(
            svp.angles_deg(read.images if preset == "L2" else read.chains, kernel)))),
    }
    record = svp_records.instance(
        "X", name, {"instance": name, "signature": signature}, hodge, read, started=started,
        oracle_agreement=agreement, nullity_oracle=nullity,
        criterion=f"relative agreement <= {FACTOR:g} tau")
    assert record["nullity"] == record["nullity_oracle"] == record["betti"][1]
    assert record["oracle_agreement"]["A"] <= FACTOR * record["tau"]
    assert record["oracle_agreement"]["B"] <= FACTOR * record["tau"]
    assert record["oracle_agreement"]["harmonic_span_rad"] <= FACTOR * record["tau"]
    assert record["oracle_agreement"]["spectrum"] <= FACTOR * record["tau"]


@pytest.mark.parametrize("lorentz", [False, True])
@pytest.mark.parametrize("N", [4, 6])
def test_grassmann_is_exact_on_rational_squared_lengths(N, lorentz, svp_records):
    """X, the rational check: on a lattice whose vertices are displaced by
    multiples of 1/8, every s_e is a dyadic rational and its binary64 value is
    exact, so the library and exact arithmetic run on identical inputs. The
    Grassmann metric is polynomial in s, so it has an exact rational value;
    the library must reproduce it to rounding, not to a fixed tolerance."""
    started = time.time()
    K, s, exact = svp.dyadic_torus(N, lorentz)
    assert all(Fraction(float(v)) == v for v in exact), "s_e is not exact in binary64"
    hodge = ch.ChainHodge(K, s, ALL, KS)
    agreement = {}
    for k in range(3):
        code = ch.WhitneyMass.assembleGrassmann(K, s, k).toarray()
        rational = svp.rational_grassmann(K, exact, k)
        value = np.array([[float(v) for v in row] for row in rational])
        agreement[f"G_{k}"] = relative(code, value)
    record = svp_records.instance(
        "X", "F1 (dyadic)",
        {"N": N, "seed": 1, "scale": 8, "signature": "lorentzian" if lorentz else "euclidean"},
        hodge, started=started, oracle_agreement=agreement,
        criterion="relative agreement with exact rational arithmetic <= 1e-14")
    assert max(record["oracle_agreement"].values()) <= 1e-14


@pytest.mark.parametrize("lorentz", [False, True])
@pytest.mark.parametrize("N", [4, 6])
def test_the_topology_is_exact_and_the_rational_kernel_has_that_dimension(N, lorentz, svp_records):
    """X, the integer topology: the Betti numbers from the exact integer ranks
    of the incidence maps (never a floating-point rank), and the dimension of
    the GRASSMANN_ALL harmonic space over Q -- the kernel of the exact stacked
    matrix [d_1 ; d_2^T G_1] by rational elimination -- against the nullity the
    library reads numerically."""
    started = time.time()
    K, s, exact = svp.dyadic_torus(N, lorentz)
    hodge = ch.ChainHodge(K, s, ALL, KS)
    read = hodge.harmonicChains(1)
    betti = svp.integer_betti(K)
    G1 = svp.rational_grassmann(K, exact, 1)
    n1 = K.numSimplices(1)
    rows = [[Fraction(v) for v in row] for row in svp.integer_boundary(K, 1)]
    d2 = svp.integer_boundary(K, 2)
    for column in range(len(d2[0])):
        coefficients = [Fraction(d2[r][column]) for r in range(n1)]
        rows.append([sum(coefficients[i] * G1[i][j] for i in range(n1) if coefficients[i])
                     for j in range(n1)])
    exact_nullity = n1 - svp.rational_rank(rows)
    record = svp_records.instance(
        "X", "F1 (dyadic)",
        {"N": N, "seed": 1, "scale": 8, "signature": "lorentzian" if lorentz else "euclidean"},
        hodge, read, started=started, betti_exact=betti, nullity_exact=exact_nullity,
        criterion="exact integer Betti numbers and exact rational nullity equal "
                  "the library's reads")
    assert record["betti_exact"] == [1, 2, 1] == record["betti"]
    assert record["nullity_exact"] == record["nullity"] == 2
