# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The scaling verification plan's spectral-statistics suite S1-S4 (#1208).

  S1  The fraction of geometries with complex-conjugate eigenvalue pairs, F4
      (CDT-like causal layering) against F5 (incoherent random signs), against
      N, under both chain metrics. The plan's evidence is that real spectra
      persist for causally layered geometries under both metrics (0/40 at
      N = 3, 5, 7) while the random-sign ensemble produces complex pairs
      (143/200 under GRASSMANN_ALL; "to run" under L2, which this settles).
  S2  The signature of G_1 and of the harmonic Gram Phi^T G_1 Phi on
      Lorentzian tori: expected (1, 1) for b_1 = 2.
  S3  The distribution of the exceptional-point indicator over the spectrum.
  S4  The scaling of the first nonzero |lambda| with N.

Every instance writes the plan's JSON record and the verdicts are derived from
those records.
"""
import math
import time

import numpy as np
import pytest

from tessera import chainhodge as ch
from tests.chainhodge import _svp as svp
from tests.chainhodge._fixtures import flat_torus, torus33

KS = ch.Branch.KontsevichSegal
L2, ALL = ch.Preset.L2, ch.Preset.GRASSMANN_ALL
PRESETS = {"L2": L2, "GRASSMANN_ALL": ALL}


def ensemble(generator, N, draws, preset, seed):
    """One S1 ensemble: `draws` geometries of a generator at size N, counting
    those whose degree-1 spectrum leaves the real axis and those whose non-real
    eigenvalues close under conjugation. A draw whose metric is too degenerate
    to factor is counted separately and excluded, never silently."""
    rng = np.random.default_rng(seed)
    counts = {"draws": draws, "non_real": 0, "conjugate_pairs": 0, "degenerate": 0}
    for _ in range(draws):
        K, s = generator(N, rng)
        try:
            spectrum = ch.ChainHodge(K, s, preset, KS).spectrum(1)
        except Exception:                       # a singular M_0 or M_1: recorded
            counts["degenerate"] += 1
            continue
        summary = svp.spectrum_summary(spectrum.eigenvalues)
        counts["non_real"] += bool(summary["complex"])
        counts["conjugate_pairs"] += bool(summary["conjugate_pairs"])
    return counts


class TestS1ComplexConjugatePairs:
    """S1. The plan's table: F4 at N = 3, 5, 7 with 40 draws each under both
    metrics, and F5 at N = 3 with 200 draws."""

    @pytest.mark.parametrize("preset", list(PRESETS))
    @pytest.mark.parametrize("N", [3, 5, 7])
    def test_causal_layering_keeps_the_spectrum_real(self, N, preset, svp_records):
        counts = ensemble(svp.cdt_like_torus, N, 40, PRESETS[preset], seed=101 + N)
        record = svp_records.write(
            test="S1", family="F4", preset=preset,
            params={"N": N, "draws": 40, "seed": 101 + N, "signature": "lorentzian"},
            spectrum_summary=counts,
            criterion="no geometry has a non-real eigenvalue (the plan's 0/40)")
        assert record["spectrum_summary"]["degenerate"] == 0
        assert record["spectrum_summary"]["non_real"] == 0

    @pytest.mark.parametrize("preset", list(PRESETS))
    def test_random_signs_are_reported(self, preset, svp_records):
        """F5 with 200 draws at N = 3, the plan's row. Under GRASSMANN_ALL the
        metric is real, so the pencil is real and its non-real eigenvalues come
        in conjugate pairs -- the plan's 143/200, measured 151/200 here with a
        different seed. Under L2 the row was never run: the Whitney metric of a
        random-sign mesh mixes top simplices of both causal types, so M_k is
        not a real matrix times a global phase, the pencil is genuinely
        complex, and (measured) 194/200 draws leave the real axis with none of
        them conjugate-paired. That answers the plan's open item 3: the two
        ensembles are not comparable rates of the same phenomenon."""
        counts = ensemble(svp.random_sign_torus, 3, 200, PRESETS[preset], seed=303)
        record = svp_records.write(
            test="S1", family="F5", preset=preset,
            params={"N": 3, "draws": 200, "seed": 303, "signature": "random signs"},
            spectrum_summary=counts, criterion="reported (no threshold)")
        counts = record["spectrum_summary"]
        assert counts["degenerate"] == 0
        assert counts["non_real"] > 100
        if preset == "GRASSMANN_ALL":
            assert counts["conjugate_pairs"] == counts["non_real"]
        else:
            assert counts["conjugate_pairs"] == 0

    @pytest.mark.parametrize("preset", list(PRESETS))
    @pytest.mark.parametrize("N", [3, 5, 7])
    def test_random_signs_against_n(self, N, preset, svp_records):
        """The plan asks for the fraction against N for both families; only
        F4 has published rows at N = 5 and 7."""
        counts = ensemble(svp.random_sign_torus, N, 40, PRESETS[preset], seed=404 + N)
        record = svp_records.write(
            test="S1", family="F5", preset=preset,
            params={"N": N, "draws": 40, "seed": 404 + N, "signature": "random signs"},
            spectrum_summary=counts, criterion="reported (no threshold)")
        assert record["spectrum_summary"]["non_real"] >= 1


class TestS2Signature:
    """S2: the signature of G_1 and of the harmonic Gram on Lorentzian tori.
    Both are complex symmetric, and on real Lorentzian data each is a global
    phase (the branch factor i of the integration specification section 4.2,
    or 1 for the real Grassmann metric) times a real symmetric matrix; the
    signature is read after dividing that phase out, on a real basis of the
    harmonic space."""

    @pytest.mark.parametrize("preset", list(PRESETS))
    @pytest.mark.parametrize("case", ["torus33", 6, 8, 10])
    def test_harmonic_gram_has_signature_one_one(self, case, preset, svp_records):
        started = time.time()
        if case == "torus33":
            K, s = torus33()
            params = {"N": 3, "fixture": "torus33", "signature": "lorentzian"}
        else:
            K, s, _ = flat_torus(case, 0.25, True, seed=1)
            params = {"N": case, "jitter": 0.25, "seed": 1, "signature": "lorentzian"}
        hodge = ch.ChainHodge(K, s, PRESETS[preset], KS)
        read = hodge.harmonicChains(1)
        phase = 1j if preset == "L2" else 1.0
        metric = (hodge.Minv(1) if preset == "L2" else hodge.chainMetricSparse(1)).toarray()
        # G_1 = M_1^{-1} under L2, so the inertia of M_1/i is that of G_1 i.
        chains = svp.real_basis(read.chains)
        images = svp.real_basis(read.images)
        gram = images.T @ metric @ images if preset == "L2" else chains.T @ metric @ chains
        record = svp_records.instance(
            "S2", "F1" if case != "torus33" else "T6", params, hodge, read, started=started,
            signature_G1=list(svp.signature(metric, phase)),
            signature_harmonic_gram=list(svp.signature(gram, phase)),
            harmonic_gram_determinant=complex(np.linalg.det(gram) / phase ** 2),
            criterion="the harmonic Gram has signature (1, 1) for b_1 = 2")
        assert record["nullity"] == record["betti"][1] == 2
        assert record["signature_harmonic_gram"] == [1, 0, 1]
        # G_1 itself is indefinite on a Lorentzian mesh: reported, not fixed.
        assert sum(record["signature_G1"]) == record["n1"]
        assert record["signature_G1"][1] == 0

    def test_euclidean_control_is_definite(self, svp_records):
        """The control the plan's expectation rests on: on a Euclidean torus
        the same reading gives a definite Gram, signature (2, 0)."""
        K, s, _ = flat_torus(6, 0.25, False, seed=1)
        hodge = ch.ChainHodge(K, s, L2, KS)
        read = hodge.harmonicChains(1)
        images = svp.real_basis(read.images)
        gram = images.T @ hodge.Minv(1).toarray() @ images
        record = svp_records.instance(
            "S2", "F1", {"N": 6, "jitter": 0.25, "seed": 1, "signature": "euclidean"},
            hodge, read, signature_harmonic_gram=list(svp.signature(gram, 1.0)),
            criterion="signature (2, 0) on Euclidean data")
        assert record["signature_harmonic_gram"] == [2, 0, 0]


class TestS3ExceptionalPointIndicator:
    """S3: the distribution of the exceptional-point indicator over the
    spectrum. For a cluster of eigenvalues with orthonormal right eigenvectors
    Q the indicator is sigma_min(Q^T B Q)/||B||, the normalized complex
    bilinear restriction B_C of the integration specification (section 5): it
    vanishes exactly at an isotropic band, where the pencil is defective."""

    @pytest.mark.parametrize("lorentz", [False, True])
    @pytest.mark.parametrize("N", [4, 6, 8])
    def test_indicator_distribution_is_reported(self, N, lorentz, svp_records):
        started = time.time()
        K, s, _ = flat_torus(N, 0.25, lorentz, seed=1)
        hodge = ch.ChainHodge(K, s, L2, KS)
        read = hodge.harmonicChains(1)
        spectrum = hodge.spectrum(1)
        pencil = hodge.pencil(1)
        indicators = svp.ep_indicators(spectrum, pencil.B)
        quantiles = np.quantile(indicators, [0.0, 0.01, 0.1, 0.5, 0.9, 1.0])
        record = svp_records.instance(
            "S3", "F1", {"N": N, "jitter": 0.25, "seed": 1,
                         "signature": "lorentzian" if lorentz else "euclidean"},
            hodge, read, started=started,
            spectrum_summary=svp.spectrum_summary(spectrum.eigenvalues),
            ep_indicator_min=float(min(indicators)),
            ep_indicator_quantiles={str(q): float(v) for q, v in
                                    zip([0.0, 0.01, 0.1, 0.5, 0.9, 1.0], quantiles)},
            residuals={"pencil": spectrum.residual},
            criterion="reported; positive (no isotropic band on these meshes)")
        assert len(indicators) == record["n1"]
        assert record["ep_indicator_min"] > 0.0
        assert record["residuals"]["pencil"] < 1e-10

    def test_the_indicator_collapses_on_lorentzian_meshes(self, svp_records):
        """Reported beside the distribution: under refinement the smallest
        indicator of a Lorentzian torus falls by a decade per doubling while
        the Euclidean one stays at O(0.1). It is the same approach to the
        rank-condition failure set that the plan's Finding 3 describes."""
        rows = {}
        for lorentz in (False, True):
            rows["lorentzian" if lorentz else "euclidean"] = []
            for N in (4, 6, 8):
                K, s, _ = flat_torus(N, 0.25, lorentz, seed=1)
                hodge = ch.ChainHodge(K, s, L2, KS)
                indicators = svp.ep_indicators(hodge.spectrum(1), hodge.pencil(1).B)
                rows["lorentzian" if lorentz else "euclidean"].append(float(min(indicators)))
        svp_records.write(test="S3", family="F1", preset="L2",
                          params={"N": [4, 6, 8], "jitter": 0.25, "seed": 1},
                          ep_indicator_min=rows, criterion="reported (no threshold)")
        assert min(rows["euclidean"]) > 0.05
        assert rows["lorentzian"][0] > 5.0 * rows["lorentzian"][-1]


class TestS4FirstNonzeroEigenvalue:
    """S4: the scaling of the first nonzero |lambda| with N. The plan sets no
    threshold; the record carries the sequence and its fitted slope. On the
    Euclidean unit torus the first nonzero eigenvalue of the Hodge Laplacian on
    1-forms is the continuum (2 pi)^2 = 39.478, which the sequence approaches
    from below; on the Lorentzian torus, where the operator is not elliptic, it
    collapses towards zero instead."""

    SIZES = (4, 6, 8, 10)
    CONTINUUM = 4.0 * math.pi ** 2

    def sequence(self, lorentz, svp_records):
        values = []
        for N in self.SIZES:
            started = time.time()
            K, s, _ = flat_torus(N, 0.25, lorentz, seed=1)
            hodge = ch.ChainHodge(K, s, L2, KS)
            read = hodge.harmonicChains(1)
            spectrum = hodge.spectrum(1)
            ev = np.array(spectrum.eigenvalues, dtype=complex)
            summary = svp.spectrum_summary(ev)
            # The b_1 harmonic eigenvalues are the b_1 smallest moduli; the
            # first nonzero one is the next, with no threshold in between (on
            # a Lorentzian torus it falls towards the rounding level itself).
            moduli = np.sort(np.abs(ev))
            record = svp_records.instance(
                "S4", "F1", {"N": N, "jitter": 0.25, "seed": 1,
                             "signature": "lorentzian" if lorentz else "euclidean"},
                hodge, read, started=started, spectrum_summary=summary,
                harmonic_abs=[float(v) for v in moduli[:read.nullity]],
                first_nonzero_abs=float(moduli[read.nullity]),
                continuum=self.CONTINUUM if not lorentz else None,
                criterion="reported (no threshold)")
            assert record["nullity"] == 2
            assert max(record["harmonic_abs"]) < 1e-6 * record["first_nonzero_abs"]
            values.append(record["first_nonzero_abs"])
        slope = float(np.polyfit(np.log(self.SIZES), np.log(values), 1)[0])
        svp_records.write(test="S4", family="F1", preset="L2",
                          params={"N": list(self.SIZES), "jitter": 0.25, "seed": 1,
                                  "signature": "lorentzian" if lorentz else "euclidean"},
                          first_nonzero_abs=values, slope_log_log=slope,
                          criterion="reported (no threshold)")
        return values, slope

    def test_euclidean_approaches_the_continuum_value(self, svp_records):
        values, slope = self.sequence(False, svp_records)
        errors = [abs(v - self.CONTINUUM) for v in values]
        assert all(a > b for a, b in zip(errors, errors[1:]))
        assert abs(slope) < 0.2                      # N-independent, as the continuum is

    def test_lorentzian_collapses(self, svp_records):
        values, slope = self.sequence(True, svp_records)
        assert all(a > b for a, b in zip(values, values[1:]))
        assert slope < -1.0
