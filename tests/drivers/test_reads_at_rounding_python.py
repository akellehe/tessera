"""The spin, band and with-quartic reads of the pole driver at the declared
tolerances: each read takes its multiplicities from the exact structure it
has (the two-eigenvalue polynomial of J^2 on three spins 1/2, the pairs of a
nontrivial projective class, the band rule's own read) and not from a
comparison of computed eigenvalues at the tolerance, and a with-quartic read
outside the range of its expansion is flagged beside the ratio built from
it."""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp


HALF, THREE = bp.SPIN_HALF, bp.SPIN_THREE_HALVES


# ------------------------------------------------------------ spin sectors


def _spin_dimensions(content):
    """The numbers of colour-singlet states of total spin 1/2 and 3/2 of a
    carrier content: the quarks of one carrier are symmetric in spin, so n
    quarks of a carrier carry spin n / 2, and the carriers' spins add."""
    spins = [n / 2.0 for n in content if n]
    totals = [0.0]
    for s in spins:
        totals = [j for total in totals
                  for j in np.arange(abs(total - s), total + s + 0.5, 1.0)]
    return (2 * sum(1 for j in totals if j == 0.5),
            4 * sum(1 for j in totals if j == 1.5))


@pytest.mark.parametrize("content", bp.contents())
def test_every_content_splits_into_its_full_spin_sectors(content):
    """The spin split of every carrier content at the declared spin-sector
    tolerance: each sector has the dimension the addition of the carriers'
    spins gives it, its basis is orthonormal as Fock vectors and is carried
    by J^2 to j(j + 1) times itself, and the measurements of the read are at
    rounding."""
    half, three = _spin_dimensions(content)
    states, _ = bp.singlet_states(content)
    sectors, read = bp.spin_sectors(states)
    assert half + three == states.shape[1]
    assert read["dimensions"] == {HALF: half, THREE: three}
    assert sorted(sectors) == [j2 for j2, n in ((HALF, half), (THREE, three))
                               if n]
    basis = bp.occupation_basis()
    spins = bp.edge_spin_matrices()
    for j2, sector in sectors.items():
        assert sector.shape[1] == read["dimensions"][j2]
        focks = np.column_stack([bp.to_fock(sector[:, k], basis)
                                 for k in range(sector.shape[1])])
        assert np.max(np.abs(focks.conj().T @ focks
                             - np.eye(sector.shape[1]))) < 1e-12
        for k in range(sector.shape[1]):
            image = np.asarray(obs.SharpSpin.applyTotalSpinSquared(
                spins, focks[:, k]))
            assert np.max(np.abs(image - j2 * focks[:, k])) < 1e-12
    assert read["tolerance"] == 1e-15
    assert max(read["trace_departure"].values()) < 1e-12
    assert read["polynomial_residual"] < 1e-12
    assert read["leakage"] < 1e-12
    assert len(read["eigenvalues"]) == states.shape[1]
    assert all(min(abs(v - HALF), abs(v - THREE)) < 1e-12
               for v in read["eigenvalues"])


def test_the_spin_split_does_not_depend_on_the_basis_of_the_states():
    """The sectors are subspaces: with the states of (2, 1, 0) replaced by
    an invertible real mixing of themselves, and by a complex one, each
    sector has the same dimension and the same span."""
    states, _ = bp.singlet_states((2, 1, 0))
    reference, _ = bp.spin_sectors(states)
    rng = np.random.default_rng(4)
    count = states.shape[1]
    for mixing in (np.eye(count) + 0.4 * rng.standard_normal((count, count)),
                   np.eye(count) + 0.4 * rng.standard_normal((count, count))
                   + 0.3j * rng.standard_normal((count, count))):
        sectors, read = bp.spin_sectors(states @ mixing)
        assert read["dimensions"] == {HALF: 2, THREE: 4}
        for j2, sector in sectors.items():
            both = np.column_stack([reference[j2], sector])
            assert np.linalg.matrix_rank(both, tol=1e-10) == \
                reference[j2].shape[1]


def test_the_doublet_sectors_carry_the_measurements_of_their_split():
    """`doublet_sectors` and `doublet_sector_read` read one split: the
    carrier content a doublet content names in the frame's carrier order,
    the sectors, and the measurements beside them. Trialities that are not
    the three Z_3 characters give a doublet content no carrier content, a
    `ValueError` by name."""
    trialities = [1, 0, 2]
    carrier_content, triality, sectors = bp.doublet_sectors([2, 1, 0],
                                                            trialities)
    # two quarks in the doublet 2 (carrier 1), one in 2' (carrier 0)
    assert carrier_content == [1, 2, 0] and triality == 1
    read = bp.doublet_sector_read([2, 1, 0], trialities)
    assert read["dimensions"] == {HALF: 2, THREE: 4}
    assert {j2: sector.shape[1] for j2, sector in sectors.items()} == \
        read["dimensions"]
    with pytest.raises(bp.DoubletLabelsWithoutValue,
                       match="are not the three Z_3 characters"):
        bp.doublet_sectors([2, 1, 0], [0, 0, 1])
    assert issubclass(bp.DoubletLabelsWithoutValue, ValueError)


# -------------------------------------------------------- the aligned frame


@pytest.mark.parametrize("degeneracy_tolerance",
                         [1e-17, 1e-15, 1e-12, 1e-9])
def test_the_reference_carrier_holds_the_coexact_doublet(
        degeneracy_tolerance):
    """The reference carrier of the declared host's aligned frame is the
    carrier that holds the coexact j = 1/2 doublet, the pair at 4, whatever
    degeneracy tolerance the bands are grouped at: its columns of the frame
    lie in the coexact sector, and the three carriers carry the trialities
    (1, 0, 2)."""
    host = bp.monopole_support()
    alignment = bp.aligned_doublet_frame(host, bp.rotation_group(),
                                         degeneracy_tolerance)
    reference = alignment["reference_carrier"]
    assert reference == 1 and alignment["reference_certified"]
    assert alignment["trialities"] == [1, 0, 2]
    columns = alignment["frame"][:, 2 * reference:2 * reference + 2]
    coexact = np.asarray(host.coexactProjector(1e-15))
    assert np.max(np.abs(coexact @ columns - columns)) < 1e-12
    for carrier in (0, 2):
        other = alignment["frame"][:, 2 * carrier:2 * carrier + 2]
        assert np.max(np.abs(coexact @ other - other)) > 0.1
    assert alignment["reference_doublet"]["eigenvalue"] == \
        pytest.approx(4.0, abs=1e-12)
    assert alignment["intertwining_residual"] < 1e-12


@pytest.mark.parametrize("degeneracy_tolerance",
                         [1e-17, 1e-15, 1e-12, 1e-9])
def test_the_bands_of_the_nontrivial_class_are_pairs(degeneracy_tolerance):
    """Under the nontrivial projective class the eigenvalues of the
    rotation-averaged edge Laplacian are equal in pairs exactly, so the
    declared host has three bands of rank two at every degeneracy tolerance
    below their separation, each a certified spinor doublet, and the middle
    one alone coexact."""
    host = bp.monopole_support()
    bands = bp.symmetry_bands(host, bp.rotation_group(), True,
                              degeneracy_tolerance)
    assert [band["dimension"] for band in bands] == [2, 2, 2]
    assert [band["start"] for band in bands] == [0, 2, 4]
    np.testing.assert_allclose(
        [band["eigenvalue"] for band in bands],
        [4 - 2 / np.sqrt(3), 4.0, 4 + 2 / np.sqrt(3)], atol=1e-12)
    assert [band["coexact"] for band in bands] == [False, True, False]
    assert all(abs(band["irreducibility_score"] - 1.0) < 1e-12
               and band["invariance_residual"] < 1e-12 for band in bands)
    assert bp.reference_doublet(bands) is bands[1]


def test_the_bands_of_the_trivial_class_are_single_eigenvalues():
    """At monopole number 2 the projective class is trivial: a unit of the
    grouping is one eigenvalue, no band is a spinor doublet, and the frame
    has no certified reference."""
    support = obs.MonopoleSupport.tetrahedron(2)
    group = obs.MonopoleSupport.tetrahedralRotations()
    read = support.spinRead(group)
    assert not read.cocycle.nontrivial
    bands = bp.symmetry_bands(support, group, False, 1e-9)
    assert sum(band["dimension"] for band in bands) == 6
    assert not any(band["spinor_doublet"] for band in bands)
    assert bp.reference_doublet(bands) is None
    with pytest.raises(bp.NoSpinorDoublet, match="no j = 1/2 doublet"):
        bp.aligned_doublet_frame(support, group)


def test_an_uncertified_reference_is_read_and_reported():
    """At a certificate tolerance below the rounding of the certificates
    (1e-17) no carrier of the declared host is a certified doublet. The
    certified read says so by name; the uncertified read takes the carrier
    of least coexact residual, which is the same carrier 1, and reports its
    certificates."""
    host, group = bp.monopole_support(), bp.rotation_group()
    with pytest.raises(bp.NoSpinorDoublet, match="coexact residuals"):
        bp.aligned_doublet_frame(host, group, 1e-15, 1e-17)
    alignment = bp.aligned_doublet_frame(host, group, 1e-15, 1e-17,
                                         certified=False)
    assert alignment["reference_carrier"] == 1
    assert alignment["reference_certified"] is False
    assert alignment["trialities"] == [1, 0, 2]
    doublet = alignment["reference_doublet"]
    assert doublet["dimension"] == 2
    assert 0.0 < doublet["coexact_residual"] < 1e-12
    assert not (doublet["spinor_doublet"] and doublet["coexact"])


def test_the_host_frame_is_read_with_its_reference_uncertified():
    """`spin_frame` at a certificate tolerance of 1e-17 on the declared
    host: the cell does not have the twelve rotations at that tolerance and
    no carrier of the host is certified, so the frame is the host's, read
    with the carrier of least coexact residual, and both facts are flagged
    with their numbers. The frame is the one read at the declared
    tolerances."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports, 1e-17)
    assert not symmetry["tetrahedral"]
    frame = bp.spin_frame(supports, symmetry, 1e-15, 1e-17, 1e-15)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    assert [flag["name"] for flag in frame["flags"]] == [
        "not tetrahedrally symmetric",
        "the host's reference doublet is not certified"]
    flag = frame["flags"][1]
    assert flag["reference_carrier"] == 1 and flag["tolerance"] == 1e-17
    assert 0.0 < flag["coexact_residual"] < 1e-12
    assert "carrier 1, the carrier of least coexact residual" in \
        flag["detail"]
    declared = bp.aligned_doublet_frame(bp.monopole_support(),
                                        bp.rotation_group())
    for sheet in frame["alignments"]:
        assert sheet["trialities"] == declared["trialities"] == [1, 0, 2]
        assert np.max(np.abs(sheet["frame"] - declared["frame"])) < 1e-12


def test_labels_that_are_not_the_three_characters_send_the_read_to_the_host(
        monkeypatch):
    """The fourth precondition of the cell's own spin read: the sheets'
    common labels are the three Z_3 characters. With the read of the cell's
    own supports replaced by one whose trialities are (0, 0, 1) on every
    sheet, the frame is the declared symmetric host's and the one flag
    names the labels."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t, 1e-12)
                for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports, 1e-12)
    assert symmetry["tetrahedral"]
    own = [support for support, _ in supports]
    real = bp.aligned_doublet_frame

    def read(support, group, degeneracy_tolerance, tolerance,
             character_tolerance, certified=True):
        out = real(support, group, degeneracy_tolerance, tolerance,
                   character_tolerance, certified)
        if any(support is sheet for sheet in own):
            out = dict(out, trialities=[0, 0, 1])
        return out
    monkeypatch.setattr(bp, "aligned_doublet_frame", read)
    frame = bp.spin_frame(supports, symmetry, 1e-12, 1e-12, 1e-12)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    (flag,) = frame["flags"]
    assert flag["name"] == \
        "the doublet labels are not the three Z_3 characters"
    assert flag["trialities"] == [0, 0, 1]
    assert [sheet["trialities"] for sheet in frame["alignments"]] == \
        [[1, 0, 2]] * 3


@pytest.mark.parametrize("character_tolerance", [1e-17, 1e-15, 1e-3])
def test_the_frame_does_not_depend_on_the_character_tolerance(
        character_tolerance):
    """The character of a doublet on a half turn is one by the half turn's
    cycle type. The trace of a half turn on the reference doublet is zero
    exactly and is computed at rounding; it is reported with whether it
    holds at the character tolerance, and the frame is the same whatever
    that tolerance is, below the rounding of the trace or far above it."""
    host, group = bp.monopole_support(), bp.rotation_group()
    declared = bp.aligned_doublet_frame(host, group)
    alignment = bp.aligned_doublet_frame(host, group, 1e-15, 1e-15,
                                         character_tolerance)
    assert np.array_equal(alignment["frame"], declared["frame"])
    assert alignment["trialities"] == [1, 0, 2]
    assert alignment["intertwining_residual"] < 1e-12
    assert alignment["half_turn_trace_residual"] < 1e-12
    assert alignment["half_turn_trace_held"] == (
        alignment["half_turn_trace_residual"] <= character_tolerance)


def test_the_distance_of_two_spectra_does_not_depend_on_their_order():
    """Two spectra that are one multiset, with two eigenvalues whose real
    parts agree to rounding and whose imaginary parts differ by two: a sort
    by real part lists that pair in either order, and the multiset distance
    pairs each eigenvalue with its own image. A spectrum with one
    eigenvalue moved is at that eigenvalue's displacement."""
    first = [1.0 + 1.0j, 1.0 + 2e-16 - 1.0j, 3.0, 5.0 + 0.5j]
    second = [1.0 + 2e-16 + 1.0j, 1.0 - 1.0j, 5.0 + 0.5j, 3.0]
    def by_real(values):
        return sorted(values, key=lambda v: (v.real, v.imag))
    assert max(abs(a - b) for a, b in zip(by_real(first),
                                          by_real(second))) > 1.9
    assert bp.multiset_distance(first, second) < 1e-15
    assert bp.multiset_distance(second, first) < 1e-15
    moved = list(second)
    moved[2] = 5.0 + 0.75j
    assert bp.multiset_distance(first, moved) == pytest.approx(0.25)
    assert bp.multiset_distance([], []) == 0.0
    with pytest.raises(ValueError, match="are not one multiset"):
        bp.multiset_distance([1.0], [1.0, 2.0])


# ------------------------------------------------------------- the bands


def _band_rule(tolerance):
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [1.0]
    declaration.band_tolerance = tolerance
    declaration.occupation_order = cob.OccupationOrder.AscendingRealPart
    return cob.BandFollower(declaration)


def _non_normal(eigenvalues, seed):
    """A non-normal operator with the given eigenvalues, and its
    eigenvector matrix."""
    rng = np.random.default_rng(seed)
    n = len(eigenvalues)
    vectors = np.eye(n) + 0.3 * rng.standard_normal((n, n)) \
        + 0.2j * rng.standard_normal((n, n))
    return vectors @ np.diag(eigenvalues) @ np.linalg.inv(vectors), vectors


def test_the_band_projectors_are_the_band_rules_own_bands():
    """`band_projectors` groups as `BandFollower.read` does, on the carrier
    of the three-sheeted host at the declared band tolerance: the same
    ranks in the same order, the same eigenvalues, and Riesz projectors
    that resolve the identity, are idempotent and commute with the
    operator."""
    host = bp.build_host()
    action = cob.JointAction(host, bp.action_declaration(host, 1.0, 1.0))
    flat = action.carrier_operator()
    carrier = bp.matrix(flat)
    bands = bp.band_projectors(carrier, 1e-15)
    read = _band_rule(1e-15).read(flat)
    assert [len(values) for values, _ in bands] == list(read.ranks)
    assert sum(len(values) for values, _ in bands) == 18
    np.testing.assert_allclose(
        np.concatenate([values for values, _ in bands]),
        np.asarray(read.ordered), atol=1e-12)
    scale = np.linalg.norm(carrier)
    assert np.max(np.abs(sum(p for _, p in bands) - np.eye(18))) < 1e-11
    for values, projector in bands:
        assert np.max(np.abs(projector @ projector - projector)) < 1e-11
        assert np.max(np.abs(projector @ carrier - carrier @ projector)) \
            < 1e-11 * scale
        assert abs(np.trace(projector) - len(values)) < 1e-11


def test_the_band_projectors_of_a_non_normal_operator():
    """A non-normal operator with the eigenvalues 1 (twice), 3 and 4 + i at
    a band tolerance of 1e-9: three bands of ranks 2, 1, 1 in ascending
    order of real part, each projector the Riesz projector of its
    eigenvectors."""
    operator, vectors = _non_normal([1.0, 3.0, 1.0, 4.0 + 1.0j], seed=2)
    inverse = np.linalg.inv(vectors)
    bands = bp.band_projectors(operator, 1e-9)
    assert [len(values) for values, _ in bands] == [2, 1, 1]
    for (values, projector), columns in zip(bands, ([0, 2], [1], [3])):
        expected = vectors[:, columns] @ inverse[columns, :]
        assert np.max(np.abs(projector - expected)) < 1e-9
    np.testing.assert_allclose([values[0] for values, _ in bands],
                               [1.0, 3.0, 4.0 + 1.0j], atol=1e-9)


def test_the_occupied_bands_are_read_off_the_state():
    """A content names the bands it fills by their index in the order they
    were chosen at the host; a band followed by continuation keeps that
    index while its place among the bands of the point changes when it
    crosses another. With the three quarks of the content (0, 0, 3) in the
    band at place 1 of the operator read (the band a crossing brought
    there), the spin decomposition reports that band: its place, the three
    quarks measured in it, its rank and its eigenvalues, and no occupation
    elsewhere."""
    energies = [0.4, 0.4, 0.4, 2.9, 2.9, 2.9, 3.6, 3.6, 3.6]
    energies += [5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0]
    operator, vectors = _non_normal(energies, seed=8)
    inverse = np.linalg.inv(vectors)
    covariance = vectors[:, 3:6] @ inverse[3:6, :]
    identity = np.eye(18, dtype=complex)
    read = bp.spin_decomposition(operator, covariance, (0, 0, 3), identity,
                                 identity, [1, 0, 2], 1e-9)
    assert read["band_ranks"] == [3] * 6
    np.testing.assert_allclose(read["band_occupations"],
                               [0, 3, 0, 0, 0, 0], atol=1e-9)
    (band,) = read["occupied_bands"]
    assert band["band"] == 1 and band["rank"] == 3
    assert band["occupation"] == pytest.approx(3.0, abs=1e-9)
    np.testing.assert_allclose(band["eigenvalues"], [2.9] * 3, atol=1e-9)
    assert sum(band["overlap"].values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(read["occupied_state"].values()) == \
        pytest.approx(3.0, abs=1e-9)
    assert read["commutator"] < 1e-9


# -------------------------------------------------------- the fingerprint


def test_the_fingerprint_holds_above_the_certificates_rounding():
    """Quark condition 7 on the declared host with the certificates graded
    at 1e-12, above their rounding (3.7e-14 on the refined support), and the
    degeneracy and isotypic tolerances at the declared 1e-15: the doublet is
    found on the host, the relabeled host and the refined support, whose
    bands are pairs at every degeneracy tolerance. The relabeling holds. On
    the refined cell the isotypic component of the doublet's type has
    dimension two (the projector's trace, an integer to rounding), it is
    one band of rank two, and it overlaps the original doublet by one on
    the shared edges, so the refinement holds; its energy shift is
    reported."""
    config = bp.default_config([1.0], [1.0],
                               tolerances={"certificate_tolerance": 1e-12})
    read = bp.spectral_fingerprint_read(bp.build_host(), 1.0, 1.0, config)
    assert read["doublet_found"]
    relabeling = read["relabeling"]
    assert relabeling["doublet_found"] and relabeling["held"]
    assert [band["dimension"] for band in relabeling["band_certificates"]] \
        == [2, 2, 2]
    refinement = read["refinement"]
    assert refinement["doublet_found"] and refinement["held"]
    assert [band["dimension"] for band in refinement["band_certificates"]] \
        == [2] * 5
    assert [band["coexact"] for band in refinement["band_certificates"]] \
        == [False, False, False, True, False]
    assert refinement["isotypic_dimension"] == 2
    assert refinement["isotypic_trace_departure"] < 1e-12
    assert refinement["isotypic_idempotency"] < 1e-12
    assert len(refinement["refined_bands"]) == 1
    assert refinement["refined_rank"] == 2
    assert refinement["overlap"] == pytest.approx(1.0, abs=1e-12)
    assert refinement["energy_shift"] == pytest.approx(0.19763, abs=1e-4)
    assert [e.held for e in bp.fingerprint_evidence(read)] == [True, True]
    assert bp.fingerprint_text(read).startswith(
        "spectral fingerprint: relabeling stable (")


def test_a_fingerprint_without_a_doublet_says_which_band_is_nearest():
    """At a certificate tolerance below the rounding of the certificates
    (1e-17) no band of the host is the certified doublet: both reads are
    unread, with the nearest band's certificates in words."""
    config = bp.default_config([1.0], [1.0],
                               tolerances={"certificate_tolerance": 1e-17})
    read = bp.spectral_fingerprint_read(bp.build_host(), 1.0, 1.0, config,
                                        tolerance=1e-12)
    assert not read["doublet_found"]
    assert [band["dimension"] for band in read["band_certificates"]] == \
        [2, 2, 2]
    for key in ("relabeling", "refinement"):
        assert read[key]["held"] is False
        assert read[key]["unread"].startswith(
            "no spinor doublet on the host: no band is a certified coexact "
            "spinor doublet at the tolerance 1e-17: the band nearest the "
            "coexact sector (eigenvalue 4, rank 2) has coexact residual ")
    assert [e.held for e in bp.fingerprint_evidence(read)] == [False, False]


# ----------------------------------------------------- the with-quartic flag


#: The truncation read of the content (3, 0, 0) on host cell (0, 1, 3, 4) at
#: tick 0 of the run of 2026-10-01 (`v18-multicobordism-5t`), whose
#: with-quartic poles of order -5e8 give that cell's with-quartic ratio.
RECORDED_TRUNCATION = {
    "induced_displacement_norm": 16822.018532665603,
    "induced_length_norm": 0.0,
    "induced_phase_norm": 16822.018532665603,
    "action_quadratic_term": 564538096.9556797 + 0.0006060022151397657j,
    "operator_relative_remainder": 1.0000000514884309,
    "action_cubic_remainder": -564487105.3683392 - 0.0006058888656441367j,
    "action_relative_remainder": 0.9999096755602227,
}


def test_the_recorded_read_outside_its_expansion_is_flagged():
    """The recorded read's remainders are each as large as the term the
    expansion keeps, along an induced phase displacement of 16822 rad: the
    flag carries the two remainders, the displacement and the tolerance,
    and says the poles are read as they are."""
    (flag,) = bp.truncation_flags(RECORDED_TRUNCATION)
    assert flag["name"] == bp.QUARTIC_OUTSIDE_EXPANSION == \
        "the elimination is outside the range of its expansion"
    assert flag["operator_relative_remainder"] == 1.0000000514884309
    assert flag["action_relative_remainder"] == 0.9999096755602227
    assert flag["induced_displacement_norm"] == 16822.018532665603
    assert flag["induced_phase_norm"] == 16822.018532665603
    assert flag["induced_length_norm"] == 0.0
    assert flag["tolerance"] == 1e-15 and flag["action_unmeasured"] is None
    assert flag["detail"] == (
        "along the induced displacement (norm 1.68e+04: squared lengths 0, "
        "link phases 1.68e+04 rad) the second-order remainder of h_1 "
        "relative to its linear term is 1 and the cubic remainder of the "
        "geometric action relative to its quadratic term is 1, against the "
        "declared tolerance 1e-15; the with-quartic poles are read as they "
        "are")
    # a declared tolerance above both remainders leaves the read unflagged
    assert bp.truncation_flags(RECORDED_TRUNCATION, 2.0) == []
    # and one between them flags it on the larger alone
    (flag,) = bp.truncation_flags(RECORDED_TRUNCATION, 0.99995)
    assert flag["tolerance"] == 0.99995


def test_the_flag_of_a_read_with_nothing_eliminated_or_unmeasured():
    """With no induced displacement nothing is eliminated and there is no
    flag. An action remainder that could not be measured leaves the range
    uncertified: the flag is raised with the reason."""
    nothing = {"induced_displacement_norm": 0.0, "induced_length_norm": 0.0,
               "induced_phase_norm": 0.0,
               "operator_relative_remainder": None,
               "action_relative_remainder": None}
    assert bp.truncation_flags(nothing) == []
    unmeasured = {"induced_displacement_norm": 0.25,
                  "induced_length_norm": 0.0, "induced_phase_norm": 0.25,
                  "operator_relative_remainder": 0.0,
                  "action_relative_remainder": None,
                  "action_unmeasured": "log W is not defined there"}
    (flag,) = bp.truncation_flags(unmeasured)
    assert flag["action_relative_remainder"] is None
    assert flag["action_unmeasured"] == "log W is not defined there"
    assert "relative to its quadratic term is unmeasured (log W is not " \
        "defined there), against" in flag["detail"]


def test_the_elimination_carries_its_flag():
    """The elimination of the unrelaxed host carrying the three lowest modes
    of h_1 at kappa = beta = 0.5: the induced displacement is not zero and
    its remainders are above the declared tolerance, so the truncation read
    carries the flag with the numbers it measured; at a declared truncation
    tolerance above both remainders it carries none."""
    spacetime = bp.build_host()

    def eliminate(tolerances=None):
        config = bp.default_config([0.5], [0.5], tolerances=tolerances)
        bare = cob.JointAction(spacetime,
                               bp.action_declaration(spacetime, 0.5, 0.5))
        declaration = bp.action_declaration(spacetime, 0.5, 0.5)
        declaration.covariance = bare.occupation_projector(3)
        action = cob.JointAction(spacetime, declaration)
        carrier = bp.matrix(action.carrier_operator())
        return bp.eliminate_fluctuations(spacetime, action, carrier, 0.5,
                                         0.5, config)["truncation"]
    truncation = eliminate()
    assert truncation["induced_displacement_norm"] > 0.0
    (flag,) = truncation["flags"]
    assert flag["name"] == bp.QUARTIC_OUTSIDE_EXPANSION
    for key in ("operator_relative_remainder", "action_relative_remainder",
                "induced_displacement_norm", "induced_phase_norm",
                "induced_length_norm"):
        assert flag[key] == truncation[key]
    assert flag["tolerance"] == 1e-15
    measured = [flag[key] for key in ("operator_relative_remainder",
                                      "action_relative_remainder")
                if flag[key] is not None]
    assert measured and max(measured) > 1e-15
    loose = eliminate({"truncation_tolerance": 10.0 * max(measured)})
    assert loose["flags"] == []


def _record(content, sectors, flags=(), quartic_flags=None,
            doublet_content=(1, 1, 1)):
    """A content record with one doublet read, its flags and the flags of
    its with-quartic read."""
    out = {}
    for key, (value, irreps) in sectors.items():
        entry = {"lowest_pole": value}
        out[key] = {"quasi_free": entry, "with_quartic": entry,
                    "restriction_to_2T": irreps,
                    "nucleon_reading": "2" in irreps,
                    "delta_reading": sorted(irreps) == ["2'", "2''"]}
    record = {"content": list(content), "flags": list(flags),
              "doublet_reads": [{"doublet_content": list(doublet_content),
                                 "sectors": out}]}
    if quartic_flags is not None:
        record["quartic"] = {"truncation": {"flags": list(quartic_flags)}}
    return record


def test_a_ratio_names_the_flags_of_the_contents_it_is_built_from():
    """The Delta pole is read from a content whose with-quartic read is
    outside its expansion and whose geometry is not allowable; the nucleon
    pole from an unflagged content. The with-quartic ratio names both flags
    beside the Delta pole, the quasi-free ratio the geometry's alone, and
    the nucleon carries none; the lines say so."""
    half, three = str(HALF), str(THREE)
    (quartic,) = bp.truncation_flags(RECORDED_TRUNCATION)
    allowable = {"name": "not Kontsevich-Segal allowable", "detail": "d"}
    records = [
        _record([3, 0, 0], {three: (9.0 + 0j, ["2'", "2''"])}, [allowable],
                [quartic], (0, 2, 1)),
        _record([1, 1, 1], {half: (12.0 + 0j, ["2"])}, [], [])]
    out = bp.ratios(records)
    for pairing in ("by_2T_reading", "by_spin_lift"):
        with_quartic = out["with_quartic"][pairing]
        assert with_quartic["delta_content"] == [3, 0, 0]
        assert with_quartic["delta_flags"] == [
            "not Kontsevich-Segal allowable", bp.QUARTIC_OUTSIDE_EXPANSION]
        assert with_quartic["nucleon_flags"] == []
        quasi_free = out["quasi_free"][pairing]
        assert quasi_free["delta_flags"] == ["not Kontsevich-Segal allowable"]
        assert quasi_free["nucleon_flags"] == []
    lines = bp.ratio_lines(out)
    assert len(lines) == 4
    assert all(line.endswith(
        "; s_D read from a flagged content (not Kontsevich-Segal allowable)")
        for line in lines[:2])
    assert all(line.endswith(
        "; s_D read from a flagged content (not Kontsevich-Segal allowable; "
        "the elimination is outside the range of its expansion)")
        for line in lines[2:])
    assert not any("s_N read from a flagged content" in line
                   for line in lines)


def test_the_flags_of_a_content_are_each_named_once():
    """`content_flags` joins a record's flags with the flag of its
    with-quartic read, a flag the record already lists named once, and the
    content's line carries them."""
    (quartic,) = bp.truncation_flags(RECORDED_TRUNCATION)
    allowable = {"name": "not Kontsevich-Segal allowable", "detail": "d"}
    record = _record([3, 0, 0], {}, [allowable], [quartic])
    assert [flag["name"] for flag in bp.content_flags(record)] == [
        "not Kontsevich-Segal allowable", bp.QUARTIC_OUTSIDE_EXPANSION]
    assert [flag["name"] for flag in
            bp.content_flags(record, "quasi_free")] == [
        "not Kontsevich-Segal allowable"]
    listed = _record([3, 0, 0], {}, [allowable, quartic], [quartic])
    assert len(bp.content_flags(listed)) == 2
    assert bp.content_flags(_record([1, 1, 1], {})) == []
    assert bp.flags_text(_record([1, 1, 1], {}, [], [])) == ""
    text = bp.flags_text(record)
    assert text.startswith("flagged: not Kontsevich-Segal allowable (d); "
                           "the elimination is outside the range of its "
                           "expansion (along the induced displacement")


def test_the_truncation_tolerance_is_a_declared_tolerance():
    """The tolerance of the flag is in the registry, at the declared value
    by default, and is carried by a config."""
    assert "truncation_tolerance" in dict(bp.TOLERANCES)
    config = bp.default_config([1.0], [1.0])
    assert bp.declared_tolerance(config, "truncation_tolerance") == 1e-15
    config = bp.default_config([1.0], [1.0],
                               tolerances={"truncation_tolerance": 0.5})
    assert bp.declared_tolerance(config, "truncation_tolerance") == 0.5
