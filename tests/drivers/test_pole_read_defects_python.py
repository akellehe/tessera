# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The reads of the pole driver read what they name.

Each test states one property of a read of `baryon_poles` and checks it at
the declared tolerances:

* a pole's certificates are read on its eigenspace, so a state and its
  partner come from one decomposition, a degenerate pole's certificates do
  not depend on a basis of it, and a pairing that is not a number is never
  sharp;
* an elimination that integrates out nothing gives the one-body operator;
* ties and the fingerprint's shifts do not move with the unit of the data;
* a fingerprint that was not read is not evaluable;
* quark condition 2 is read on the cell's own supports and says which frame
  the other doublets belong to, and the colour transport of one cell is not
  evaluated;
* the names of the configuration are validated when it is built;
* a written record is JSON, and what is not a finite number is read back;
* a run records the environment its numbers depend on.
"""
import json
import math
import os

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

from tests.drivers import test_pole_read_properties_python as PROPERTIES


# ------------------------------------------------- the pole certificates


@pytest.fixture(scope="module")
def three_halves():
    """The four colour-singlet states of spin 3/2 of the carrier content
    (0, 1, 2), their bilinear dual and the sector's spin images."""
    states, _ = bp.singlet_states((0, 1, 2))
    sectors, _ = bp.spin_sectors(states)
    sector = sectors[bp.SPIN_THREE_HALVES]
    assert sector.shape[1] == 4
    dual = bp.left_inverse(sector)
    images = bp.sector_spin_images(sector, dual, bp.occupation_basis(),
                                   bp.edge_spin_matrices())
    return sector, dual, images


def _certificates(block, sector, dual, images,
                  tolerance=bp.DECLARED_CERTIFICATE_TOLERANCE):
    """The poles of an operator whose compressed block on a sector is
    ``block`` (`sector_poles`) and each pole's certificates, graded at
    ``tolerance``."""
    _, _, read = bp.sector_poles(sector @ block @ dual, sector)
    return read, [
        bp.pole_certificates(projector, multiplicity, bp.SPIN_THREE_HALVES,
                             sector, dual, images, tolerance=tolerance)
        for projector, multiplicity in zip(
            bp.pole_projectors(read, block.shape[0]), read.multiplicity)]


def test_a_sector_s_images_pair_its_states_with_their_duals(three_halves):
    """The Fock vectors of the sector's states and of its dual are
    biorthogonal, as the states and the dual are."""
    _, _, images = three_halves
    pairing = images["left"].T @ images["right"]
    assert np.max(np.abs(pairing - np.eye(4))) < 1e-14


def test_a_simple_pole_s_certificates_are_those_of_its_partners(three_halves):
    """On a block with four simple poles, the certificates of each pole are
    the library's sharp-spin read of the pole's right eigenvector with its
    partner, the matching row of the inverse eigenvector matrix: the
    residuals agree to rounding and the expectation is 15/4."""
    sector, dual, images = three_halves
    rng = np.random.default_rng(3)
    block = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    read, certificates = _certificates(block, sector, dual, images)
    assert list(read.multiplicity) == [1, 1, 1, 1]
    compressed, _, _ = bp.sector_poles(sector @ block @ dual, sector)
    values, right = np.linalg.eig(compressed)
    partners = np.linalg.inv(right)
    basis, spins = bp.occupation_basis(), bp.edge_spin_matrices()
    for pole, certificate in zip(read.poles, certificates):
        k = int(np.argmin(np.abs(values - pole)))
        library = obs.SharpSpin.read(
            spins, bp.to_fock(sector @ right[:, k], basis),
            bp.to_fock(dual.T @ partners[k, :], basis),
            bp.SPIN_THREE_HALVES, 1e-15)
        assert certificate["eigenspace_dimension"] == 1
        assert certificate["spin_lift_sharp"] == library.sharp
        assert abs(certificate["spin_lift_right_residual"]
                   - library.right_residual) < 1e-14
        assert abs(certificate["spin_lift_left_residual"]
                   - library.left_residual) < 1e-14
        assert certificate["spin_lift_expectation"] == pytest.approx(
            bp.SPIN_THREE_HALVES, abs=1e-12)
        assert certificate["determinant_count"] == library.determinant_count
        assert certificate["colour_casimir_residual"] < 1e-14


def test_a_degenerate_pole_s_certificates_do_not_depend_on_the_basis(
        three_halves):
    """An operator with a double pole on the sector, read in two bases of
    the sector (the sector's states and a mixture of them, with the dual of
    each): the pole is one pole of multiplicity two in both, and its
    certificates are the same numbers up to the rounding of the two bases,
    whatever eigenvectors a decomposition of either block would return
    inside the eigenspace. The residuals are of the size of that rounding,
    so the verdicts are compared at 1e-12."""
    sector, dual, images = three_halves
    rng = np.random.default_rng(5)
    mixing = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    shape = np.linalg.solve(mixing, np.diag([2.0, 2.0, 5.0, 7.0]) @ mixing)
    first_read, first = _certificates(shape, sector, dual, images, 1e-12)
    # the same operator on the mixed basis of the same sector
    change = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    mixed = sector @ change
    mixed_dual = bp.left_inverse(mixed)
    mixed_images = bp.sector_spin_images(mixed, mixed_dual,
                                         bp.occupation_basis(),
                                         bp.edge_spin_matrices())
    block = np.linalg.solve(change, shape @ change)
    second_read, second = _certificates(block, mixed, mixed_dual,
                                        mixed_images, 1e-12)
    assert list(first_read.multiplicity) == [2, 1, 1]
    assert list(second_read.multiplicity) == [2, 1, 1]
    for a, b in zip(first, second):
        assert a["eigenspace_dimension"] == b["eigenspace_dimension"]
        assert a["spin_lift_sharp"] and b["spin_lift_sharp"]
        assert a["spin_lift_expectation"] == pytest.approx(
            b["spin_lift_expectation"], abs=1e-11)
        assert a["spin_lift_expectation"] == pytest.approx(
            bp.SPIN_THREE_HALVES, abs=1e-11)
        for key in ("spin_lift_right_residual", "spin_lift_left_residual",
                    "colour_casimir_residual"):
            assert abs(a[key] - b[key]) < 1e-13


@pytest.mark.parametrize("entry", [None, (0, 1), (1, 0), (2, 3)])
def test_a_block_that_is_scalar_up_to_rounding_is_one_sharp_pole(
        three_halves, entry):
    """A block that is twice the identity up to one entry of 1e-16 is one
    pole of multiplicity four at the declared pole rank tolerance. Its
    certificate is that of the whole sector: sharp, with the expectation
    15/4. An eigensolver returns eigenvectors of such a block that are
    parallel to rounding, whose partners in a separate decomposition do not
    pair with them."""
    sector, dual, images = three_halves
    block = 2.0 * np.eye(4, dtype=complex)
    if entry is not None:
        block[entry] = 1e-16
    read, (certificate,) = _certificates(block, sector, dual, images)
    assert list(read.multiplicity) == [4]
    assert certificate["eigenspace_dimension"] == 4
    assert certificate["spin_lift_sharp"]
    assert certificate["spin_lift_expectation"] == pytest.approx(
        bp.SPIN_THREE_HALVES, abs=1e-12)


def test_a_pairing_that_is_not_a_number_is_never_sharp(three_halves):
    """Both eigen-equations hold on every state of the sector, so their
    residuals are at rounding whatever state is read. A projector whose
    trace vanishes has no pairing: the expectation is not a number, and the
    certificate is not sharp although both residuals are within the
    tolerance."""
    sector, dual, images = three_halves
    nilpotent = np.zeros((4, 4), dtype=complex)
    nilpotent[0, 1] = 1.0
    certificate = bp.pole_certificates(nilpotent, 1, bp.SPIN_THREE_HALVES,
                                       sector, dual, images,
                                       tolerance=1e-12)
    assert certificate["spin_lift_right_residual"] < 1e-12
    assert certificate["spin_lift_left_residual"] < 1e-12
    assert not np.isfinite(certificate["spin_lift_expectation"])
    assert certificate["spin_lift_sharp"] is False


def test_the_spinor_certificate_of_an_eigenspace(three_halves):
    """With the identity as the isotypic projector both projector equations
    hold exactly and the weight is one; with the zero projector the
    residuals are one and the weight zero, and the certificate fails."""
    sector, dual, images = three_halves
    size = sector.shape[0]
    projector = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    held = bp.pole_certificates(projector, 2, bp.SPIN_THREE_HALVES, sector,
                                dual, images, np.eye(size), "all")
    assert held["sharp_spinor"] and held["spinor_type"] == "all"
    assert held["spinor_right_residual"] == 0.0
    assert held["spinor_left_residual"] == 0.0
    assert held["spinor_weight"] == pytest.approx(1.0, abs=1e-13)
    failed = bp.pole_certificates(projector, 2, bp.SPIN_THREE_HALVES, sector,
                                  dual, images, np.zeros((size, size)),
                                  "none")
    assert not failed["sharp_spinor"]
    assert failed["spinor_right_residual"] == pytest.approx(1.0, abs=1e-13)
    assert failed["spinor_left_residual"] == pytest.approx(1.0, abs=1e-13)
    assert failed["spinor_weight"] == 0.0
    bare = bp.pole_certificates(projector, 2, bp.SPIN_THREE_HALVES, sector,
                                dual, images)
    assert bare["sharp_spinor"] is None and bare["spinor_weight"] is None


def test_a_sector_s_entry_keeps_its_multiplicities_beside_the_certificates(
        three_halves):
    """`sector_entry` lists the multiplicity of every pole and, parallel to
    it, the certificates of every pole; the lowest pole's certificates are
    repeated beside the poles without replacing the list."""
    sector, dual, _ = three_halves
    shape = np.diag([2.0, 2.0, 5.0, 7.0]).astype(complex)
    entry = bp.sector_entry(bp.SPIN_THREE_HALVES, 0, sector,
                            (("quasi_free", sector @ shape @ dual),))
    read = entry["quasi_free"]
    assert read["multiplicity"] == [2, 1, 1]
    assert [c["eigenspace_dimension"] for c in read["pole_certificates"]] \
        == [2, 1, 1]
    assert read["eigenspace_dimension"] == 2
    assert read["lowest_pole"] == pytest.approx(2.0, abs=1e-12)
    assert all(c["spin_lift_sharp"] for c in read["pole_certificates"])


# -------------------------------------------- an elimination of nothing


def test_an_elimination_of_nothing_is_the_one_body_operator():
    """With no reduced coupling and a 0 by 0 reduced stiffness the quartic
    is the zero operator: the eliminated operator is dGamma of the carrier
    in the frame, the operator the library forms as the one-body part of an
    elimination, and the read says nothing was eliminated."""
    rng = np.random.default_rng(11)
    size = bp.SHEETS * bp.BASE_EDGES
    carrier = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size,
                                                                    size))
    frame = np.eye(size) + 0.1 * rng.normal(size=(size, size))
    dual = np.linalg.inv(frame)
    coupling = rng.normal(size=(size, size)).astype(complex)
    library, eliminated = bp.many_body_operators(
        carrier, [coupling], np.array([[2.0 + 0j]]), frame, dual)
    one_body, nothing = bp.many_body_operators(
        carrier, [], np.zeros((0, 0)), frame, dual)
    assert one_body.shape == library.shape == (816, 816)
    assert np.linalg.norm(one_body - library) \
        < 1e-12 * np.linalg.norm(library)
    assert nothing["effective_action"] is one_body
    assert nothing["eliminated_dimension"] == 0
    assert nothing["stiffness_asymmetry"] is None
    assert nothing["stiffness_conditioning"] is None
    assert nothing["frame_pairing_defect"] == pytest.approx(
        eliminated["frame_pairing_defect"], abs=1e-13)
    assert "no fluctuation is eliminated" in nothing["certificate"]
    assert eliminated["eliminated_dimension"] == 1
    assert np.linalg.norm(eliminated["effective_action"] - library) \
        > 1e-3 * np.linalg.norm(library)


# ------------------------------------------------ ties and the fingerprint


def test_a_tie_is_relative_to_the_poles_and_a_nan_ties_with_nothing():
    for unit in (1e-9, 1.0, 1e9):
        assert not bp._tied(unit * (2.0 + 0j), unit * (2.0 + 1e-7 + 0j))
        assert bp._tied(unit * (2.0 + 0j), unit * (2.0 + 1e-7 + 0j), 1e-6)
        assert bp._tied(unit * (2.0 + 0j), unit * (2.0 + 0j))
    assert bp._tied(0j, 0j)
    assert not bp._tied(complex(math.nan, 0.0), 1.0 + 0j)
    assert not bp._tied(complex(math.nan, 0.0), complex(math.nan, 0.0))


def test_a_shift_is_relative_to_a_scale_of_its_unit():
    assert bp._relative_shift(0.0, 0.0) == 0.0
    assert bp._relative_shift(0.0, 5.0) == 0.0
    assert bp._relative_shift(1e-9, 1e-9) == 1.0
    assert bp._relative_shift(1e-9, 0.0) == math.inf


def _fingerprint(cell):
    """The fingerprint read of a host with the certificates graded at 1e-12,
    above their rounding on the refined support."""
    spacetime = bp.build_host(cell=cell)
    config = bp.default_config([1.0], [1.0],
                               tolerances={"certificate_tolerance": 1e-12})
    return bp.spectral_fingerprint_read(spacetime, 1.0, 1.0, config)


def test_the_fingerprint_s_shifts_do_not_move_with_the_unit_of_length():
    """The fixture with its squared lengths multiplied by 1e9 has h_1
    divided by 1e9 and the same fingerprint: every shift of the relabeling
    and of the refinement, relative to the size of h_1's spectrum, is the
    number it is on the fixture, and the verdicts are the same."""
    fixture = PROPERTIES._fixture_cell()
    small = _fingerprint(fixture)
    large = _fingerprint(PROPERTIES._scaled(fixture, 1e9))
    assert max(abs(v) for v in large["spectrum"]) == pytest.approx(
        1e-9 * max(abs(v) for v in small["spectrum"]), rel=1e-9)
    for read in (small, large):
        assert read["doublet_found"]
        relabeling = read["relabeling"]
        assert relabeling["held"] is True
        for key in ("spectrum_shift", "averaged_spectrum_shift",
                    "doublet_energy_shift", "doublet_weight_shift"):
            assert relabeling[key] < 1e-12
    assert large["refinement"]["held"] == small["refinement"]["held"]
    assert large["refinement"]["refined_rank"] \
        == small["refinement"]["refined_rank"]
    assert large["refinement"]["energy_shift"] == pytest.approx(
        small["refinement"]["energy_shift"], rel=1e-6)
    assert large["refinement"]["overlap"] == pytest.approx(
        small["refinement"]["overlap"], abs=1e-9)


def test_a_fingerprint_that_was_not_read_is_not_evaluable():
    """A host with no spinor doublet has no fingerprint to re-read: the two
    items of quark condition 7 are not evaluable, with the reason, and the
    text says unread. A fingerprint that was read and moved fails."""
    read = _fingerprint(PROPERTIES.DILATED_CELL)
    assert not read["doublet_found"]
    for key in ("relabeling", "refinement"):
        assert read[key]["held"] is None
        assert read[key]["unread"].startswith("no spinor doublet on the host")
    evidence = bp.fingerprint_evidence(read)
    assert [e.name for e in evidence] == ["refinement-stability",
                                          "relabeling-stability"]
    assert [e.held for e in evidence] == [None, None]
    assert all("no spinor doublet on the host" in e.detail for e in evidence)
    text = bp.fingerprint_text(read)
    assert "relabeling unread" in text and "refinement unread" in text

    moved = {"relabeling": {"held": False,
                            "unread": "no spinor doublet on the relabeled "
                                      "host: x"},
             "refinement": {"held": True, "overlap": 0.95,
                            "overlap_floor": 0.9, "refined_rank": 2,
                            "energy_shift": 0.2, "isotypic_dimension": 4}}
    evidence = bp.fingerprint_evidence(moved)
    assert [e.held for e in evidence] == [True, False]
    assert "relabeling unstable" in bp.fingerprint_text(moved)


# -------------------------------------------------- the quark conditions


def _quark(cell):
    spacetime, supports, _, frame = PROPERTIES._spin_frame_of(cell)
    quark = bp.quark_conditions(spacetime, frame["alignments"],
                                {"refusal": "not read"}, 0.0, None,
                                frame_name=frame["name"])
    return frame, {c["number"]: c for c in quark["conditions"]}


def _evidence(condition):
    return {e["name"]: e for e in condition["evidence"]}


def test_the_base_band_of_a_cell_in_the_host_s_frame_is_the_cell_s():
    """The tick-1 cell is read in the declared symmetric host's frame. Its
    own supports carry no certified coexact spinor doublet, and the two
    pieces of evidence of quark condition 2 say so; the detail names the
    host's frame and the doublets of that frame, which are certified."""
    frame, conditions = _quark(PROPERTIES.DILATED_CELL)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    assert all(a["reference_doublet"]["coexact"]
               for a in frame["alignments"])
    two = _evidence(conditions[2])
    assert two["base-band-sector"]["held"] is False
    assert two["protected-base-band"]["held"] is False
    for name in ("base-band-sector", "protected-base-band"):
        detail = two[name]["detail"]
        assert detail.startswith("on the cell's own supports")
        assert "the aligned frame the spin is read in (%s)" \
            % bp.SPIN_FRAME_OF_THE_HOST in detail
    assert conditions[2]["status"] == "Failed"


def test_the_base_band_of_the_fixture_is_its_own():
    """The fixture is read in its own frame, and its own supports carry the
    certified coexact doublet: both pieces of evidence hold."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t, 1e-12)
                for t in range(bp.SHEETS)]
    own = bp.cell_base_bands(supports, 1e-12, 1e-12)
    assert [o["certified"] for o in own] == [True] * 3
    assert [o["nontrivial"] for o in own] == [True] * 3
    assert all(o["band"]["dimension"] == 2 and o["band"]["coexact"]
               and "frame" not in o["band"] for o in own)
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group(), 1e-12, 1e-12)
    quark = bp.quark_conditions(spacetime, [alignment] * bp.SHEETS,
                                {"refusal": "not read"}, 0.0, None,
                                tolerance=1e-12,
                                frame_name=bp.SPIN_FRAME_OF_THE_CELL,
                                degeneracy_tolerance=1e-12)
    two = _evidence({c["number"]: c for c in quark["conditions"]}[2])
    assert two["base-band-sector"]["held"] is True
    assert two["protected-base-band"]["held"] is True


def test_the_colour_transport_of_one_cell_is_not_evaluated():
    """The read of one cell has one cluster and no simplex connecting two:
    the evidence of the colour transport is not evaluable and says why, with
    the declared rule's attachment named as a rule beside it."""
    _, conditions = _quark(PROPERTIES.DILATED_CELL)
    five = _evidence(conditions[5])
    transport = five["color-transport-full-rank"]
    assert transport["held"] is None
    assert transport["detail"].startswith(bp.COLOUR_TRANSPORT_UNREAD)
    assert "det S = (1+0j)" in transport["detail"]
    assert "color-transport-full-rank" in conditions[5]["missing"]
    assert conditions[5]["status"] == "NotEvaluable"


# ------------------------------------------------- names and the records


def test_the_ward_read_names_its_state():
    assert bp.ward_read(None, None, [], None, {}) == {
        "directions": 0, "state": bp.WARD_STATE}
    assert "three modes of h_1 of smallest real part" in bp.WARD_STATE


def test_the_hinges_of_the_regge_sum_are_declared_by_name():
    host = bp.build_host()
    assert bp.action_declaration(host, 1.0, 1.0).regge_hinges \
        == cob.ReggeHinges.Interior
    assert bp.action_declaration(host, 1.0, 1.0, "all").regge_hinges \
        == cob.ReggeHinges.All
    for name in ("every", "Interior", "", None):
        with pytest.raises(ValueError, match="the hinges of the Regge sum "
                                             "are one of interior, all"):
            bp.action_declaration(host, 1.0, 1.0, name)


@pytest.mark.parametrize("declared", [
    {"regge_hinges": "every"}, {"elimination": "phases"},
    {"band_selection": "sorted"}, {"fiber_pinning": "moments"}])
def test_a_name_that_is_not_declared_is_refused_when_the_config_is_built(
        declared):
    with pytest.raises(ValueError, match="is one of|are one of"):
        bp.default_config([1.0], [1.0], **declared)


@pytest.mark.parametrize("content", [(1, 1, 0), (0, 0, 4), (2, 2, 2),
                                     (-1, 2, 2), (1.5, 0.5, 1), (1, 2),
                                     "abc", 3])
def test_a_content_is_three_quarks_in_three_bands(content):
    with pytest.raises(ValueError, match="a content is three non-negative "
                                         "integers that sum to three"):
        bp.checked_contents([content])
    with pytest.raises(ValueError, match="a content is three"):
        bp.default_config([1.0], [1.0], selected_contents=[content])


def test_the_ten_contents_are_accepted_in_any_container():
    assert bp.checked_contents(bp.contents()) == bp.contents()
    assert bp.checked_contents([[1, 1, 1], np.array([0, 0, 3])]) \
        == [(1, 1, 1), (0, 0, 3)]
    config = bp.default_config([1.0], [1.0],
                               selected_contents=[(1, 1, 1), [3, 0, 0]])
    assert config["contents"] == [[1, 1, 1], [3, 0, 0]]
    assert len(bp.default_config([1.0], [1.0])["contents"]) == 10


def test_the_monopole_record_reports_the_bands_the_driver_reads():
    """The record of the declared host's spin read lists the three doublets
    the aligned frame is read on, names the coexact one, and keeps the
    library's own grouping under its name."""
    declared = bp.aligned_doublet_frame(
        bp.monopole_support(), bp.rotation_group(), certified=False)
    assert [band["dimension"] for band in declared["bands"]] == [2, 2, 2]
    assert all("frame" not in band for band in declared["bands"])
    record = bp._monopole_record(declared["spin_read"], declared["bands"])
    assert record["monopole_number"] == 1 and record["odd"]
    assert [band["dimension"] for band in record["bands"]] == [2, 2, 2]
    assert record["half_integer_doublet"] == declared["reference_certified"]
    if record["half_integer_doublet"]:
        assert record["doublet_index"] == declared["reference_carrier"]
        assert record["bands"][record["doublet_index"]]["coexact"]
    else:
        assert record["doublet_index"] == len(record["bands"])
    library = declared["spin_read"]
    assert record["library"]["doublet_index"] == int(library.doublet_index)
    assert [band["dimension"] for band in record["library"]["bands"]] \
        == [int(band.dimension) for band in library.bands]
    json.dumps(bp._jsonable(record), allow_nan=False)


# ------------------------------------------------------- the written files


def _strict(text):
    def refuse(name):
        raise ValueError("%s is not JSON" % name)
    return json.loads(text, parse_constant=refuse)


def test_a_number_that_is_not_finite_is_written_by_name_and_read_back():
    record = {"gap": math.inf, "rank_gap": math.nan, "low": -math.inf,
              "pole": complex(math.nan, 2.0), "norm": np.float64("inf"),
              "values": np.array([1.0, np.nan]), "finite": 1.5,
              "count": 3, "flag": np.bool_(True), "flags": [np.bool_(False)],
              "text": "nan is a word here"}
    written = bp._jsonable(record)
    assert written == {
        "gap": "inf", "rank_gap": "nan", "low": "-inf",
        "pole": {"re": "nan", "im": 2.0}, "norm": "inf",
        "values": [1.0, "nan"], "finite": 1.5, "count": 3, "flag": True,
        "flags": [False], "text": "nan is a word here"}
    assert type(written["flag"]) is bool and type(written["count"]) is int
    read = _strict(json.dumps(written, allow_nan=False))
    assert bp.number(read["gap"]) == math.inf
    assert bp.number(read["low"]) == -math.inf
    assert math.isnan(bp.number(read["rank_gap"]))
    pole = bp.number(read["pole"])
    assert isinstance(pole, complex) and math.isnan(pole.real) \
        and pole.imag == 2.0
    assert bp.number(read["finite"]) == 1.5
    assert bp.number(read["text"]) == "nan is a word here"
    assert bp.number(None) is None
    assert bp.number({"re": 1.0, "im": -2.0}) == 1.0 - 2.0j
    assert bp.number({"re": 1.0}) == {"re": 1.0}


def test_a_line_of_the_points_file_is_json(tmp_path):
    path = tmp_path / "run.points.jsonl"
    bp._append_line(path, {"separation": [math.inf, 0.5],
                           "trace": [{"constrained_rank_gap": math.nan}],
                           "held": np.bool_(True),
                           "frame": np.array([[1 + 1j, 2.0]])})
    (line,) = path.read_text().splitlines()
    assert _strict(line) == {
        "separation": ["inf", 0.5],
        "trace": [{"constrained_rank_gap": "nan"}], "held": True,
        "frame": [[{"re": 1.0, "im": 1.0}, {"re": 2.0, "im": 0.0}]]}


# --------------------------------------------------------- the environment


def test_a_run_records_its_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    record = bp.environment_record()
    assert set(record) == {"argv", "threads", "processors",
                           "linear_algebra", "python", "numpy", "package",
                           "checkout"}
    assert record["argv"] and all(isinstance(a, str) for a in record["argv"])
    assert record["threads"]["OMP_NUM_THREADS"] == "3"
    assert record["threads"]["MKL_NUM_THREADS"] is None
    assert record["processors"] == os.cpu_count()
    assert record["numpy"] == np.__version__
    assert record["package"] and record["package"] != "unknown"
    assert set(record["checkout"]) == {"root", "ref", "commit"}
    commit = record["checkout"]["commit"]
    assert commit is None or (len(commit) == 40
                              and set(commit) <= set("0123456789abcdef"))
    _strict(json.dumps(bp._jsonable(record), allow_nan=False))


def test_the_commit_of_a_checkout_is_read_from_its_files(tmp_path):
    """HEAD and the ref it names are read from the `.git` directory, from
    `packed-refs` when the ref has no file, and through the file a linked
    worktree has in place of the directory."""
    sha, other = "a" * 40, "b" * 40
    plain = tmp_path / "plain"
    (plain / ".git" / "refs" / "heads").mkdir(parents=True)
    (plain / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (plain / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    (plain / "tessera").mkdir()
    assert bp._git_commit(str(plain / "tessera")) == {
        "root": str(plain), "ref": "refs/heads/main", "commit": sha}

    (plain / ".git" / "refs" / "heads" / "main").unlink()
    (plain / ".git" / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n%s refs/heads/main\n"
        % other)
    assert bp._git_commit(str(plain))["commit"] == other

    (plain / ".git" / "HEAD").write_text(sha + "\n")
    assert bp._git_commit(str(plain)) == {
        "root": str(plain), "ref": None, "commit": sha}

    linked = tmp_path / "linked"
    linked.mkdir()
    private = plain / ".git" / "worktrees" / "linked"
    private.mkdir(parents=True)
    (private / "HEAD").write_text("ref: refs/heads/topic\n")
    (private / "commondir").write_text("../..\n")
    (plain / ".git" / "refs" / "heads" / "topic").write_text(other + "\n")
    (linked / ".git").write_text("gitdir: %s\n" % private)
    assert bp._git_commit(str(linked)) == {
        "root": str(linked), "ref": "refs/heads/topic", "commit": other}

    bare = tmp_path / "nothing" / "below"
    bare.mkdir(parents=True)
    found = bp._git_commit(str(bare))
    assert found["commit"] is None or found["root"] != str(bare)


# ------------------------------------------- numbers that are not numbers


def test_a_ratio_whose_denominator_vanishes_is_recorded_as_no_number():
    """A Delta pole at zero, or on the imaginary axis, leaves a ratio with
    no value: the pair is recorded with that ratio not a number and the
    others as they are, and nothing is raised."""
    pair = bp._pair(1.0 + 1.0j, 2.0j, {"label": "kept"})
    assert math.isnan(pair["real_part_ratio"])
    assert pair["pole_ratio"] == pytest.approx(0.5 - 0.5j)
    assert pair["modulus_ratio"] == pytest.approx(math.sqrt(2.0) / 2.0)
    assert pair["label"] == "kept"
    zero = bp._pair(1.0 + 0j, 0j, {})
    assert all(math.isnan(abs(zero[key])) for key in (
        "pole_ratio", "modulus_ratio", "real_part_ratio"))
    regular = bp._pair(1.0 + 0j, 2.0 + 0j, {})
    assert regular["pole_ratio"] == 0.5
    assert regular["modulus_ratio"] == 0.5 and regular["real_part_ratio"] == 0.5
    json.dumps(bp._jsonable(zero), allow_nan=False)


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0),
                                   (1, 2, 0)])
def test_the_lowest_pole_does_not_depend_on_where_a_non_number_stands(order):
    """A pole that is not a number is below nothing: the lowest candidate is
    the lowest number wherever the non-number stands in the list."""
    poles = [complex(math.nan, 0.0), 3.0 + 0j, 2.0 + 1j]
    listed = [poles[k] for k in order]
    assert bp.lowest_pole(listed) == 2.0 + 1j
    candidates = [{"pole": pole, "content": [1, 1, 1],
                   "doublet_content": [k, 0, 0]}
                  for k, pole in enumerate(listed)]
    best = bp.lowest_of(candidates)
    assert best["pole"] == 2.0 + 1j and best["tied"] == []


def test_the_lowest_of_poles_that_are_all_non_numbers_is_the_first():
    nan = complex(math.nan, 0.0)
    assert bp.lowest_pole([]) is None
    assert math.isnan(bp.lowest_pole([nan, nan]).real)
    best = bp.lowest_of([{"pole": nan, "name": "first"},
                         {"pole": nan, "name": "second"}])
    assert best["name"] == "first" and best["tied"] == []
    assert bp.lowest_of([]) is None


def test_the_sign_of_a_hessian_that_is_not_a_number_is_unread():
    assert bp.hessian_sign(2.0 + 0j, 1.0) == "positive"
    assert bp.hessian_sign(-2.0 + 0j, 1.0) == "negative"
    assert bp.hessian_sign(0j, 1.0) == "zero"
    assert bp.hessian_sign(2.0 + 1e-3j, 1.0) == "complex"
    assert bp.hessian_sign(complex(math.nan, 0.0), 1.0) == "unread"
    assert bp.hessian_sign(2.0 + 1.0j, math.nan) == "unread"
    assert bp.hessian_sign(2.0 + 1.0j, math.inf) == "unread"


def _leakage_verdict(recursion):
    spacetime = bp.build_host()
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group(), 1e-12, 1e-12)
    quark = bp.quark_conditions(spacetime, [alignment] * bp.SHEETS,
                                recursion, 0.0, None, tolerance=1e-12)
    conditions = {c["number"]: c for c in quark["conditions"]}
    return (_evidence(conditions[1])["external-leakage"],
            _evidence(conditions[5])["base-transport-leakage"])


def _level(transport_norms, fiber_norms=None):
    record = {"partition": [[0], [1]], "band_ranks": [1, 1],
              "bands_accepted": [True, True], "isolation_gaps": [1.0, 1.0],
              "projector_idempotency": [0.0, 0.0],
              "transport_norms": list(transport_norms)}
    if fiber_norms is not None:
        record["fiber_norms"] = list(fiber_norms)
    return record


@pytest.mark.parametrize("unit", [1e-9, 1.0, 1e9])
def test_the_leakage_is_relative_to_the_fibers_own_operators(unit):
    """A transport between two components of 1e-6 of the components' own
    fiber operators is a leak in every unit of the operator, and one of
    1e-14 of them is none at the tolerance 1e-12."""
    for relative, held in ((1e-6, False), (1e-14, True), (0.0, True)):
        external, base = _leakage_verdict(
            _level([unit * relative], [unit * 2.0, unit * 1.0]))
        assert external["held"] is held and base["held"] is held
        assert "of the largest fiber operator norm" in external["detail"]


def test_a_level_read_without_fiber_norms_gives_the_leakage_as_it_is():
    external, base = _leakage_verdict(_level([0.146]))
    assert external["held"] is False and base["held"] is False
    assert external["detail"] == "inter-component transport norms [0.146]"
    external, _ = _leakage_verdict(_level([]))
    assert external["held"] is True
