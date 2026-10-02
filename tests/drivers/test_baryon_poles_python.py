# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The nucleon-to-Delta synthesis driver: every certificate it reports, on
the three-sheeted unit-monopole tetrahedron, and its --live path.

The live path is exercised with matplotlib stubbed, as the emergence driver's
live tests do: a GUI cannot be opened in a test process, and the claims under
test are the refusal of a file-only or WebAgg backend and the identity of the
outputs with and without the flag.
"""

import json
import math
import time

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

from tests.drivers import _recursion_run_2026_09_23 as RUN


@pytest.fixture(scope="module")
def alignment():
    """The aligned frame of the declared host at the declared tolerances."""
    return bp.aligned_doublet_frame(bp.monopole_support(),
                                    bp.rotation_group())


@pytest.fixture(scope="module")
def run_alignment():
    """The aligned frame of the declared host at the recorded run's
    tolerances, the ones the spin frames below are read at."""
    return bp.aligned_doublet_frame(
        bp.monopole_support(), bp.rotation_group(),
        RUN.TOLERANCES["degeneracy_tolerance"],
        RUN.TOLERANCES["certificate_tolerance"],
        RUN.TOLERANCES["character_tolerance"])


def _run_spin_frame(supports, symmetry):
    """`spin_frame` at the recorded run's tolerances."""
    return bp.spin_frame(supports, symmetry,
                         RUN.TOLERANCES["degeneracy_tolerance"],
                         RUN.TOLERANCES["certificate_tolerance"],
                         RUN.TOLERANCES["character_tolerance"])


# ------------------------------------------------------------------ host


def test_the_host_carries_the_unit_monopole_on_every_sheet():
    spacetime = bp.build_host()
    for sheet in range(bp.SHEETS):
        support, departure = bp.sheet_support(spacetime, sheet)
        read = support.monopoleNumber()
        assert read.monopole_number == 1 and read.odd and read.bundle
        assert departure < 1e-14
    faces = cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0)).face_holonomies()
    # every face holonomy is a primitive fourth root of unity, +-i
    assert np.max(np.abs(np.abs(np.asarray(faces)) - 1.0)) < 1e-14
    assert np.max(np.abs(np.asarray(faces) ** 2 + 1.0)) < 1e-12


def test_the_declared_host_has_the_tetrahedral_rotation_group():
    """`cell_symmetry` on the declared host: every one of the twelve rotations
    leaves the six equal squared lengths invariant and carries the symmetric
    monopole connection to a gauge-equivalent one on every sheet, with a
    compensation residual at rounding, so the cell's group is the whole
    tetrahedral group at the declared tolerance 1e-15."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports)
    assert symmetry["tetrahedral"] and symmetry["order"] == 12
    assert symmetry["tolerance"] == bp.DECLARED_TOLERANCE
    assert symmetry["length_departure"] == 0.0
    assert symmetry["compensation_residual"] < 1e-15
    assert [r["rotation"] for r in symmetry["rotations"]] == \
        [list(g) for g in bp.rotation_group()]
    assert symmetry["group"][0] == [0, 1, 2, 3]


def test_a_cell_with_one_length_changed_keeps_the_rotations_fixing_that_edge():
    """With the squared length of the edge (0, 2) alone moved from 8 to 8.5
    on every sheet, the rotations that are still symmetries are the identity
    and the half turn about the axis through the midpoints of (0, 2) and
    (1, 3), which exchanges 0 with 2 and 1 with 3; the other ten carry that
    edge onto an edge of squared length 8, a departure of 0.5 / 8.5 of the
    largest squared length, and the cell's group is not tetrahedral."""
    host = bp.build_host()
    cell = {"squared_lengths": [8.0, 8.5, 8.0, 8.0, 8.0, 8.0],
            "links": [complex(u) for u in bp.sheet_links(host, 0)]}
    spacetime = bp.build_host(cell=cell)
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    symmetry = bp.cell_symmetry(spacetime, supports)
    assert not symmetry["tetrahedral"] and symmetry["order"] == 2
    assert symmetry["group"] == [[0, 1, 2, 3], [2, 3, 0, 1]]
    assert symmetry["length_departure"] == pytest.approx(0.5 / 8.5)
    moved = [r for r in symmetry["rotations"] if not r["symmetric"]]
    assert len(moved) == 10
    assert all(r["length_departure"] == pytest.approx(0.5 / 8.5)
               for r in moved)
    assert all(r["compensation_residual"] < 1e-15
               for r in symmetry["rotations"])


def _one_length_changed():
    """The declared host with the squared length of the edge (0, 2) alone
    moved from 8 to 8.5 on every sheet, and its sheets' supports."""
    host = bp.build_host()
    cell = {"squared_lengths": [8.0, 8.5, 8.0, 8.0, 8.0, 8.0],
            "links": [complex(u) for u in bp.sheet_links(host, 0)]}
    spacetime = bp.build_host(cell=cell)
    return spacetime, [bp.sheet_support(spacetime, t)
                       for t in range(bp.SHEETS)]


def test_the_spin_of_a_symmetric_cell_is_read_in_its_own_frame(
        run_alignment):
    alignment = run_alignment
    """`spin_frame` on the declared host at the run's tolerances. The cell
    has all twelve rotations, the spin read of every sheet names the
    j = 1/2 doublet, and the three sheets, which carry the same tetrahedron,
    carry the same doublet labels: the three preconditions of the cell's own
    spin read hold, so the frame is the cell's own and no flag is raised.
    The sheets carry the fixture's connection, so the twelve actions and the
    aligned frame of every sheet are those of the declared symmetric host,
    with the reference doublet the middle carrier (the eigenvalue 4)."""
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    tolerance = RUN.TOLERANCES["certificate_tolerance"]
    symmetry = bp.cell_symmetry(spacetime, supports, tolerance)
    frame = _run_spin_frame(supports, symmetry)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_CELL == \
        "the relaxed cell's own rotation group"
    assert frame["flags"] == []
    assert len(frame["actions"]) == 12 and len(frame["alignments"]) == 3
    declared = bp.rotation_action([bp.monopole_support()] * bp.SHEETS)
    for own, host in zip(frame["actions"], declared):
        assert np.abs(own - host).max() < 1e-14
    for sheet in frame["alignments"]:
        assert sheet["reference_carrier"] == 1
        assert sheet["trialities"] == alignment["trialities"]
        assert np.allclose(sheet["averaged_eigenvalues"],
                           alignment["averaged_eigenvalues"], atol=1e-12)
        assert sheet["intertwining_residual"] < 1e-10


def test_the_spin_of_a_cell_without_the_tetrahedral_group_is_read_in_the_host_frame(
        run_alignment):
    alignment = run_alignment
    """`spin_frame` on the cell with one squared length moved from 8 to 8.5.
    The cell keeps two of the twelve rotations (the test above), so the first
    precondition of its own spin read does not hold. The read is made in the
    frame of the declared symmetric host: the actions of the fixture on
    every sheet and the fixture's aligned frame, the same on the three
    sheets. The one flag names the precondition and carries the numbers
    that decided it: two of twelve rotations, the length departure
    0.5 / 8.5 of the ten rotations that move the changed edge, and a
    gauge-compensation residual at rounding, because the connection is the
    symmetric one."""
    spacetime, supports = _one_length_changed()
    tolerance = RUN.TOLERANCES["certificate_tolerance"]
    symmetry = bp.cell_symmetry(spacetime, supports, tolerance)
    frame = _run_spin_frame(supports, symmetry)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST == \
        "the declared symmetric host"
    (flag,) = frame["flags"]
    assert flag["name"] == "not tetrahedrally symmetric"
    assert flag["order"] == 2 and flag["rotations"] == 12
    assert flag["tolerance"] == tolerance
    assert flag["length_departure"] == pytest.approx(0.5 / 8.5)
    assert flag["compensation_residual"] < 1e-15
    assert flag["detail"].startswith(
        "the relaxed cell has 2 of the 12 rotations of the tetrahedron as "
        "symmetries at the certificate tolerance 1e-08 (largest length "
        "departure 0.0588, ")
    declared = bp.rotation_action([bp.monopole_support()] * bp.SHEETS)
    assert len(frame["actions"]) == 12
    for used, host in zip(frame["actions"], declared):
        assert np.array_equal(used, host)
    assert len(frame["alignments"]) == 3
    for sheet in frame["alignments"]:
        assert sheet["reference_carrier"] == alignment["reference_carrier"]
        assert sheet["trialities"] == alignment["trialities"]
        assert np.array_equal(sheet["frame"], alignment["frame"])


def _symmetric_cell():
    spacetime = bp.build_host()
    supports = [bp.sheet_support(spacetime, t) for t in range(bp.SHEETS)]
    tolerance = RUN.TOLERANCES["certificate_tolerance"]
    return supports, bp.cell_symmetry(spacetime, supports, tolerance)


def test_a_cell_whose_spin_read_names_no_doublet_is_read_in_the_host_frame(
        monkeypatch, run_alignment):
    """The second precondition of the cell's own spin read: every sheet's
    spin read names a j = 1/2 doublet. With the read of the cell's own
    supports replaced by one that names none (`NoSpinorDoublet`), on a cell
    that has the whole tetrahedral group, the frame is the declared
    symmetric host's and the one flag is "no j = 1/2 doublet", with the
    read's own words and the two tolerances it was made at."""
    alignment = run_alignment
    supports, symmetry = _symmetric_cell()
    own = [support for support, _ in supports]
    real = bp.aligned_doublet_frame

    def read(support, group, degeneracy_tolerance, tolerance,
             character_tolerance, certified=True):
        if any(support is sheet for sheet in own):
            raise bp.NoSpinorDoublet("the support carries no j = 1/2 "
                                     "doublet: no band of rank two")
        return real(support, group, degeneracy_tolerance, tolerance,
                    character_tolerance, certified)
    monkeypatch.setattr(bp, "aligned_doublet_frame", read)
    frame = _run_spin_frame(supports, symmetry)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    assert frame["flags"] == [{
        "name": "no j = 1/2 doublet",
        "detail": "the spin read of the relaxed cell names no reference "
                  "doublet: the support carries no j = 1/2 doublet: no band "
                  "of rank two",
        "degeneracy_tolerance": RUN.TOLERANCES["degeneracy_tolerance"],
        "tolerance": RUN.TOLERANCES["certificate_tolerance"]}]
    for sheet in frame["alignments"]:
        assert np.array_equal(sheet["frame"], alignment["frame"])


def test_a_cell_whose_sheets_disagree_on_the_labels_is_read_in_the_host_frame(
        monkeypatch, run_alignment):
    """The third precondition of the cell's own spin read: the aligned
    frames of the sheets carry the same doublet labels. With the read of the
    third sheet replaced by one whose reference carrier is 0 while the first
    two keep the host's 1, on a cell that has the whole tetrahedral group,
    the frame is the declared symmetric host's and the one flag is "the
    sheets' doublet labels disagree", with the labels of the three sheets."""
    alignment = run_alignment
    supports, symmetry = _symmetric_cell()
    third = supports[2][0]
    real = bp.aligned_doublet_frame

    def read(support, group, degeneracy_tolerance, tolerance,
             character_tolerance, certified=True):
        out = real(support, group, degeneracy_tolerance, tolerance,
                   character_tolerance, certified)
        if support is third:
            out = dict(out, reference_carrier=0)
        return out
    monkeypatch.setattr(bp, "aligned_doublet_frame", read)
    frame = _run_spin_frame(supports, symmetry)
    assert frame["name"] == bp.SPIN_FRAME_OF_THE_HOST
    (flag,) = frame["flags"]
    assert flag["name"] == "the sheets' doublet labels disagree"
    labels = [list(alignment["trialities"]), 1]
    assert flag["labels"] == [labels, labels,
                              [list(alignment["trialities"]), 0]]
    assert str(flag["labels"]) in flag["detail"]
    assert [sheet["reference_carrier"] for sheet in frame["alignments"]] == \
        [1, 1, 1]


def test_the_sheets_are_isomorphic():
    spacetime = bp.build_host()
    read = obs.SheetedSupport(3, 6).certifyIsomorphism(
        [np.array(bp.sheet_squared_lengths(spacetime, t)) for t in range(3)],
        [np.array(bp.sheet_links(spacetime, t)) for t in range(3)])
    assert read.isomorphic


def test_the_primal_regge_term_is_empty_on_the_host():
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    assert action.regge_hinge_count() == 0
    assert action.regge_term() == 0


def test_the_monopole_spin_read(alignment):
    """The spin read of the declared host at the declared tolerances. The
    genuine j = 1/2 doublet is the pair at 4, the second of the three
    carriers. The library's read lists it as its third band, because it
    reads the two eigenvalues at 4 - 2/sqrt(3), computed 3e-15 apart, as two
    bands of rank one at the degeneracy tolerance 1e-15. The aligned frame
    names its reference by the carrier that holds the doublet, 1, with that
    carrier's certificates."""
    read = alignment["spin_read"]
    assert read.monopole.monopole_number == 1
    assert read.cocycle.nontrivial
    assert abs(read.cocycle.commutator_phase + 1.0) < 1e-12
    assert read.half_integer_doublet
    assert [band.dimension for band in read.bands] == [1, 1, 2, 2]
    assert read.doublet_index == 2
    assert read.bands[2].coexact and not read.bands[3].coexact
    values = alignment["averaged_eigenvalues"]
    expected = [4 - 2 / np.sqrt(3)] * 2 + [4.0] * 2 + [4 + 2 / np.sqrt(3)] * 2
    assert np.allclose(values, expected, atol=1e-12)
    assert alignment["reference_carrier"] == 1
    assert alignment["reference_certified"] is True
    doublet = alignment["reference_doublet"]
    assert doublet["dimension"] == 2
    assert doublet["spinor_doublet"] and doublet["coexact"]
    assert doublet["eigenvalue"] == pytest.approx(4.0, abs=1e-12)
    assert doublet["coexact_residual"] <= 1e-15
    assert alignment["trialities"] == [1, 0, 2]


def test_the_doublets_are_aligned_to_one_su2_action(alignment):
    assert alignment["intertwining_residual"] < 1e-10
    frame = alignment["frame"]
    assert np.linalg.cond(frame) < 1e6


def test_the_one_per_doublet_sector_is_two_halves_and_a_three_halves():
    states, _ = bp.singlet_states((1, 1, 1))
    sectors, values = bp.spin_sectors(states)
    assert sectors[bp.SPIN_HALF].shape[1] == 4
    assert sectors[bp.SPIN_THREE_HALVES].shape[1] == 4


def test_contents_carry_the_spins_they_can():
    """The colour-singlet states of a content split by total spin: (3, 0, 0)
    has four states of spin 3/2, (2, 1, 0) two of spin 1/2 and four of spin
    3/2, and (1, 1, 1) four of each."""
    for content, half, three in (((3, 0, 0), 0, 4), ((2, 1, 0), 2, 4),
                                 ((1, 1, 1), 4, 4)):
        states, _ = bp.singlet_states(content)
        sectors, read = bp.spin_sectors(states)
        assert sectors.get(bp.SPIN_HALF, np.zeros((0, 0))).shape[-1] == half
        assert sectors[bp.SPIN_THREE_HALVES].shape[1] == three
        assert read["dimensions"] == {bp.SPIN_HALF: half,
                                      bp.SPIN_THREE_HALVES: three}


def test_every_singlet_state_is_a_colour_singlet():
    states, _ = bp.singlet_states((2, 1, 0))
    basis = bp.occupation_basis()
    for k in range(states.shape[1]):
        fock = bp.to_fock(states[:, k], basis)
        residual = np.linalg.norm(bp.colour_casimir(fock)) / \
            np.linalg.norm(fock)
        assert residual < 1e-12


def test_a_colour_nonsinglet_is_caught():
    """Three quarks on one sheet are not a colour singlet."""
    fock = np.asarray(obs.SharpSpin.determinant([0, 3, 6], 18))
    assert np.linalg.norm(bp.colour_casimir(fock)) > 0.5


def test_the_covariant_operator_at_the_monopole_is_not_rotation_symmetric():
    """A measured property of the host, recorded because the calculation
    rests on it: the Whitney covariant operator h_1(z, U) of the specification
    (Definition 2, base-vertex twist b = min) at the unit-monopole connection
    has six simple eigenvalues per sheet and does not commute with the
    projective rotation action D_1(g); the three spinor doublets appear only in
    the rotation-averaged operator (WP line 497, "the T-averaged twisted edge
    Laplacian")."""
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, 1.0))
    h = bp.matrix(action.carrier_operator())[:6, :6]
    values = np.sort(np.linalg.eigvals(h).real)
    assert np.min(np.diff(values)) > 0.1
    support = bp.monopole_support()
    worst = max(np.linalg.norm(np.asarray(support.edgeRepresentation(g)) @ h
                               - h @ np.asarray(support.edgeRepresentation(g)))
                for g in bp.rotation_group())
    assert worst / np.linalg.norm(h) > 0.1


def test_the_host_connection_is_stiff_under_the_villain_term():
    """Every face of the host carries F = +-i. The Wilson form
    beta (1 - cos Theta), the reference computed here from the host's own
    face holonomies, has curvature beta cos Theta = 0 there and would leave
    all 18 phase directions flat; the Villain curvature there is
    beta_V (<m^2>_i - <m>_i^2) = kappa(i) per face, so the nine coexact
    directions have stiffness 4 kappa(i) and the nine pure-gauge ones
    zero."""
    beta = 1.0
    spacetime = bp.build_host()
    action = cob.JointAction(spacetime,
                             bp.action_declaration(spacetime, 1.0, beta))
    faces = np.asarray(action.face_holonomies())
    wilson = beta * np.cos(np.angle(faces))
    assert np.max(np.abs(wilson)) < 1e-12
    villain = np.array(action.holonomy_hessian()).reshape(18, 18)
    kappa = -cob.VillainCharacter(beta).second_derivative(1j).real
    assert kappa == pytest.approx(0.9979584724666767, abs=1e-12)
    values = np.sort(np.linalg.eigvalsh(-villain.real))
    assert np.max(np.abs(values[:9])) < 1e-11
    assert np.allclose(values[9:], 4.0 * kappa, atol=1e-11)


# --------------------------------------------- the Section 7 elimination


def _host_problem(beta, elimination="lengths-and-phases",
                  regge_hinges="interior"):
    """The unrelaxed host carrying the three lowest modes of h_1, and the
    elimination of its fluctuations at kappa = 0.5 under the declared
    action."""
    spacetime = bp.build_host()
    config = bp.default_config([0.5], [beta], regge_hinges=regge_hinges,
                               elimination=elimination)
    bare = cob.JointAction(spacetime, bp.action_declaration(
        spacetime, 0.5, beta, regge_hinges))
    declaration = bp.action_declaration(spacetime, 0.5, beta, regge_hinges)
    declaration.covariance = bare.occupation_projector(3)
    action = cob.JointAction(spacetime, declaration)
    carrier = bp.matrix(action.carrier_operator())
    return spacetime, action, carrier, bp.eliminate_fluctuations(
        spacetime, action, carrier, 0.5, beta, config)


@pytest.fixture(scope="module")
def host_problem():
    return _host_problem(1.0)


def test_the_drazin_inverse_integrates_out_exactly_the_coexact_phases(
        host_problem):
    """The declared action has no length stiffness on the lone tetrahedron:
    its Regge term is zero by structure (no interior hinge) and the holonomy
    term does not see the lengths. With the Villain term the coexact phase
    block is nonsingular at the host's F = +-i, so the null space of A is the
    eighteen lengths and the nine pure-gauge phases (twelve vertices less one
    per sheet), the record says the null space is not pure gauge, A^D is the
    Drazin inverse on the nine coexact phases, and the reduced coordinates
    rebuild it exactly."""
    _, _, _, problem = host_problem
    drazin = problem["record"]["drazin"]
    assert drazin["coordinates"] == 36
    assert drazin["zero_by_structure"] is False
    assert drazin["null_dimension"] == 27
    assert drazin["eliminated_dimension"] == 9
    assert drazin["gauge_dimension"] == 9
    assert not drazin["null_space_is_pure_gauge"]
    assert drazin["gauge_projector_residual"] < 1e-12
    assert drazin["projector_idempotency"] < 1e-12
    assert drazin["drazin_identity_residual"] < 1e-12
    assert drazin["reduction_residual"] < 1e-12
    # the geometric action has no length-phase coupling with the matter term
    # off: the cross block is measured, and it is zero
    checks = problem["record"]["stiffness_checks"]
    assert checks["cross_block_norm"] == 0.0
    assert checks["phase_block_jacobian_departure"] < 1e-6
    assert problem["record"]["expectation_force_check"] < 1e-12


def test_the_ward_identity_holds_on_every_pure_gauge_direction(
        host_problem):
    """(D - Pi(0)) g = 0 to rounding on each pure-gauge direction (measured
    1.1e-15), while Pi(0) g alone is 0.271 of ||Pi(0)|| ||g||: the identity
    is a genuine cancellation between the diamagnetic and the paramagnetic
    terms. The diamagnetic term is read with the commutator of each coupling
    with the direction's generator, and each of the nine directions is the
    coboundary of its vertex function exactly."""
    _, _, _, problem = host_problem
    ward = problem["record"]["ward_identity"]
    assert set(ward) == {"directions", "gauge_derivative",
                         "coboundary_departure", "residual",
                         "paramagnetic_alone"}
    assert ward["directions"] == 9
    assert ward["coboundary_departure"] == 0.0
    assert ward["residual"] < 1e-14
    assert ward["paramagnetic_alone"] == pytest.approx(0.271241535260997,
                                                       abs=1e-12)


def _contour_gauge_derivative(spacetime, shift, radius=0.1, nodes=8):
    """The oracle of `gauge_derivative`: the derivative of every coupling
    along the phase shift ``shift`` by the Cauchy rule on a circle of radius
    ``radius`` in the complex gauge parameter, with ``nodes`` nodes. The
    couplings are entire along a gauge direction, so the rule's error is its
    rounding, of the order of the couplings' size times 1e-16 / radius."""
    edges = spacetime.getEdgeList().toVector()
    saved = [complex(edge.getPhase()) for edge in edges]
    derivative = None
    try:
        for k in range(nodes):
            root = np.exp(2j * np.pi * k / nodes)
            for edge, phase, step in zip(edges, saved, shift):
                edge.setPhase(phase + radius * root * step)
            here = bp.fluctuation_couplings(spacetime, True)
            if derivative is None:
                derivative = [np.zeros_like(o) for o in here]
            for a, o in enumerate(here):
                derivative[a] += root.conjugate() / (nodes * radius) * o
    finally:
        for edge, phase in zip(edges, saved):
            edge.setPhase(phase)
    return derivative


def _commutator_departure_from_the_contour(spacetime):
    """The largest entry of d_g O_a by the commutator minus d_g O_a by the
    contour rule, over the 36 couplings and the nine pure-gauge directions,
    and the largest entry of the derivatives themselves."""
    couplings = bp.fluctuation_couplings(spacetime, True)
    directions = bp.gauge_directions(spacetime, True)
    edges = len(bp.edge_records(spacetime))
    departure, scale = 0.0, 0.0
    for column in range(directions.shape[1]):
        shift = directions[edges:, column]
        generator, off = bp.gauge_generator(spacetime, shift)
        assert off == 0.0
        oracle = _contour_gauge_derivative(spacetime, shift)
        for coupling, expected in zip(couplings, oracle):
            exact = bp.gauge_derivative(coupling, generator)
            departure = max(departure, float(np.abs(exact - expected).max()))
            scale = max(scale, float(np.abs(exact).max()))
    return departure, scale


def test_the_gauge_derivative_is_the_commutator_with_the_generator():
    """`gauge_derivative`: d_g O_a = i (O_a Lambda_g - Lambda_g O_a) with
    Lambda_g the vertex function of the direction at the lowest vertex of
    each degree-one cell (`gauge_generator`), against the Cauchy rule along
    the direction, for every coupling (18 lengths, 18 phases) and every
    pure-gauge direction. On the declared host the derivatives are as large
    as 14.6 and the two agree to 4.1e-13, the contour rule's rounding; on a
    host whose links are off the unit circle (every stored phase moved by a
    complex number of standard deviation 0.3 in each part, link moduli up to
    0.46 from one) they are as large as 29.9 and agree to 7.8e-13; with the
    lengths complex as well, 42.0 and 1.1e-12."""
    spacetime = bp.build_host()
    departure, scale = _commutator_departure_from_the_contour(spacetime)
    assert scale == pytest.approx(14.555, abs=1e-3)
    assert departure < 1e-11
    rng = np.random.default_rng(0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setPhase(complex(edge.getPhase())
                      + complex(rng.normal(0, 0.3), rng.normal(0, 0.3)))
    departure, scale = _commutator_departure_from_the_contour(spacetime)
    assert scale == pytest.approx(29.898, abs=1e-3)
    assert departure < 1e-11
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(complex(edge.getLength())
                       * complex(1 + rng.normal(0, 0.1), rng.normal(0, 0.1)))
    departure, scale = _commutator_departure_from_the_contour(spacetime)
    assert scale == pytest.approx(41.995, abs=1e-3)
    assert departure < 1e-11


def test_the_vertex_function_is_read_along_a_spanning_tree():
    """`gauge_vertex_function` on the host's stored edges: for the shift
    chi_y - chi_x of a complex vertex function chi it returns chi up to one
    constant per sheet (the lowest vertex of each sheet is the root, with
    value zero), and the shift departs from the coboundary of what it
    returns by rounding alone (4.5e-16)."""
    records = bp.edge_records(bp.build_host())
    rng = np.random.default_rng(3)
    chi = rng.normal(size=12) + 1j * rng.normal(size=12)
    shift = [chi[y] - chi[x] for x, y in records]
    read, departure = bp.gauge_vertex_function(records, shift)
    assert sorted(read) == list(range(12))
    for vertex in range(12):
        root = 4 * (vertex // 4)
        assert read[root] == 0.0
        assert abs(read[vertex] - (chi[vertex] - chi[root])) < 1e-15
    assert departure < 1e-15


def test_a_direction_that_is_not_pure_gauge_is_read_with_its_departure(
        host_problem):
    """The phase of the first stored edge (0, 1) alone is not a coboundary.
    The tree gives chi_1 = 1 and zero elsewhere, whose coboundary differs
    from the shift on the edges (1, 2) and (1, 3) by one each, so the
    departure is sqrt(2). The read is made all the same: its residual is
    0.337, the identity failing on a direction that is not a gauge
    direction, with the departure reported beside it."""
    spacetime, _, carrier, problem = host_problem
    records = bp.edge_records(spacetime)
    assert records[0] == (0, 1)
    shift = np.zeros(len(records), dtype=complex)
    shift[0] = 1.0
    chi, departure = bp.gauge_vertex_function(records, shift)
    assert {v: x for v, x in chi.items() if x != 0} == {1: 1.0}
    assert departure == pytest.approx(np.sqrt(2.0), abs=1e-15)
    direction = np.zeros((2 * len(records), 1), dtype=complex)
    direction[len(records):, 0] = shift
    config = bp.default_config([0.5], [1.0])
    ward = bp.ward_read(spacetime, carrier, problem["couplings"], direction,
                        config)
    assert ward["directions"] == 1
    assert ward["coboundary_departure"] == pytest.approx(np.sqrt(2.0),
                                                         abs=1e-15)
    assert ward["residual"] == pytest.approx(0.3372383239704032, abs=1e-12)
    assert ward["paramagnetic_alone"] == pytest.approx(0.1694293597862595,
                                                       abs=1e-12)


def test_the_elimination_matches_a_dense_reference(host_problem,
                                                   alignment):
    """-1/2 J^T A^D J on the three-particle space: the library's elimination in
    the reduced coordinates against a dense assembly from A^D itself."""
    _, _, carrier, problem = host_problem
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    dual = np.linalg.inv(frame)
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = 18
    declaration.carrier = list(carrier.reshape(-1))
    declaration.couplings = [list(o.reshape(-1))
                             for o in problem["reduced_couplings"]]
    declaration.bare_stiffness = list(
        problem["reduced_stiffness"].reshape(-1))
    declaration.occupied_modes = 3
    read = cob.DressedFluctuation(declaration).effective_action(
        list(frame.reshape(-1)), list(dual.reshape(-1)), 3)
    dimension = int(read.dimension)
    # the dense lift is in the library's three-particle basis
    one_body = np.asarray(read.one_body).reshape(dimension, dimension)
    assert np.abs(one_body - bp.second_quantized(dual @ carrier @ frame)
                  ).max() < 1e-10 * np.abs(one_body).max()
    quartic = np.asarray(read.quartic).reshape(dimension, dimension)
    reference = bp.dense_quartic_reference(problem["couplings"],
                                           problem["drazin"], frame, dual)
    assert np.abs(quartic - reference).max() < 1e-9 * np.abs(reference).max()


def test_the_lengths_only_elimination_is_the_plain_inverse():
    """With every hinge of the lone tetrahedron in the Regge sum the length
    block is nonsingular, so the lengths-only elimination inverts it whole:
    no null space, no gauge directions, nothing for the Ward identity to
    check."""
    _, _, _, problem = _host_problem(1.0, elimination="lengths",
                                     regge_hinges="all")
    drazin = problem["record"]["drazin"]
    assert drazin["coordinates"] == 18
    assert drazin["zero_by_structure"] is False
    assert drazin["null_dimension"] == 0
    assert drazin["eliminated_dimension"] == 18
    assert problem["record"]["ward_identity"] == {"directions": 0}


def test_a_lengths_only_elimination_without_an_interior_hinge_eliminates_nothing():
    """The length stiffness is the Regge term's alone. With only the
    interior hinges in the Regge sum the lone tetrahedron has none, so the
    18 x 18 length block A is the zero matrix by structure. The Drazin
    inverse of the zero matrix is the zero matrix: every one of the eighteen
    coordinates lies in the null space, the reduced coordinates are none (R
    has no column and R^T A R is the 0 x 0 matrix), and the elimination
    proceeds with that. No reduced coupling is carried, the induced
    displacement -A^D <J> is zero, and so are the one-body shift
    sum_a (A^D <J>)_a O_a and the constant -1/2 <J>^T A^D <J>. The record
    says the block is zero by structure; its two residuals are zero exactly
    (A A^D A - A and the rebuilt inverse less A^D are both the zero matrix),
    the empty reduced stiffness has no condition number, and the projector
    onto the null space is the identity, idempotent to rounding. With no
    induced displacement the operator's linear change is zero, so its
    relative remainder has no value and is recorded as None."""
    _, _, _, problem = _host_problem(1.0, elimination="lengths",
                                     regge_hinges="interior")
    assert problem["stiffness"].shape == (18, 18)
    assert not problem["stiffness"].any()
    drazin = problem["record"]["drazin"]
    assert drazin["zero_by_structure"] is True
    assert drazin["coordinates"] == 18
    assert drazin["null_dimension"] == 18
    assert drazin["eliminated_dimension"] == 0
    assert drazin["drazin_identity_residual"] == 0.0
    assert drazin["reduction_residual"] == 0.0
    assert drazin["reduced_conditioning"] is None
    assert drazin["projector_idempotency"] < 1e-15
    assert "gauge_dimension" not in drazin
    assert problem["drazin"].shape == (18, 18)
    assert not problem["drazin"].any()
    assert problem["reduced_stiffness"].shape == (0, 0)
    assert problem["reduced_couplings"] == []
    assert len(problem["couplings"]) == 18
    assert problem["induced"].shape == (18,) and not problem["induced"].any()
    assert not problem["shift"].any()
    assert problem["constant"] == 0
    assert problem["record"]["ward_identity"] == {"directions": 0}
    truncation = problem["truncation"]
    assert truncation["induced_displacement_norm"] == 0.0
    assert truncation["action_quadratic_term"] == 0
    assert truncation["operator_relative_remainder"] is None


def test_the_drazin_inverse_of_the_zero_matrix_is_the_zero_matrix():
    """`drazin_elimination` on the 4 x 4 zero matrix with two pure-gauge
    directions, e_3 and e_4: A^D is the zero matrix, the basis of the
    eliminated space has no column, the reduced stiffness is 0 x 0, the
    null space is all four coordinates and the eliminated space none. The
    projector onto the null space is the identity, so it reproduces the two
    gauge directions exactly; the null space is larger than the span of the
    gauge directions (four against two), so it is recorded as not pure
    gauge. A nonzero matrix is recorded as not zero by structure:
    diag(2, 4) has the plain inverse diag(1/2, 1/4)."""
    stiffness = np.zeros((4, 4), dtype=complex)
    directions = np.zeros((4, 2), dtype=complex)
    directions[2, 0] = directions[3, 1] = 1.0
    drazin, basis, reduced, record = bp.drazin_elimination(
        stiffness, directions, bp.DECLARED_TOLERANCE)
    assert drazin.shape == (4, 4) and not drazin.any()
    assert basis.shape == (4, 0) and reduced.shape == (0, 0)
    assert record["zero_by_structure"] is True
    assert record["coordinates"] == 4
    assert record["null_dimension"] == 4
    assert record["eliminated_dimension"] == 0
    assert record["drazin_identity_residual"] == 0.0
    assert record["reduction_residual"] == 0.0
    assert record["reduced_conditioning"] is None
    assert record["gauge_dimension"] == 2
    assert record["gauge_projector_residual"] < 1e-15
    assert record["null_space_is_pure_gauge"] is False
    drazin, basis, reduced, record = bp.drazin_elimination(
        np.diag([2.0, 4.0]).astype(complex), None,
        bp.DECLARED_TOLERANCE)
    assert record["zero_by_structure"] is False
    assert record["eliminated_dimension"] == 2
    assert np.allclose(drazin, np.diag([0.5, 0.25]), atol=1e-15)


def test_the_gauge_dimension_is_read_at_the_elimination_tolerance():
    """The rank of the pure-gauge directions is the number of their singular
    values above the elimination tolerance times the largest. Two directions
    whose second singular value is 5e-13 of the first span two dimensions at
    the declared 1e-15 and one at 1e-9."""
    stiffness = np.zeros((4, 4), dtype=complex)
    directions = np.zeros((4, 2), dtype=complex)
    directions[2, 0] = directions[2, 1] = 1.0
    directions[3, 1] = 1e-12
    singular = np.linalg.svd(directions, compute_uv=False)
    assert singular[1] / singular[0] == pytest.approx(5e-13, rel=1e-3)
    _, _, _, record = bp.drazin_elimination(
        stiffness, directions, bp.DECLARED_TOLERANCE)
    assert record["gauge_dimension"] == 2
    _, _, _, record = bp.drazin_elimination(
        stiffness, directions, bp.DECLARED_TOLERANCE, tolerance=1e-9)
    assert record["gauge_dimension"] == 1


def test_a_content_without_a_value_is_recorded_and_the_scan_continues(
        monkeypatch):
    """A content at which the library names no value (here a band that
    cannot hold the occupation) is recorded with the library's message and
    no pole, and the scan goes on to the next content. A content whose read
    is flagged is a content like any other and is named among the flagged
    ones."""
    def evaluate(content, kappa, beta, config):
        if tuple(content) == (0, 3, 0):
            raise ValueError("band 1 has rank 2")
        flags = ([{"name": "not Kontsevich-Segal allowable", "detail": "d"}]
                 if tuple(content) == (0, 2, 1) else [])
        return {"content": list(content), "doublet_reads": [],
                "flags": flags}
    monkeypatch.setattr(bp, "evaluate_content", evaluate)
    config = bp.default_config(
        [0.5], [0.5], selected_contents=[(0, 3, 0), (1, 1, 1), (0, 2, 1)])
    point = bp.scan_point(0.5, 0.5, config)
    assert point["failed_contents"] == [[0, 3, 0]]
    assert point["flagged_contents"] == [[0, 2, 1]]
    assert point["contents"][0]["failed"] == "band 1 has rank 2"
    assert point["contents"][0]["doublet_reads"] == []
    assert "reason" not in point["contents"][0]
    assert point["contents"][1]["content"] == [1, 1, 1]
    assert bp.point_lines(point)[0] == (
        "kappa=0.5 beta=0.5: 3 contents, 1 without a value, 1 flagged; one "
        "line per (content, doublet content) pair, poles s with "
        "multiplicity x")
    (line,) = bp.content_pair_lines(point["contents"][0], "  ")
    assert line == "  content [0, 3, 0]: no value: band 1 has rank 2"


def test_a_solve_that_overflowed_the_double_has_no_value(monkeypatch):
    """A solve that left a squared length beyond the largest finite double
    leaves no finite geometry, so there is no operator to read a pole on.
    `evaluate_content` raises `ReadWithoutValue` with the reason by name,
    and `scan_point` records the content with that reason, the message, the
    solve's record and no pole; the report says "no value" and the plots
    mark the content so, with the short name of the reason. The solve is
    replaced by a host with one length set to infinity, since no declared
    host overflows."""
    import types

    spacetime = bp.build_host()
    spacetime.getEdgeList().toVector()[3].setLength(complex(math.inf, 0.0))
    report = types.SimpleNamespace(kontsevich_segal_margin=math.pi)
    drive = {"changed": False, "moves_committed": 0}
    solve = {"converged": False, "force_norm": 1.0, "iterations": 7}
    monkeypatch.setattr(bp, "relax_content",
                        lambda content, kappa, beta, config:
                        (spacetime, None, report, drive))
    monkeypatch.setattr(bp, "relaxation_record",
                        lambda report, drive, hessian_reality_tolerance:
                        dict(solve))
    name = "the squared lengths overflowed the double"
    assert name == bp.NO_VALUE_OVERFLOW
    message = (name + " (|z| is not finite on edge 3), so there is no "
               "finite geometry to read a pole on")
    assert bp.geometry_without_value(spacetime, drive) == (name, message)
    assert bp.geometry_without_value(bp.build_host(), drive) is None
    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)])
    with pytest.raises(bp.ReadWithoutValue, match="no finite geometry"):
        bp.evaluate_content((2, 0, 1), 1.0, 1.0, config)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[2, 0, 1]]
    assert point["flagged_contents"] == []
    (record,) = point["contents"]
    assert record["failed"] == message and record["reason"] == name
    assert record["relaxation"] == solve
    assert record["doublet_reads"] == []
    (line,) = bp.content_pair_lines(record, "  ")
    assert line == ("  content [2, 0, 1]: no value: " + message + "; mean "
                    "field converged False (force norm 1 after 7 iterations)")
    assert bp.point_lines(point)[0].startswith(
        "kappa=1 beta=1: 1 contents, 1 without a value, 0 flagged; ")
    assert bp.solve_state(record) == {
        "state": "no value", "reason": "lengths overflowed", "iterations": 7}
    _, _, slots = bp.pole_marks([("201", record)])
    assert slots == [(0.0, "no value")]


def test_a_cell_a_committed_move_changed_has_no_value(monkeypatch):
    """The pole read is defined on the three-sheeted tetrahedron. A drive
    that committed a Pachner move leaves a base complex that is not a
    tetrahedron, and the three-sheeted complex of that base has no pole
    read: `geometry_without_value` names it with the moves and the base
    complex they left, `evaluate_content` raises `ReadWithoutValue`, and
    `scan_point` records the content with that reason and no pole. The
    drive is replaced by its record, a base of five vertices after one
    committed move."""
    import types

    spacetime = bp.build_host()
    report = types.SimpleNamespace(kontsevich_segal_margin=math.pi)
    drive = {"changed": True, "moves_committed": 1,
             "complex_after": {"vertices": 5, "edges": 10, "cells": 4}}
    solve = {"converged": False, "force_norm": 1.0, "iterations": 7}
    monkeypatch.setattr(bp, "relax_content",
                        lambda content, kappa, beta, config:
                        (spacetime, None, report, drive))
    monkeypatch.setattr(bp, "relaxation_record",
                        lambda report, drive, hessian_reality_tolerance:
                        dict(solve))
    name = "the cell is not a tetrahedron after its Pachner moves"
    assert name == bp.NO_VALUE_MOVED
    message = (name + " (1 committed move updates left a base complex of 5 "
               "vertices, 10 edges and 4 cells), and the pole read is "
               "defined on the three-sheeted tetrahedron")
    assert bp.geometry_without_value(spacetime, drive) == (name, message)
    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)])
    with pytest.raises(bp.ReadWithoutValue, match="not a tetrahedron"):
        bp.evaluate_content((2, 0, 1), 1.0, 1.0, config)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == [[2, 0, 1]]
    (record,) = point["contents"]
    assert record["failed"] == message and record["reason"] == name
    assert record["relaxation"] == solve and record["doublet_reads"] == []
    assert bp.solve_state(record) == {
        "state": "no value", "reason": "cell moved", "iterations": 7}


def test_the_elimination_rule_is_declared_and_recorded():
    assert bp.build_parser().parse_args(["run"]).eliminate == \
        "lengths-and-phases"
    assert bp.default_config()["elimination"] == "lengths-and-phases"
    assert bp.build_parser().parse_args(
        ["run", "--eliminate", "lengths"]).eliminate == "lengths"


# ------------------------------------------------------------------ ratios


def test_ratios_are_taken_on_the_lowest_poles():
    def record(content, sectors, doublet_content=(1, 1, 1)):
        """A content record with one doublet read: a content names bands of
        h_1, the sectors are read per doublet content of h-bar_1."""
        out = {}
        for key, (value, irreps) in sectors.items():
            entry = {"lowest_pole": value}
            out[key] = {"quasi_free": entry, "with_quartic": entry,
                        "restriction_to_2T": irreps,
                        "nucleon_reading": "2" in irreps,
                        "delta_reading": sorted(irreps) == ["2'", "2''"]}
        return {"content": content, "doublet_reads": [
            {"doublet_content": list(doublet_content), "sectors": out}]}
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)
    out = bp.ratios([
        record([3, 0, 0], {three: (9.0 + 0j, ["2'", "2''"])}, (0, 2, 1)),
        record([2, 1, 0], {half: (10.0 + 0j, ["2'"]),
                           three: (10.0 + 0j, ["2''", "2"])}),
        record([1, 1, 1], {half: (12.0 + 0j, ["2"]),
                           three: (12.0 + 0j, ["2'", "2''"])})])
    by_spin = out["quasi_free"]["by_spin_lift"]
    assert by_spin["nucleon_content"] == [2, 1, 0]
    assert by_spin["delta_content"] == [3, 0, 0]
    assert by_spin["delta_doublet_content"] == [0, 2, 1]
    assert by_spin["nucleon_doublet_content"] == [1, 1, 1]
    assert by_spin["pole_ratio"] == pytest.approx(10.0 / 9.0)
    assert by_spin["delta_is_a_delta_reading"]
    assert not by_spin["delta_ambiguous_with_spin_half"]
    # the pole is the rest energy (WP v18 §13.3): the target is the mass
    # ratio itself
    assert by_spin["target_mass_ratio"] == pytest.approx(938.272 / 1232.0)
    assert "target_mass_squared_ratio" not in by_spin
    by_reading = out["quasi_free"]["by_2T_reading"]
    # the lowest pole with a 2 in its restriction is the spin-3/2 sector of
    # (2, 1, 0): the tetrahedral ambiguity in action
    assert by_reading["nucleon_content"] == [2, 1, 0]
    assert by_reading["nucleon_spin_j_j_plus_1"] == pytest.approx(3.75)
    assert by_reading["delta_content"] == [3, 0, 0]


def test_the_restriction_to_the_binary_tetrahedral_group():
    assert bp.restriction(bp.SPIN_HALF, 0) == ["2"]
    assert bp.restriction(bp.SPIN_HALF, 1) == ["2'"]
    assert bp.restriction(bp.SPIN_THREE_HALVES, 0) == ["2'", "2''"]
    assert bp.restriction(bp.SPIN_THREE_HALVES, 1) == ["2''", "2"]


def test_the_T_averaged_operator_carries_three_doublets(alignment):
    """h-bar_1 = (1/|T|) sum_g D_1(g)^-1 h_1 D_1(g) (WP line 497) is block
    scalar in the aligned doublet frame; applied to the library's
    combinatorial twisted edge Laplacian it gives the paper's six
    eigenvalues."""
    support = bp.monopole_support()
    actions = bp.rotation_action([support] * bp.SHEETS)
    spacetime = bp.build_host()
    h = bp.matrix(cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0)
    ).carrier_operator())
    averaged = bp.rotation_averaged(h, actions)
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    energies, residual = bp._block_scalar(np.linalg.solve(frame,
                                                          averaged @ frame))
    assert residual < 1e-12
    assert len({round(e.real, 9) for e in energies}) == 3
    combinatorial = np.kron(np.eye(bp.SHEETS),
                            np.asarray(support.edgeLaplacian()))
    values = np.sort(np.linalg.eigvals(
        bp.rotation_averaged(combinatorial, actions)).real)[::3]
    expected = [4 - 2 / np.sqrt(3)] * 2 + [4.0] * 2 + [4 + 2 / np.sqrt(3)] * 2
    assert np.allclose(values, expected, atol=1e-12)


def test_the_trialities_are_the_three_z3_characters(alignment):
    assert sorted(alignment["trialities"]) == [0, 1, 2]
    assert alignment["trialities"][alignment["reference_carrier"]] == 0


# -------------------------------------------------------------------- live


class _Canvas:
    """Enough of a figure canvas for the live loop: repaints are counted, the
    event loop sleeps briefly, and the close callback can be fired."""

    def __init__(self):
        self.draws = 0
        self.callbacks = {}

    def draw_idle(self):
        self.draws += 1

    def start_event_loop(self, interval):
        time.sleep(0.001)

    def mpl_connect(self, name, callback):
        self.callbacks[name] = callback
        return len(self.callbacks)

    def close(self):
        self.callbacks["close_event"](object())


class _Text:
    def __init__(self, message):
        self.messages = [message]

    def set_text(self, message):
        self.messages.append(message)


class _Figure:
    """Enough of a figure for the live loop and its status line."""

    def __init__(self):
        self.canvas = _Canvas()
        self.texts = []
        self.axes = []

    def text(self, x, y, message, **kwargs):
        made = _Text(message)
        self.texts.append(made)
        return made

    def messages(self):
        return [m for text in self.texts for m in text.messages]


def _stub_matplotlib(monkeypatch, backend="qtagg", figures=None,
                     closes=None):
    """Stub pyplot for the live loop. `plt.close` fires the figure's close
    event, as the Qt backend does, and is recorded in `closes`."""
    import matplotlib
    import matplotlib.pyplot as plt

    def close(figure):
        if closes is not None:
            closes.append(figure)
        figure.canvas.close()

    def figure(**kwargs):
        made = _Figure()
        if figures is not None:
            figures.append(made)
        return made

    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)
    monkeypatch.setattr(plt, "isinteractive", lambda: True)
    monkeypatch.setattr(plt, "figure", figure)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "close", close)
    monkeypatch.setattr(plt, "pause", lambda interval: time.sleep(0.001))


@pytest.mark.parametrize("backend", ["agg", "webagg"])
def test_live_refuses_a_non_interactive_or_webagg_backend(monkeypatch,
                                                          backend):
    _stub_matplotlib(monkeypatch, backend)
    with pytest.raises(RuntimeError, match="--live needs an interactive"):
        bp.drive_live(bp.default_config([1.0], [1.0],
                                        selected_contents=[(3, 0, 0)]))


def _cheap_scan_point(kappa, beta, config, on_content=None):
    """A deterministic stand-in for one scan point, so the live path is tested
    without the relaxation's cost; the claim under test is that the live
    worker runs the same `drive` and returns the same result."""
    record = {"content": [3, 0, 0], "seconds": 0.0, "doublet_reads": [{
        "doublet_content": [0, 2, 1],
        "sectors": {str(bp.SPIN_THREE_HALVES): {
            "quasi_free": {"lowest_pole": complex(kappa, beta)},
            "with_quartic": {"lowest_pole": complex(beta, kappa)}}}}]}
    return {"kappa": kappa, "beta": beta, "contents": [record],
            "ratios": bp.ratios([record])}


def test_live_and_headless_outputs_are_identical(monkeypatch):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0], [0.5], selected_contents=[(3, 0, 0)])
    headless = bp.drive(dict(config))
    _stub_matplotlib(monkeypatch)
    drawn = []
    monkeypatch.setattr(bp, "draw_frame",
                        lambda figure, frames, index: drawn.append(index))
    live = bp.drive_live(dict(config))
    assert drawn == [0, 1]
    assert json.dumps(bp._jsonable(live), sort_keys=True) == \
        json.dumps(bp._jsonable(headless), sort_keys=True)


def test_the_window_says_which_scan_point_is_running(monkeypatch):
    def slow_point(kappa, beta, config, on_content=None):
        time.sleep(0.05)
        return _cheap_scan_point(kappa, beta, config)

    monkeypatch.setattr(bp, "scan_point", slow_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    figures = []
    _stub_matplotlib(monkeypatch, figures=figures)
    bp.drive_live(bp.default_config([1.0, 2.0], [0.5],
                                    selected_contents=[(3, 0, 0)]))
    messages = figures[0].messages()
    assert any(m.startswith("scan point 1 of 2 (kappa 1, beta 0.5) is "
                            "running: 0:00 elapsed") for m in messages)
    assert any(m.startswith("scan point 2 of 2 (kappa 2, beta 0.5)")
               for m in messages)


def test_the_drivers_own_close_at_the_end_is_not_reported(monkeypatch,
                                                          capsys):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    closes = []
    _stub_matplotlib(monkeypatch, closes=closes)
    bp.drive_live(bp.default_config([1.0], [0.5],
                                    selected_contents=[(3, 0, 0)]))
    assert len(closes) == 1
    assert "was closed" not in capsys.readouterr().out


def test_keep_open_leaves_the_final_frame_on_screen(monkeypatch):
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    monkeypatch.setattr(bp, "draw_frame", lambda figure, frames, index: None)
    figures, closes = [], []
    _stub_matplotlib(monkeypatch, figures=figures, closes=closes)
    bp.drive_live(bp.default_config([1.0], [0.5],
                                    selected_contents=[(3, 0, 0)]),
                  keep_open=True)
    assert closes == []
    assert figures[0].messages()[-1] == \
        "the scan is complete; writing the outputs"


def test_the_status_line_is_centred_until_a_frame_is_drawn():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure()
    FigureCanvasAgg(figure)
    bp.draw_status(figure, "tick 0 of 1 is running: 0:00 elapsed")
    (centred,) = figure.texts
    assert centred.get_position() == (0.5, 0.5)
    bp.draw_status(figure, "tick 0 of 1 is running: 0:02 elapsed")
    assert figure.texts == [centred]
    assert centred.get_text().endswith("0:02 elapsed")
    figure.add_subplot()
    figure.clear()
    figure.add_subplot()
    bp.draw_status(figure, "tick 1 of 1 is running: 0:00 elapsed")
    (bottom,) = figure.texts
    assert bottom.get_position() == (0.5, 0.004)


def test_elapsed_text():
    assert bp.elapsed_text(0) == "0:00"
    assert bp.elapsed_text(75.9) == "1:15"
    assert bp.elapsed_text(3725) == "1:02:05"


def test_holding_without_an_open_window_returns_at_once():
    import matplotlib.pyplot as plt
    plt.close("all")
    assert bp.hold_live_window("close this window to exit") is False


def test_a_worker_error_reaches_the_main_thread(monkeypatch):
    def exploding(*args, **kwargs):
        raise ValueError("the scan point failed")
    monkeypatch.setattr(bp, "scan_point", exploding)
    _stub_matplotlib(monkeypatch)
    with pytest.raises(ValueError, match="the scan point failed"):
        bp.drive_live(bp.default_config([1.0], [1.0]))


def test_closing_the_window_switches_the_run_to_headless(monkeypatch,
                                                         tmp_path, capsys):
    """The window is closed after the first frame: the scan still runs to the
    end, the result equals the headless one, nothing more is drawn, every
    point reaches the JSON-lines file, and stdout says so."""
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0, 3.0], [0.5],
                               selected_contents=[(3, 0, 0)])
    headless = bp.drive(dict(config))
    figures = []
    _stub_matplotlib(monkeypatch, figures=figures)
    drawn = []

    def draw_then_close(figure, frames, index):
        drawn.append(index)
        figure.canvas.close()

    monkeypatch.setattr(bp, "draw_frame", draw_then_close)
    points = tmp_path / "run.points.jsonl"
    live = bp.drive_live(dict(config), points_file=str(points))
    assert drawn == [0]
    assert len(live["points"]) == 3 and not live["stopped"]
    assert json.dumps(bp._jsonable(live), sort_keys=True) == \
        json.dumps(bp._jsonable(headless), sort_keys=True)
    assert "continues headless" in capsys.readouterr().out
    lines = points.read_text().splitlines()
    assert len(lines) == 4
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [1.0, 2.0, 3.0]


def test_each_point_is_written_as_it_completes(monkeypatch, tmp_path):
    """A scan stopped after two points leaves both on disk."""
    monkeypatch.setattr(bp, "scan_point", _cheap_scan_point)
    config = bp.default_config([1.0, 2.0, 3.0], [0.5],
                               selected_contents=[(3, 0, 0)])
    points = tmp_path / "run.points.jsonl"
    seen = []

    def stop_after_two():
        return len(seen) >= 2

    bp.drive(dict(config), on_frame=lambda frames, index: seen.append(index),
             stop_requested=stop_after_two, points_file=str(points))
    lines = points.read_text().splitlines()
    assert "config" in json.loads(lines[0])
    assert [json.loads(line)["kappa"] for line in lines[1:]] == [1.0, 2.0]
    assert bp.points_path("out/run.json") == "out/run.points.jsonl"


def test_the_pole_table_keeps_every_pole_with_its_pair():
    """One row per pole of every (content, doublet content) sector, not only
    each sector's lowest, in ascending order of real part, each row naming its
    content and doublet content."""
    half, three = str(bp.SPIN_HALF), str(bp.SPIN_THREE_HALVES)

    def column(values):
        return {"poles": values, "multiplicity": [2] * len(values),
                "lowest_pole": min(values, key=lambda p: p.real)}

    def record(content, doublet_content, poles):
        return {"content": content, "doublet_reads": [{
            "doublet_content": doublet_content, "sectors": {
                key: {"quasi_free": column(values),
                      "with_quartic": column(values),
                      "restriction_to_2T": ["2"]}
                for key, values in poles.items()}}]}

    table = bp.pole_table([
        record([2, 1, 0], [1, 1, 1], {half: [7.0 + 0j, 5.0 + 0j],
                                      three: [5.0 + 0j]}),
        record([3, 0, 0], [0, 3, 0], {three: [6.0 + 0j, 4.0 + 0j]})])
    rows = table["quasi_free"][three]
    assert [(row["content"], row["doublet_content"], row["pole"],
             row["lowest_in_sector"]) for row in rows] == [
        ([3, 0, 0], [0, 3, 0], 4.0, True),
        ([2, 1, 0], [1, 1, 1], 5.0, True),
        ([3, 0, 0], [0, 3, 0], 6.0, False)]
    assert [row["pole"] for row in table["with_quartic"][half]] == [5.0, 7.0]
    assert all(row["spin_j_j_plus_1"] == bp.SPIN_HALF
               for row in table["with_quartic"][half])


@pytest.fixture(scope="module")
def projectors(alignment):
    """The isotypic projectors of 2, 2', 2'' on the three-particle sector of
    the declared host, as `evaluate_content` forms them."""
    frame = bp._micro_frame([alignment] * bp.SHEETS)
    return bp.isotypic_projectors(
        alignment, bp.rotation_action([bp.monopole_support()] * bp.SHEETS),
        frame, np.linalg.inv(frame))


def test_the_isotypic_projectors_decompose_the_three_particle_sector(
        projectors):
    """The three projectors of the double cover's spinor types are
    idempotent, orthogonal, and exhaust the 816-dimensional three-particle
    sector of the 18 modes: three particles of the odd-monopole tetrahedron
    are a spinor of one of the three types (WP v18 Section 11.1)."""
    matrices, record = projectors
    assert sorted(matrices) == sorted(bp.IRREP_NAMES)
    total = sum(record[name]["rank"] for name in bp.IRREP_NAMES)
    assert total == len(bp.occupation_basis()) == 816
    for name in bp.IRREP_NAMES:
        assert record[name]["idempotency_residual"] < 1e-9
        assert record[name]["rank"] > 0
    for a, b in (("2", "2'"), ("2'", "2''"), ("2", "2''")):
        assert np.max(np.abs(matrices[a] @ matrices[b])) < 1e-9


def test_the_lift_agrees_with_the_types(alignment, projectors):
    """The spin sectors of the constructed lift are the isotypic components
    the finite group names: for the doublet content (1, 1, 1) of total
    triality zero, the spin-1/2 sector lies in the type 2 and the spin-3/2
    sector in 2' + 2'', half in each."""
    matrices, _ = projectors
    _, triality, sectors = bp.doublet_sectors([1, 1, 1],
                                              alignment["trialities"])
    assert triality == 0
    entry = bp.sector_entry(bp.SPIN_HALF, triality,
                            sectors[bp.SPIN_HALF], (), matrices)
    assert entry["spinor_type"] == "2"
    weights = entry["isotypic_weights"]
    assert abs(weights["2"] - 1.0) < 1e-9
    assert abs(weights["2'"]) < 1e-9 and abs(weights["2''"]) < 1e-9
    entry = bp.sector_entry(bp.SPIN_THREE_HALVES, triality,
                            sectors[bp.SPIN_THREE_HALVES], (), matrices)
    assert entry["spinor_type"] == "2' + 2''"
    weights = entry["isotypic_weights"]
    assert abs(weights["2"]) < 1e-9
    assert abs(weights["2'"] - 0.5) < 1e-9 and abs(weights["2''"] - 0.5) < 1e-9


def test_every_pole_of_a_sector_carries_its_own_certificates(alignment,
                                                              projectors):
    """An operator diagonal in the occupation basis with distinct entries
    splits the four-dimensional spin-1/2 sector of the doublet content
    (1, 1, 1) into distinct poles. Each pole is read with its own spinor,
    spin-lift and colour certificates, and the lowest pole's are repeated
    beside it. Every vector of the sector is a colour singlet of type 2 and
    of spin 1/2 under the lift. At the declared certificate tolerance 1e-15
    the colour certificate holds exactly and the spinor certificate fails on
    every pole: the residuals of the isotypic projector equations are
    1.2e-15 to 1.8e-15, the rounding of the projector. The spin-lift
    residuals are 1.7e-16 to 4.9e-16, and the certificate holds on the four
    poles."""
    matrices, _ = projectors
    _, triality, sectors = bp.doublet_sectors([1, 1, 1],
                                              alignment["trialities"])
    sector = sectors[bp.SPIN_HALF]
    assert sector.shape[1] == 4
    rng = np.random.default_rng(7)
    operator = np.diag(rng.uniform(1.0, 2.0, size=len(
        bp.occupation_basis()))).astype(complex)
    entry = bp.sector_entry(bp.SPIN_HALF, triality, sector,
                            (("quasi_free", operator),), matrices)
    read = entry["quasi_free"]
    poles = read["poles"]
    assert len(poles) >= 2
    assert len(read["pole_certificates"]) == len(poles)
    assert len(poles) == 4
    for certificate in read["pole_certificates"]:
        assert not certificate["sharp_spinor"]
        assert certificate["spinor_type"] == "2"
        assert 1e-15 < certificate["spinor_right_residual"] < 1e-14
        assert 1e-15 < certificate["spinor_left_residual"] < 1e-14
        assert abs(certificate["spinor_weight"] - 1.0) < 1e-9
        assert certificate["spin_lift_right_residual"] < 2e-15
        assert certificate["spin_lift_left_residual"] < 2e-15
        assert certificate["colour_casimir_residual"] == 0.0
    assert [c["spin_lift_sharp"] for c in read["pole_certificates"]] == [
        True, True, True, True]
    # without projectors the spinor certificate is unmeasured, not false
    bare = bp.sector_entry(bp.SPIN_HALF, triality, sector,
                           (("quasi_free", operator),))
    assert bare["quasi_free"]["pole_certificates"][0]["sharp_spinor"] is None
    assert bare["quasi_free"]["pole_certificates"][0]["spin_lift_sharp"]
    lowest = poles.index(read["lowest_pole"])
    for key, value in read["pole_certificates"][lowest].items():
        assert read[key] == value


# -------------------------------------------------------- the anchor atlas


def test_the_anchor_atlas_of_the_declared_host(alignment):
    """Quark condition 3 on the declared host (WP v18 Section 10): the
    reference doublet, the coexact j = 1/2 doublet, anchors on every face of
    every sheet, with the connection-dressed covariance and the transition
    cocycle at machine precision and the invariant coordinates attached; on
    the symmetric host its profile has one modulus on every face."""
    anchor = bp.anchor_atlas_read(bp.build_host(), [alignment] * bp.SHEETS)
    assert anchor["anchored"] and anchor["anchoring_faces"] == 4
    assert anchor["covariance_residual"] < 1e-12
    assert anchor["transition_cocycle_residual"] < 1e-12
    assert anchor["invariant_coordinates_attached"]
    assert anchor["failed_certificates"] == []
    assert anchor["faces"] == [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    assert len(anchor["sheets"]) == bp.SHEETS
    for sheet in anchor["sheets"]:
        assert len(sheet["coordinates"]) == 4
        assert all(len(row) == 3 for row in sheet["coordinates"])
        moduli = [abs(c) for row in sheet["coordinates"] for c in row]
        assert max(moduli) - min(moduli) < 1e-9 and min(moduli) > 0.1
        assert len(sheet["invariant_coordinates"]) == 4
        assert all(abs(a) > 1.0 for a in sheet["invariant_coordinates"])
    evidence = bp.anchor_evidence(anchor)
    assert [e.name for e in evidence] == [
        "anchor-profile-nonzero", "anchor-covariance", "anchor-transitions",
        "anchor-stable-across-frames"]
    assert [e.held for e in evidence] == [True, True, True, None]
    assert bp.anchor_text(anchor).startswith(
        "anchor atlas anchored (4 of 4 faces anchor on every sheet")


def test_the_anchor_evidence_reads_a_refusal_and_an_absent_read():
    refused = {"band": "the band", "faces": [[0, 1, 2]] * 4,
               "tolerance": 1e-8, "anchored": False, "anchoring_faces": 0,
               "covariance_residual": 0.5,
               "transition_cocycle_residual": 0.0,
               "invariant_coordinates_attached": False,
               "failed_certificates": ["identically-zero-profile"],
               "sheets": [{"anchoring_faces": 0}]}
    evidence = bp.anchor_evidence(refused)
    assert [e.held for e in evidence] == [False, False, False, None]
    assert "identically-zero-profile" in evidence[0].detail
    assert "not attached" in evidence[2].detail
    assert bp.anchor_text(refused).startswith("anchor atlas refused")
    assert [e.held for e in bp.anchor_evidence(None)] == [None] * 4
    assert bp.anchor_text(None) == "anchor atlas unread"


# ------------------------------------------------- the spectral fingerprint


def test_the_centroid_lengths_of_the_regular_tetrahedron():
    """On the regular tetrahedron of squared edge length 8 every vertex is at
    squared distance 3 from the centroid, the squared circumradius."""
    radii = bp.centroid_squared_lengths([8.0] * 6)
    assert np.allclose(radii, [3.0] * 4)
    # and on a stretched one the identity for four points holds
    squared = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    total = sum(bp.centroid_squared_lengths(squared))
    assert abs(total - sum(squared) / 4.0) < 1e-12


def test_the_refined_host_keeps_the_boundary_and_its_monopole():
    """The stellar subdivision of the declared host: fifteen vertices, thirty
    edges and twelve tetrahedra; each sheet's boundary faces, their
    holonomies and the unit monopole through them are unchanged, and the new
    edges carry the centroid's lengths and the trivial link. The refined
    support's projective class is nontrivial. Its spin read at the declared
    tolerances (1e-15) finds no j = 1/2 doublet: of the five doubly
    degenerate eigenvalues of its averaged edge Laplacian only the lowest
    pair is read as one band of rank two, which is not coexact, and the
    other four pairs are read as bands of rank one."""
    refined, data = bp.refined_host(bp.build_host())
    assert len(refined.getVertexList().toVector()) == 15
    assert len(refined.getEdgeList().toVector()) == 30
    assert len(data) == bp.SHEETS
    for sheet in data:
        assert len(sheet["squared_lengths"]) == 10
        new = [k for k, e in enumerate(bp.REFINED_EDGES)
               if bp.REFINED_CENTRE in e]
        assert all(abs(sheet["squared_lengths"][k] - 3.0) < 1e-12
                   for k in new)
        assert all(abs(sheet["links"][k] - 1.0) < 1e-12 for k in new)
        support = bp.refined_support(sheet)
        read = support.monopoleNumber()
        assert read.monopole_number == 1 and read.odd
        spin = support.spinRead(bp.refined_rotation_group())
        assert spin.cocycle.nontrivial and not spin.half_integer_doublet
        assert [band.dimension for band in spin.bands] == [2] + [1] * 8
        assert spin.doublet_index == len(spin.bands) == 9
        assert not any(band.coexact for band in spin.bands)


def test_the_spectral_fingerprint_of_the_declared_host():
    """Quark condition 7 on the declared host at the declared tolerances
    (1e-15). The reference doublet is found on the host, with weight 1/3 on
    every edge, and on the relabeled host: under the declared odd relabeling
    the doublet's energy and edge weights move by 2.8e-16 and 5.0e-16, the
    spectrum of the T-average by 7.3e-16, and the spectrum of h_1 by 2.1e-15
    of its scale, above the certificate tolerance, so the relabeling does
    not hold. The refined support's bands are five pairs; the coexact pair
    at 4 has an invariance residual of 3.7e-14 and a coexact residual of
    2.3e-14, above the certificate tolerance, so no band is the certified
    doublet there, the overlap is not read, and the read says which band is
    nearest and by how much. Both pieces of evidence are measured and
    fail."""
    config = bp.default_config([1.0], [1.0])
    read = bp.spectral_fingerprint_read(bp.build_host(), 1.0, 1.0, config)
    assert read["doublet_found"] and read["sheet"] == 0
    assert read["tolerance"] == 1e-15
    assert np.allclose(read["doublet_weights"], [1.0 / 3.0] * 6)
    assert [band["dimension"] for band in read["band_certificates"]] == \
        [2, 2, 2]
    relabeling = read["relabeling"]
    assert relabeling["permutation"] == [1, 0, 2, 3]
    assert relabeling["monopole_number"] == -1  # the orientation flips
    assert relabeling["doublet_found"] and not relabeling["held"]
    assert 1e-15 < relabeling["spectrum_shift"] < 1e-14
    assert relabeling["averaged_spectrum_shift"] < 1e-15
    assert relabeling["doublet_energy_shift"] < 1e-15
    assert relabeling["doublet_weight_shift"] < 1e-15
    refinement = read["refinement"]
    assert not refinement["doublet_found"] and not refinement["held"]
    assert refinement["monopole_number"] == 1
    assert refinement["overlap_floor"] == bp.DECLARED_REFINEMENT_OVERLAP
    assert "overlap" not in refinement
    bands = refinement["band_certificates"]
    assert [band["dimension"] for band in bands] == [2] * 5
    nearest = min(bands, key=lambda band: band["coexact_residual"])
    assert nearest["eigenvalue"] == pytest.approx(4.0, abs=1e-12)
    assert 1e-15 < nearest["coexact_residual"] < 1e-12
    assert 1e-15 < nearest["invariance_residual"] < 1e-12
    assert refinement["unread"].startswith(
        "no spinor doublet on the refined support: no band is a certified "
        "coexact spinor doublet at the tolerance 1e-15: the band nearest "
        "the coexact sector (eigenvalue 4, rank 2) has coexact residual ")
    evidence = bp.fingerprint_evidence(read)
    assert [e.name for e in evidence] == ["refinement-stability",
                                          "relabeling-stability"]
    assert [e.held for e in evidence] == [False, False]
    assert evidence[0].detail == refinement["unread"]
    assert bp.fingerprint_text(read).startswith(
        "spectral fingerprint: relabeling unstable (spectrum shift 2.1e-15, "
        "doublet weight shift 5e-16); refinement unstable (no spinor "
        "doublet on the refined support: no band is a certified coexact ")
    assert [e.held for e in bp.fingerprint_evidence(None)] == [None, None]
    assert bp.fingerprint_text(None) == "spectral fingerprint unread"


# --------------------------------------------- flags after the mean field


def _poles_of(record):
    """Every pole of a content record, over its doublet contents, spins and
    columns."""
    return [pole for read in record["doublet_reads"]
            for entry in read["sectors"].values()
            for name in bp.COLUMNS for pole in entry[name]["poles"]]


def test_a_relaxed_cell_without_the_tetrahedral_group_is_read_in_the_host_frame():
    """(0123, 201) of the recursion's tick-0 run with the occupied fiber
    pinned, at the declared tolerances: the drive takes the residual norm to
    1.7e-14, and the relaxed cell has lost every rotation but the identity
    (squared lengths that differ by two thirds of the largest, a
    gauge-compensation residual of order one). The first precondition of the
    cell's own spin read does not hold, so the spin is read in the frame of
    the declared symmetric host and the content is flagged "not
    tetrahedrally symmetric" with the numbers of the symmetry read. The
    geometry is Kontsevich-Segal allowable, so that is the one flag. The
    content is read like any other: it is not among the contents with no
    value, it has the ten doublet contents (the triples of occupations of
    2, 2', 2'' that sum to three), each with its sectors and their poles,
    and its line in the report lists the flag and the frame."""
    from tessera.drivers import recursion as R

    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)])
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == []
    assert point["flagged_contents"] == [[2, 0, 1]]
    (record,) = point["contents"]
    assert "failed" not in record and "reason" not in record
    (flag,) = record["flags"]
    assert flag["name"] == "not tetrahedrally symmetric"
    assert flag["detail"].startswith(
        "the relaxed cell has 1 of the 12 rotations of the tetrahedron as "
        "symmetries at the certificate tolerance")
    assert flag["order"] == 1 and flag["rotations"] == 12
    assert flag["tolerance"] == bp.declared_tolerance(
        config, "certificate_tolerance") == 1e-15
    assert flag["length_departure"] == pytest.approx(0.6657, abs=5e-4)
    assert flag["compensation_residual"] > 1.0
    relaxation = record["relaxation"]
    assert relaxation["band_selection"] == "continuation"
    # the end point is at rounding, above the declared tolerance 1e-15
    assert 1e-15 < relaxation["force_norm"] < 1e-12
    assert relaxation["converged"] is False
    assert relaxation["stop_reason"] == \
        "no move and no scaled step lowers the residual norm"
    assert relaxation["moves_committed"] == 0
    assert relaxation["complex_after"] == {"vertices": 4, "edges": 6,
                                           "cells": 1}
    assert relaxation["spin_frame"] == bp.SPIN_FRAME_OF_THE_HOST == \
        "the declared symmetric host"
    symmetry = relaxation["symmetry"]
    assert symmetry["order"] == 1 and symmetry["group"] == [[0, 1, 2, 3]]
    assert symmetry["length_departure"] == flag["length_departure"]
    assert symmetry["compensation_residual"] == flag["compensation_residual"]
    # the frames recorded are the declared host's, the same on every sheet
    assert [f["reference_carrier"] for f in relaxation["frames"]] == [1] * 3
    assert len({tuple(f["trialities"]) for f in relaxation["frames"]}) == 1
    # the poles are read: ten doublet contents, each with its sectors
    assert [read["doublet_content"] for read in record["doublet_reads"]] == \
        [list(c) for c in bp.contents()]
    assert all(read["sectors"] for read in record["doublet_reads"])
    assert len(_poles_of(record)) > 0
    lines = bp.point_lines(point)
    assert lines[0].startswith(
        "kappa=1 beta=1: 1 contents, 0 without a value, 1 flagged; ")
    assert "; flagged: not tetrahedrally symmetric (the relaxed cell has 1 " \
        "of the 12 rotations" in lines[1]
    assert lines[1].endswith(
        "; spin read in the frame of the declared symmetric host")
    text = R._content_line({"cell": [0, 1, 2, 3]}, record)
    assert "): read, flagged: not tetrahedrally symmetric (" in text
    # the plots mark a flagged content by its solve, as any other
    assert bp.solve_state(record) == {
        "state": "not converged", "reason": "no descent",
        "iterations": relaxation["iterations"]}


def test_a_read_on_the_boundary_of_the_allowable_domain_is_flagged_by_name():
    """(0123, 201) of the recursion's tick-0 run with no constraint of the
    occupied fiber pinned, at the declared tolerances: nothing holds the
    lengths against the band's force, and the drive follows the step until
    one squared length is negative, on the boundary of the Kontsevich-Segal
    allowable domain, past which the engine scores no trial (7 accepted
    updates, the largest |z| 29 times the host's, residual norm 12.7). That
    geometry is finite, so it is read; its margin, zero to rounding, is not
    above the declared tolerance, so the content is flagged "not
    Kontsevich-Segal allowable" with the margin and the tolerance the margin
    was compared with, and the flags of the geometry come first in the
    record. The record carries the solve (the band selection, the accepted
    updates, the force, why it stopped, every step proposal) and the ten
    doublet contents, and the report prints the flag on the content's
    line."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(2, 0, 1)],
                               fiber_moments=0)
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    point = bp.scan_point(1.0, 1.0, config)
    assert point["failed_contents"] == []
    assert point["flagged_contents"] == [[2, 0, 1]]
    (record,) = point["contents"]
    assert "failed" not in record
    flag = record["flags"][0]
    assert flag["name"] == "not Kontsevich-Segal allowable"
    assert flag["detail"].startswith(
        "the geometry the mean-field solve reached is not Kontsevich-Segal "
        "allowable (margin ")
    solve = record["relaxation"]
    assert flag["kontsevich_segal_margin"] == \
        solve["kontsevich_segal_margin"]
    assert abs(flag["kontsevich_segal_margin"]) < 1e-12
    assert flag["tolerance"] == bp.declared_tolerance(
        config, "allowability_tolerance") == 1e-15
    assert flag["kontsevich_segal_margin"] <= flag["tolerance"]
    # every further flag is one of the spin read's
    assert {f["name"] for f in record["flags"][1:]} <= {
        "not tetrahedrally symmetric", "no j = 1/2 doublet",
        "the sheets' doublet labels disagree"}
    assert solve["spin_frame"] in (bp.SPIN_FRAME_OF_THE_CELL,
                                   bp.SPIN_FRAME_OF_THE_HOST)
    assert (solve["spin_frame"] == bp.SPIN_FRAME_OF_THE_HOST) == \
        (len(record["flags"]) == 2)
    assert solve["band_selection"] == "continuation"
    assert solve["converged"] is False
    stationary = "no move and no scaled step lowers the residual norm"
    assert solve["stop_reason"] == stationary
    assert solve["iterations"] == 7 and solve["moves_committed"] == 0
    assert len(solve["trace"]) > solve["iterations"]
    assert solve["residual_trace"][-1] == pytest.approx(12.69, rel=1e-2)
    assert solve["largest_length_ratio"] == pytest.approx(29.41, rel=1e-2)
    assert all("jacobian_rank" in entry and "force_norm" in entry
               for entry in solve["trace"])
    assert [read["doublet_content"] for read in record["doublet_reads"]] == \
        [list(c) for c in bp.contents()]
    lines = bp.point_lines(point)
    assert lines[0].startswith(
        "kappa=1 beta=1: 1 contents, 0 without a value, 1 flagged; ")
    assert lines[1].startswith("  content [2, 0, 1] mean field converged "
                               "False")
    assert "; stopped: " + stationary + " (" in lines[1]
    assert "; flagged: not Kontsevich-Segal allowable (the geometry the " \
        "mean-field solve reached is not Kontsevich-Segal allowable" \
        in lines[1]
    text = R._content_line({"cell": [0, 1, 2, 3]}, record)
    assert "): read, flagged: not Kontsevich-Segal allowable (" in text
    assert "; stopped: " + stationary + " (" in text
    assert bp.solve_state(record) == {
        "state": "not converged", "reason": "no descent",
        "iterations": solve["iterations"]}


def test_pinning_every_power_sum_of_a_degenerate_fiber_at_the_declared_tolerances():
    """`--fiber-pinning power-sums` pins every power sum of the occupied
    fiber at the host (m_c = r). On (0123, 030) of the recursion's tick-0
    run the fiber is one eigenvalue repeated on the three sheets, so its
    three power sums are one independent constraint stated three times. At
    the declared rank tolerance 1e-15 the dependent rows are not read as
    zero, their least-squares multipliers are of order 1e11 to 1e12, and
    the drive ends after 7 accepted updates at a residual norm of 0.023
    (9.03 at the host) with the pinned moments 0.2 % to 0.6 % off their
    targets. With multipliers of that size the end point is decided at
    rounding, so the residual and the moments are asserted within a factor
    of ten. The record carries the fiber's rank, the three multipliers,
    the residuals, the Hessian along the Hellmann-Feynman force with its
    sign, and the content's line prints them."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)],
                               fiber_pinning="power-sums")
    assert config["fiber_moments"] == "r"
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    _, _, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    solve = bp.relaxation_record(
        report, drive, bp.declared_tolerance(config,
                                             "hessian_reality_tolerance"))
    assert solve["converged"] is False
    assert solve["stop_reason"] == \
        "no move and no scaled step lowers the residual norm"
    assert solve["residual_trace"][0] == pytest.approx(9.0311, rel=1e-4)
    assert 2.3e-3 < solve["residual_trace"][-1] < 0.23
    assert solve["fiber_rank"] == 3
    assert solve["fiber_pinning"] == "power-sums"
    assert solve["fiber_moments"] == 3 and len(solve["multipliers"]) == 3
    assert max(abs(m) for m in solve["multipliers"]) > 1e9
    relative = [abs(r) / abs(t) for r, t in zip(solve["moment_residuals"],
                                                solve["moment_targets"])]
    assert 2e-4 < min(relative) and max(relative) < 0.06
    assert solve["joint_jacobian"]["size"] == 15
    assert solve["force_hessian_sign"] in ("positive", "negative", "complex")
    text = bp.relaxation_text(solve)
    assert "3 of the occupied fiber's 3 power sums pinned at the host" in text
    assert "Hessian on the range of the Hellmann-Feynman force" in text
    assert "(%s)" % solve["force_hessian_sign"] in text


def test_the_declared_read_pins_the_eigenvalue_of_every_occupied_band():
    """The declared pinning: the eigenvalue of each occupied band, one
    constraint per band. (0123, 030) occupies one band of rank three, so one
    eigenvalue is pinned with one multiplier; the drive takes the residual
    norm from 9.03 at the host to 1.4e-14 in 8 accepted updates with the
    eigenvalue held at the host's value 0.708 to parts in 1e14 and the
    multiplier -3, and the record and the content's line say so. At the
    declared tolerance 1e-15 that end point is reported not converged."""
    from tessera.drivers import recursion as R
    from tests.drivers import _recursion_run_2026_09_23 as RUN

    config = bp.default_config([1.0], [1.0], selected_contents=[(0, 3, 0)])
    assert config["fiber_pinning"] == bp.DECLARED_FIBER_PINNING == "eigenvalues"
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 2, 3)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    _, _, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    assert report.fiber_constraint_form == \
        cob.FiberConstraintForm.BandEigenvalues
    solve = bp.relaxation_record(report, drive)
    assert solve["residual_trace"][0] == pytest.approx(9.0311, rel=1e-4)
    assert solve["residual_trace"][-1] < 1e-12
    assert 1e-15 < solve["force_norm"] < 1e-12 and solve["converged"] is False
    assert solve["stop_reason"] == \
        "no move and no scaled step lowers the residual norm"
    assert solve["iterations"] == drive["accepted_updates"] == 8
    assert solve["moves_committed"] == 0
    assert solve["fiber_rank"] == 3
    assert solve["fiber_pinning"] == "eigenvalues"
    assert solve["fiber_moments"] == 1 and len(solve["multipliers"]) == 1
    assert solve["multipliers"][0] == pytest.approx(-3.0, abs=1e-9)
    (band,) = report.bands
    assert solve["moment_targets"][0] == pytest.approx(0.70797, abs=5e-5)
    assert solve["moment_targets"][0] == pytest.approx(band.eigenvalues[0],
                                                        rel=1e-9)
    assert abs(solve["moment_residuals"][0]) < 1e-12 * abs(
        solve["moment_targets"][0])
    text = bp.relaxation_text(solve)
    assert ("the eigenvalues of 1 occupied bands (fiber rank 3) pinned at "
            "the host") in text
