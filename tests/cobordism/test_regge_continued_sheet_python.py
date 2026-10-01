# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The primal Regge term of `JointAction` on continued Riemann sheets.

A Euclidean tetrahedron's face cofactors C_ii are negative real, which is the
cut of the principal square root. With the principal branch, an imaginary part
of 1e-9 of either sign in one squared length flips the sign of
sqrt(C_ii) sqrt(C_jj) and sends a dihedral angle theta to pi - theta. The
branch ledger of whitepaper v17 and specification section 4.2 require every
root to carry a sheet continued from a Euclidean reference, and
`ReggeBranch.Continued` does this. These tests check four things:

- the continued residual is continuous under infinitesimal imaginary
  perturbations, where the principal one jumps;
- the continued and principal values agree on real Euclidean input;
- a lengths-only Newton solve of Regge plus the matter term w tr(h), whose
  stationary point on the regular boundary of the 4-simplex is known in
  closed form, reaches it on the continued sheet from a start on the cut,
  where the principal sheet's solve is refused;
- a complex with no interior hinge reports its Regge term as structurally
  zero.
"""
import cmath
import itertools
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob

#: The boundary of the 4-simplex, K_1 of the recursion's small case.
BOUNDARY_OF_FOUR_SIMPLEX = [list(c) for c in itertools.combinations(range(5), 4)]

#: The grown squared lengths of tick 1 of the small case (two three-sheeted
#: unit-monopole tetrahedra, Villain beta = 1, kappa = 1), one sheet, as the
#: investigation rebuilt them with the driver's own functions.
TICK_ONE_SQUARED_LENGTHS = {
    (0, 1): 3514.0423787734694 - 4.3145912891174506e-16j,
    (0, 2): 2979.714322017706 - 0.014839706049857057j,
    (0, 3): 2979.714322017705 + 0.014839706050435563j,
    (0, 4): 2988.418933181496 + 2.894118586061205e-13j,
    (1, 2): 5372.075510137858 - 5.948555137902076j,
    (1, 3): 5372.075510137858 + 5.948555137902638j,
    (1, 4): 5367.231340215099 + 2.847674256197719e-13j,
    (2, 3): 6691.453221076351 + 3.055333763768431e-13j,
    (2, 4): 6694.668957613439 + 0.028063009933435502j,
    (3, 4): 6694.668957613439 - 0.028063009932878913j,
}


def _complex(cells, squared):
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        a, b = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        edge.setLength(cmath.sqrt(complex(squared[(a, b)])))
    return spacetime


def _declaration(spacetime, branch, matter=0.0):
    """Primal Regge over the interior hinges on the named sheet, and, with a
    nonzero ``matter`` weight, the matter term at the identity covariance,
    w tr(h)."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 1.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.Interior
    declaration.regge_branch = branch
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = matter
    if matter:
        edges = len(spacetime.getEdgeList().toVector())
        declaration.covariance = list(np.eye(edges, dtype=complex).reshape(-1))
    return declaration


def _perturbed_gradient_jumps(spacetime, action, delta):
    """|grad(z + delta e_k) - grad(z)| for every edge k."""
    edges = spacetime.getEdgeList().toVector()
    base = np.array(action.length_stationarity())
    jumps = []
    for edge in edges:
        length = complex(edge.getLength())
        edge.setLength(cmath.sqrt(length * length + delta))
        jumps.append(np.linalg.norm(np.array(action.length_stationarity())
                                    - base))
        edge.setLength(length)
    return np.array(jumps)


@pytest.mark.parametrize("delta", [1e-9j, -1e-9j, 1e-9, -1e-9])
def test_the_continued_residual_is_continuous_where_the_principal_one_jumps(
        delta):
    spacetime = _complex(BOUNDARY_OF_FOUR_SIMPLEX, TICK_ONE_SQUARED_LENGTHS)
    continued = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Continued))
    principal = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Principal))
    # the fixture sits on the cut: the principal evaluation puts 12 of the 30
    # dihedral angles on another sheet than the continued one
    assert continued.regge_off_principal_angles() == 12
    assert principal.regge_off_principal_angles() == 0
    # a change of 1e-9 in one squared length moves a continuous gradient by
    # O(1e-9 |d grad/dz|), which is below 1e-12 here
    assert _perturbed_gradient_jumps(spacetime, continued, delta).max() < 1e-12
    if delta == -1e-9j:
        # the perturbation that crosses the cut: the principal gradient jumps
        assert _perturbed_gradient_jumps(spacetime, principal,
                                         delta).max() > 1e-3


def test_the_continued_and_principal_values_agree_on_real_euclidean_input():
    rng = np.random.default_rng(11)
    points = rng.normal(size=(5, 4))
    squared = {(a, b): complex(np.sum((points[a] - points[b]) ** 2))
               for a, b in itertools.combinations(range(5), 2)}
    spacetime = _complex(BOUNDARY_OF_FOUR_SIMPLEX, squared)
    continued = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Continued))
    principal = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Principal))
    assert continued.regge_off_principal_angles() == 0
    assert continued.regge_term() == principal.regge_term()
    np.testing.assert_array_equal(
        np.array(continued.length_stationarity()),
        np.array(principal.length_stationarity()))


def test_a_path_through_a_degenerate_face_is_refused_by_name():
    """The continued sheets follow every angle and content root along the
    straight path from the Euclidean reference to the current geometry,
    refining each step until every root turns by at most a quarter turn.
    With the squared length of the edge (0, 1) moved from 1 to 9 on a
    regular tetrahedron the faces on that edge pass through a degenerate
    triangle (l_01 = l_02 + l_12 = 2 at z = 4, three eighths of the way),
    where the cofactor a root is taken of vanishes and the root's turn is a
    jump no step is fine across. The walk refuses by name there instead of
    accepting a step that is not fine; the principal sheet, which follows no
    path, is unaffected."""
    edges = list(itertools.combinations(range(4), 2))
    regular = _complex([[0, 1, 2, 3]], {e: 1.0 + 0j for e in edges})

    def declaration(branch):
        declared = cob.JointActionDeclaration()
        declared.carrier_degree = 1
        declared.gravitational_weight = 1.0
        declared.regge_form = cob.ReggeForm.Primal
        declared.regge_hinges = cob.ReggeHinges.All
        declared.regge_branch = branch
        declared.holonomy_weight = 0.0
        declared.matter_weight = 0.0
        return declared

    continued = cob.JointAction(regular, declaration(cob.ReggeBranch.Continued))
    principal = cob.JointAction(regular, declaration(cob.ReggeBranch.Principal))
    assert continued.regge_term() == pytest.approx(principal.regge_term())
    for edge in regular.getEdgeList().toVector():
        pair = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        if pair == [0, 1]:
            edge.setLength(3.0 + 0j)
    with pytest.raises(ValueError, match="cannot be followed") as refusal:
        continued.regge_term()
    assert "makes no fine step" in str(refusal.value)
    assert "2^-30" in str(refusal.value)
    # the principal sheet evaluates the geometry as it stands
    principal.regge_term()


def test_regge_and_matter_converge_on_the_continued_sheet():
    """Primal Regge plus w tr(h) on the boundary of the 4-simplex. At the
    regular metric every length equation is the same by symmetry:
    dS_Regge/dz_e = eps_0 / (2 sqrt z) with eps_0 = 2 pi - 3 arccos(1/3)
    (Schlaefli), and by the degree -1 homogeneity of h (WP v17 line 263)
    w d tr(h)/dz_e = -w c / (10 z^2) with c = z tr h, so the stationary
    point is the regular metric at z* = (w c / (5 eps_0))^{2/3}. From 5%
    away in the real parts, with imaginary parts of 1e-9 that put 14 of the
    30 dihedral angles on the supplement sheet of the principal branch, the
    Newton solve on the continued sheet reaches z* (to the tolerance 1e-10
    on a residual whose derivative in z is of order 1e-4); on the principal
    sheet it is refused."""
    edges = list(itertools.combinations(range(5), 2))
    unit = _complex(BOUNDARY_OF_FOUR_SIMPLEX, {e: 1.0 + 0j for e in edges})
    c = complex(cob.JointAction(unit, _declaration(
        unit, cob.ReggeBranch.Continued, matter=1.0)).matter_term()).real
    eps_0 = 2 * math.pi - 3 * math.acos(1.0 / 3.0)
    w = 100.0
    z_star = (w * c / (5 * eps_0)) ** (2.0 / 3.0)
    regular = _complex(BOUNDARY_OF_FOUR_SIMPLEX,
                       {e: z_star + 0j for e in edges})
    assert np.max(np.abs(cob.JointAction(regular, _declaration(
        regular, cob.ReggeBranch.Continued, matter=w)).length_stationarity())) \
        < 1e-13
    start = {e: z_star * (1.0 + 0.05 * ((3 * i) % 4 - 1.5))
             + 1e-9j * ((7 * i) % 5 - 1.5) for i, e in enumerate(edges)}
    solve = cob.HolomorphicRelaxationDeclaration()
    solve.relax_lengths = True
    solve.relax_links = False
    solve.relax_multipliers = False
    solve.tolerance = 1e-10
    solve.jacobian_mode = cob.HolomorphicJacobianMode.RealAxisDifference
    solve.contour_radius = 1e-6
    spacetime = _complex(BOUNDARY_OF_FOUR_SIMPLEX, start)
    action = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Continued, matter=w))
    assert action.regge_off_principal_angles() == 14
    report = cob.HolomorphicRelaxation(action, solve).solve()
    assert report.converged
    assert report.residual_norm < 1e-10
    assert len(report.steps) <= 6
    assert report.regge_hinge_count == 10
    assert not report.regge_structurally_zero
    for edge in spacetime.getEdgeList().toVector():
        assert abs(complex(edge.getLength()) ** 2 - z_star) < 1e-6

    principal_space = _complex(BOUNDARY_OF_FOUR_SIMPLEX, start)
    principal = cob.HolomorphicRelaxation(
        cob.JointAction(principal_space, _declaration(
            principal_space, cob.ReggeBranch.Principal, matter=w)),
        solve).solve()
    assert not principal.converged


def test_a_complex_with_no_interior_hinge_reports_a_structurally_zero_regge():
    """Every 4-subset of 7 vertices: every triangle lies in four tetrahedra,
    so no hinge has a closed link and the interior-hinge sum is empty."""
    cells = [list(c) for c in itertools.combinations(range(7), 4)]
    squared = {e: 8.0 + 0j for e in itertools.combinations(range(7), 2)}
    spacetime = _complex(cells, squared)
    action = cob.JointAction(spacetime, _declaration(
        spacetime, cob.ReggeBranch.Continued))
    assert action.regge_hinge_count() == 0
    assert action.regge_structurally_zero()
    assert action.regge_term() == 0
    assert not np.any(np.array(action.length_stationarity()))
    # with the Regge weight declared zero the term is absent, not zero by
    # structure
    declaration = _declaration(spacetime, cob.ReggeBranch.Continued)
    declaration.gravitational_weight = 0.0
    assert not cob.JointAction(spacetime, declaration).regge_structurally_zero()
