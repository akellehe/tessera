# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The loads of the projector functions by quadrature on the tetrahedra (#1170):
the rule against the closed-form integrals of the library, the loads against
those of the interpolant they replace and against the matrix elements of plane
waves, at the zone centre and at a crystal momentum, and the levels of a
confined pseudo-atom against the radial equation by both routes."""
import itertools
from math import factorial

import numpy as np
import pytest
from scipy.special import erf

from tessera.drivers.bands import abinitio, coulomb, loads
from tessera.drivers.bands import pseudopotential as pp
from tessera.drivers.bands.crystal import CrystalCell, richardson
from tessera.drivers.bands.settings import Approximations

TRICLINIC = np.array([[6.0, 0.0, 0.0], [0.5, 6.2, 0.0], [0.3, -0.4, 5.8]])


def two_channel_atom():
    """A smooth two-electron pseudo-atom with one s and one p projector."""
    r = np.linspace(1e-4, 12.0, 2400)
    local = -4.0 * erf(r / 1.1) / r
    s = np.exp(-(r / 0.9) ** 2)
    p = r * np.exp(-(r / 0.8) ** 2)
    s, p = (f / np.sqrt(np.trapezoid((r * f) ** 2, r)) for f in (s, p))
    density = 2.0 * (1.0 / (np.pi * 1.3 ** 2)) ** 1.5 * np.exp(-(r / 1.3) ** 2)
    return pp.Pseudopotential("X", 2.0, r, local, [(0, r * s), (1, r * p)], np.diag([1.5, -0.8]),
                              4.0 * np.pi * r ** 2 * density)


def rule_of(points):
    """The loads by `points` Gauss points per direction (0: the interpolant's);
    one term of the kernel's refinement series, which these tests do not read."""
    return Approximations(1, 1, refinement_terms=1, projector_quadrature=points)


@pytest.mark.parametrize("points", [1, 2, 3, 4, 6])
def test_the_collapsed_rule_is_exact_to_its_degree(points):
    """int_T prod lambda_a^(n_a) = |T| 3! prod n_a! / (3 + sum n_a)! for every
    monomial of degree 2 points - 1 at most; the next degree is not exact."""
    barycentric, weights = loads.collapsed_gauss_rule(points)
    assert len(weights) == points ** 3 and weights.sum() == pytest.approx(1.0, abs=1e-14)
    assert barycentric.min() > 0.0 and np.abs(barycentric.sum(axis=1) - 1.0).max() < 1e-14

    def defect(powers):
        exact = 6.0 * np.prod([factorial(n) for n in powers]) / factorial(3 + sum(powers))
        return abs(weights @ np.prod(barycentric ** np.array(powers), axis=1) - exact) / exact

    for powers in itertools.product(range(2 * points), repeat=4):
        if sum(powers) <= 2 * points - 1:
            assert defect(powers) < 1e-12, powers
    assert defect((0, 2 * points, 0, 0)) > 1e-6


def test_the_rule_reproduces_the_mass_matrix_and_the_triple_integrals_of_the_library():
    cell = CrystalCell(TRICLINIC, 5, kinetic_scale=1.0)
    rule = loads.SimplexQuadrature(cell, 2)                                       # exact to degree 3
    assert rule.volumes.sum() == pytest.approx(cell.volume, rel=1e-13)
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=cell.size), rng.normal(size=cell.size)
    assert np.abs(rule.integrate(rule.at_points(x)[:, :, None])[:, 0] - cell.mass.dressed().real @ x).max() < 1e-13
    triple = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
    product = (rule.at_points(x) * rule.at_points(y))[:, :, None]
    assert np.abs(rule.integrate(product)[:, 0] - triple.loads(x, y)[:, 0]).max() < 1e-13


def test_the_simplices_are_laid_out_by_the_displacements_of_the_grid():
    cell = CrystalCell(TRICLINIC, 4, kinetic_scale=1.0)
    rule = loads.SimplexQuadrature(cell, 1)
    assert all(int(simplex[0]) == i for i, simplex in enumerate(cell.complex.kSimplexVertices(0)))
    for t in range(0, len(rule.tops), 7):
        for a in range(1, 4):
            steps = np.array(cell.grid.displacement(int(rule.tops[t, 0]), int(rule.tops[t, a])))
            assert np.allclose((rule.corners[t, a] - rule.corners[t, 0]) * np.array(cell.divisions), steps)
            assert np.allclose(rule.corners[t, a] - rule.shift[t, a], cell.fractional[rule.tops[t, a]])


def test_a_function_of_compact_support_is_loaded_with_its_whole_integral_across_the_wrap():
    """A Gaussian centred near a corner of the cell reaches across the periodic
    wrap and beyond one image: the loads sum to its integral, and every entry
    carries the displacement of its vertex from the image it belongs to."""
    cell = CrystalCell(TRICLINIC, 8, kinetic_scale=1.0)
    rule = loads.SimplexQuadrature(cell, 6)
    center, width = np.array([0.04, 0.97, 0.5]), 1.1
    local = rule.loads(lambda offsets: np.exp(-(offsets ** 2).sum(axis=1) / width ** 2)[:, None], center,
                       reach=6.5, images=(-1, 0, 1))
    assert local.assemble().sum() == pytest.approx((np.pi * width ** 2) ** 1.5, rel=1e-9)
    fractional = (local.displacement @ np.linalg.inv(cell.lattice)) + center - cell.fractional[local.vertex]
    assert np.abs(fractional - np.rint(fractional)).max() < 1e-12
    assert len(np.unique(np.rint(fractional), axis=0)) > 1                        # more than one image reaches the cell


def _meshes(crystal, n):
    assert Approximations().projector_quadrature == 6
    return [abinitio.MeshCrystal(crystal, n, approximations=rule_of(points)) for points in (0, 6)]


@pytest.mark.slow
def test_the_projector_loads_tend_to_those_of_the_interpolant_and_are_stable_in_the_degree():
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(two_channel_atom(), np.array([0.37, 0.52, 0.45]))])
    differences = []
    for n in (8, 12, 16):
        nodal, quadrature = _meshes(crystal, n)
        assert quadrature.P.shape == nodal.P.shape == (n ** 3, 4) and np.array_equal(quadrature.D, nodal.D)
        differences.append(np.linalg.norm(quadrature.P - nodal.P, axis=0) / np.linalg.norm(nodal.P, axis=0))
        if n == 8:
            finer = abinitio.MeshCrystal(crystal, n, approximations=rule_of(8))
            assert np.linalg.norm(finer.P - quadrature.P) < 2e-6 * np.linalg.norm(quadrature.P)
    differences = np.array(differences)
    assert (differences[1:] < 0.7 * differences[:-1]).all() and differences[0].max() < 0.3 and differences[-1].max() < 0.1


@pytest.mark.parametrize("kappa", [(0.0, 0.0, 0.0), (0.2, -0.1, 0.35)])
def test_the_projector_loads_against_the_matrix_elements_of_plane_waves(kappa):
    """<G + k| beta> of the plane-wave twin, an integral over all space, against
    the load contracted with the vertex values of the plane wave. Both loads
    converge to it at the rate of the interpolated plane wave; the quadrature
    loads carry about half the error at every mesh, and none beyond the rule's
    for the constant."""
    atom, position = two_channel_atom(), np.array([0.37, 0.52, 0.45])
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(atom, position)])
    k = np.asarray(kappa) @ (2.0 * np.pi * np.linalg.inv(crystal.lattice).T)
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [k], [1.0], cutoff=16.0)
    basis = plane_waves.bases[0]
    lowest = np.argsort(basis["kinetic"])[:19]
    G = basis["indices"][lowest] @ plane_waves.reciprocal
    target = basis["P"][lowest] * np.exp(1j * (position @ crystal.lattice) @ k)   # the mesh loads are centred on the ion
    errors = []
    for n in (8, 12):
        row = []
        for mesh in _meshes(crystal, n):
            waves = np.exp(1j * mesh.cell.positions @ G.T) / np.sqrt(crystal.volume)
            P = mesh._projectors_at(kappa, mesh.cell.pencil(kappa)[1]) if any(kappa) else mesh.P
            row.append(np.abs(waves.conj().T @ P - target))
        errors.append(row)
        assert row[1].max() < 0.65 * row[0].max()
        if not any(kappa):
            assert row[1][0, 0] < 1e-7 < row[0][0, 0]                            # the integral of the s projector
    assert errors[1][0].max() < 0.6 * errors[0][0].max() and errors[1][1].max() < 0.6 * errors[0][1].max()


def test_the_loads_at_a_momentum_and_their_derivative():
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(two_channel_atom(), np.array([0.37, 0.52, 0.45]))])
    nodal, mesh = _meshes(crystal, 8)
    kappa = (0.2, -0.1, 0.35)
    k = mesh.cell.momentum(kappa)
    at_momentum = mesh._projectors_at(kappa, None)
    # Where one image reaches a vertex the load is the zone-centre load times the phase of that vertex.
    phase = np.exp(-1j * mesh._displacements(crystal.ions[0][1]) @ k)
    assert np.abs(at_momentum - phase[:, None] * mesh.P).max() < 2e-4 * np.abs(mesh.P).max()
    assert np.abs(mesh._projectors_at((0.0, 0.0, 0.0), None) - mesh.P).max() < 1e-15
    # A reciprocal-lattice vector changes the loads by the phase of each vertex and nothing else.
    shifted = mesh._projectors_at((1.2, -0.1, 0.35), None)
    wrap = np.exp(-1j * (mesh.cell.positions - crystal.ions[0][1] @ crystal.lattice) @ mesh.cell.momentum((1, 0, 0)))
    assert np.abs(shifted - wrap[:, None] * at_momentum).max() < 1e-12
    step = 1e-5
    for axis in range(3):
        direction = np.eye(3)[axis] * step
        numerical = (mesh._loads[0].assemble(direction) - mesh._loads[0].assemble(-direction)) / (2.0 * step)
        assert np.abs(mesh._projector_derivative(axis, None) - numerical).max() < 1e-8
        # The interpolant's route: the same derivative up to the difference of the loads.
        dM = mesh.cell.covariant().sparsePencilPhaseDerivativeAlong(
            [float(w) for w in mesh.cell.edge_displacements[:, axis]]).M
        assert np.linalg.norm(nodal._projector_derivative(axis, dM) - numerical) < 0.5 * np.linalg.norm(numerical)


@pytest.mark.slow
def test_the_levels_of_the_confined_pseudo_atom_extrapolate_to_the_radial_levels():
    """The screened, confined pseudo-atom with its s and p projectors, the local
    potential interpolated at the vertices as the plan has it: the lowest s and
    p levels against the radial equation. The quadrature loads extrapolate at
    least as well as those of the interpolant."""
    atom, strength, position = two_channel_atom(), 1.0, np.full(3, 0.5)
    crystal = abinitio.Crystal(9.0 * np.eye(3), [(atom, position)])
    radial = np.array([pp.radial_levels(atom, 0, 1, strength)[0],
                       pp.radial_levels(atom, 1, 1, strength, step=0.005)[0]])
    defects = {}
    for points in (0, 6):
        spacings, levels = [], []
        for n in (10, 14, 18):
            mesh = abinitio.MeshCrystal(crystal, n, approximations=rule_of(points))
            V = pp.screened_potential(atom, np.linalg.norm(mesh._displacements(position), axis=1), strength)
            A = (mesh.stiffness + mesh.cell.weighted_mass(V).dressed().real).tocsc()
            values, _, _, below = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 4, float(V.min()) - 2.0)
            assert below
            spacings.append(mesh.cell.spacing)
            levels.append([values[0], values[1:4].mean()])        # the mesh splits the p level in two and one
        defects[points] = np.abs(richardson(spacings, levels)[0] - radial)
        assert np.abs(np.array(levels[-1]) - radial).max() > 0.1
    assert (defects[6] < 6e-3).all() and (defects[6] <= defects[0]).all()
