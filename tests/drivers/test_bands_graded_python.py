# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The periodic mesh graded toward the ions (#1167): the Kuhn grid bisected
about the ions and moved toward them, the premises of the Hermitian
specialization on it, and the Coulomb kernel without translation invariance.
Each counterpart of something the uniform grid has in closed form is held to
that closed form on the Kuhn grid, and then to an independent reference on a
graded mesh."""
import itertools

import numpy as np
import pytest
import scipy.sparse as sp

from tessera import chainhodge as ch
from tessera.drivers.bands import abinitio, coulomb
from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands import pseudopotential as pp
from tessera.drivers.bands.crystal import CrystalCell
from tessera.drivers.bands.graded import (GradedCell, GradedCoulombKernel, IonGrading, IonRefinement, KuhnBisection,
                                          pseudo_atom_levels)

LATTICE = 6.0 * np.eye(3)
CENTRE = [[0.5, 0.5, 0.5]]
KAPPA = (0.1, -0.2, 0.3)


def test_without_grading_the_cell_is_the_kuhn_grid():
    uniform = CrystalCell(LATTICE, 6, kinetic_scale=1.0)
    graded = GradedCell(LATTICE, 6, kinetic_scale=1.0)
    assert sorted(map(tuple, graded.complex.kSimplexVertices(3))) == sorted(map(tuple, uniform.grid.cells()))
    assert np.abs(graded.fractional - uniform.fractional).max() == 0.0
    for mine, theirs in zip(graded.pencil(KAPPA), uniform.pencil(KAPPA)):
        assert abs(mine - theirs).max() < 1e-13
    potential = pot.cosine_potential(uniform, 0.3)
    assert abs(graded.pencil(KAPPA, potential)[0] - uniform.pencil(KAPPA, potential)[0]).max() < 1e-13
    assert graded.orientation_margin == pytest.approx(1.0)
    assert graded.certify(KAPPA).holds()


class TestMovedVertices:
    def _cell(self, n=8):
        return GradedCell(LATTICE, n, IonGrading(CENTRE, 2.5, 3.0), kinetic_scale=1.0)

    def test_the_premises_hold_and_the_volume_is_kept(self):
        cell = self._cell()
        certificate = cell.certify(KAPPA)
        assert certificate.holds(), certificate
        assert 0.0 < cell.orientation_margin < 1.0
        A, M = cell.pencil()
        assert M.sum().real == pytest.approx(216.0, rel=1e-12)
        assert np.abs(A @ np.ones(cell.size)).max() < 1e-12          # the constants are the kernel of the stiffness matrix
        # The mesh is finer at the ion than far from it.
        lengths = np.sqrt(np.real(cell.squared_lengths))
        middle = cell.positions[np.asarray(cell.edges)[:, 0]] + 0.5 * cell.edge_displacements - 3.0
        near = np.linalg.norm(middle, axis=1) < 0.5
        assert lengths[near].mean() < 0.6 * lengths[~near].mean()

    def test_the_links_on_the_true_displacements_are_a_gauge_of_the_links_on_the_grid_steps(self):
        cell = self._cell(6)
        A, M = cell.pencil(KAPPA)
        steps = ch.CovariantChainHodge(cell.base, ch.Connection(cell.complex, cell.grid.blochLinks(cell.edges, list(KAPPA))),
                                       7, False).sparsePencil()
        gauge = sp.diags(np.exp(1j * (cell.vertex_offsets @ cell.momentum(KAPPA))))
        assert abs(gauge.conj() @ sp.csc_matrix(steps.A) @ gauge - A).max() < 1e-12
        assert abs(gauge.conj() @ sp.csc_matrix(steps.M) @ gauge - M).max() < 1e-12

    def test_the_geometry_reaches_the_solver_through_the_declared_fields_of_a_spacetime(self):
        cell = self._cell(6)
        for mine, theirs in zip(cell.pencil(KAPPA), cell.pencil_from_spacetime(KAPPA)):
            assert abs(mine - theirs).max() < 1e-12

    def test_a_grading_that_folds_the_mesh_or_moves_an_ion_is_refused(self):
        with pytest.raises(ValueError, match="folds"):
            # Two ions whose pulls cancel on each other and compress the mesh across the line that joins them.
            GradedCell(LATTICE, 6, IonGrading([[0.0, 0.5, 0.5], [0.5, 0.5, 0.5]], 4.5, 50.0), kinetic_scale=1.0)
        with pytest.raises(ValueError, match="do not cancel"):
            GradedCell(LATTICE, 6, IonGrading([[0.5, 0.5, 0.5], [0.75, 0.5, 0.5]], 2.5, 2.0), kinetic_scale=1.0)
        with pytest.raises(ValueError, match="even power"):
            IonGrading(CENTRE, 2.5, 2.0, flatness=3)


class TestBisection:
    def test_the_bisected_mesh_is_conforming_and_periodic(self):
        cell = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, [(2.0, 1), (1.0, 2)]), kinetic_scale=1.0)
        assert cell.size > 4 * 216
        # The vertices of the grid come first, in the numbering of the grid.
        assert np.abs(cell.fractional[:216] - CrystalCell(LATTICE, 6, kinetic_scale=1.0).fractional).max() == 0.0
        # Every triangle is a face of exactly two top simplices: no vertex hangs inside a face or an edge.
        faces = {}
        for simplex in cell.complex.kSimplexVertices(3):
            for face in itertools.combinations(simplex, 3):
                faces[face] = faces.get(face, 0) + 1
        assert set(faces.values()) == {2}
        assert len(faces) == len(cell.complex.kSimplexVertices(2))
        # Three levels of bisection halve every edge: the simplices are Kuhn simplices of 1, 1/2 and 1/4 the size.
        assert cell.largest_volume_ratio == pytest.approx(64.0)
        certificate = cell.certify(KAPPA)
        assert certificate.holds(), certificate
        A, M = cell.pencil()
        assert M.sum().real == pytest.approx(216.0, rel=1e-12)
        assert np.abs(A @ np.ones(cell.size)).max() < 1e-12
        with pytest.raises(NotImplementedError, match="Topology"):
            cell.spacetime()

    def test_the_closure_bisects_the_neighbours_a_cut_edge_needs(self):
        mesh = KuhnBisection((3, 3, 3))
        before = len(mesh.elements)
        mesh.refine(0)
        assert len(mesh.elements) == before + 6                  # the six simplices around a cube's diagonal
        mesh.refine(next(iter(mesh.elements)))
        cells, fractional, (v, w, d) = mesh.arrays()
        assert len(fractional) == 27 + len(mesh.midpoints)
        assert np.all(v < w) and np.abs(d).max() <= 1.0 / 3.0 * np.sqrt(3.0)

    def test_a_free_particle_and_both_gradings_together(self):
        k2 = float(np.sum(CrystalCell(LATTICE, 6, kinetic_scale=1.0).momentum(KAPPA) ** 2))
        uniform = CrystalCell(LATTICE, 6, kinetic_scale=1.0).solve(KAPPA, 2).energies[0] - k2
        refined = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, [(2.0, 1)]), kinetic_scale=1.0)
        assert 0.0 < refined.solve(KAPPA, 2).energies[0] - k2 < uniform
        both = GradedCell(LATTICE, 6, IonGrading(CENTRE, 2.5, 2.0), IonRefinement(CENTRE, [(1.5, 1)]), kinetic_scale=1.0)
        assert both.certify(KAPPA).holds()
        assert 0.0 < both.solve(KAPPA, 2).energies[0] - k2 < 2.0 * uniform


def test_localized_levels_converge_at_second_order_where_the_uniform_mesh_fails():
    """A well of width 0.35 in a cell of side 6, with levels 1s, 1p, 2s and 1d:
    the uniform mesh of 4096 vertices is off by a third of the binding energy,
    the bisected mesh of 2322 by six per cent, and one more halving about the
    ion divides the error by four."""
    depth, width = 160.0, 0.35
    well = lambda r: -depth * np.exp(-r * r / (2.0 * width ** 2))
    s = pot.radial_levels(well, 1.0, 0, 2, radius=20.0)
    reference = np.sort(np.concatenate([s, np.repeat(pot.radial_levels(well, 1.0, 1, 1, radius=20.0), 3),
                                        np.repeat(pot.radial_levels(well, 1.0, 2, 1, radius=20.0), 5)]))

    def errors(cell):
        levels = cell.solve((0.0, 0.0, 0.0), 10, potential=pot.gaussian_well(cell, depth, width), tolerance=1e-8).energies
        return levels - reference

    uniform = errors(CrystalCell(LATTICE, 12, kinetic_scale=1.0))
    assert uniform[0] > 0.3 * abs(reference[0])
    shells = [(2.0, 1), (1.2, 2), (0.7, 3)]
    coarse = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, shells), kinetic_scale=1.0)
    fine = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, [(2.4, 1), (1.6, 2), (1.0, 3), (0.6, 4)]),
                      kinetic_scale=1.0)
    first, second = errors(coarse), errors(fine)
    assert coarse.size < 2400
    assert np.all(first > 0.0) and np.all(second > 0.0)          # a Galerkin method with the interpolated well above it
    assert first[0] < 0.07 * abs(reference[0])
    assert np.all(second < 0.4 * first)
    assert second[5:].max() < 2.0 and uniform[5:].min() > 10.0   # the 1d level, 17.5 below zero


class TestCoulombKernel:
    def test_on_the_kuhn_grid_it_is_the_fourier_kernel(self):
        uniform, graded = CrystalCell(LATTICE, 6, kinetic_scale=1.0), GradedCell(LATTICE, 6, kinetic_scale=1.0)
        fourier, sparse = coulomb.GridCoulombKernel(uniform, 8.0 * np.pi), coulomb.CoulombKernel.of_cell(graded, 8.0 * np.pi)
        assert isinstance(sparse, GradedCoulombKernel)
        rng = np.random.default_rng(7)
        rho = rng.normal(size=(uniform.size, 3)) + 1j * rng.normal(size=(uniform.size, 3))
        for arguments in ((None, None), (None, 0.7), (KAPPA, None), (KAPPA, 0.0), (KAPPA, 0.7)):
            assert np.abs(fourier.potential(rho, *arguments) - sparse.potential(rho, *arguments)).max() < 1e-11
        assert sparse.momentum_entry(KAPPA) == pytest.approx(fourier.momentum_entry(KAPPA), rel=1e-12)
        assert sparse.momentum_entry_limit([1.0, 2.0, 3.0]) == pytest.approx(8.0 * np.pi / 216.0, rel=1e-12)
        assert sparse.auxiliary_function() == pytest.approx(fourier.auxiliary_function(), rel=1e-12)
        assert sparse.auxiliary_function(KAPPA) == pytest.approx(fourier.auxiliary_function(KAPPA), rel=1e-12)
        for axis in range(3):
            assert np.abs(fourier.potential_derivative(rho, axis) - sparse.potential_derivative(rho, axis)).max() < 1e-11
        assert sparse.zero_momentum_constant((2, 3, 4)) == pytest.approx(fourier.zero_momentum_constant((2, 3, 4)), abs=1e-12)
        transfers = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        assert sparse.zero_momentum_constant((2, 3), transfers) == pytest.approx(
            fourier.zero_momentum_constant((2, 3), transfers), abs=1e-12)

    def test_on_a_graded_mesh_the_solve_is_the_resolvent_of_the_library(self):
        cell = GradedCell(LATTICE, 4, IonGrading(CENTRE, 2.5, 1.5), IonRefinement(CENTRE, [(1.6, 1)]), kinetic_scale=1.0)
        kernel = GradedCoulombKernel(cell, 1.0)
        rng = np.random.default_rng(3)
        rho = rng.normal(size=(cell.size, 2)) + 1j * rng.normal(size=(cell.size, 2))
        # (zeta - h_0)^-1 rho = M (zeta M - A)^-1 rho at zeta = 0.
        resolvent = np.asarray(cell.covariant(KAPPA).resolvent(0, 0.0, rho))
        mass = cell.pencil(KAPPA)[1]
        assert np.abs(mass @ kernel.potential(rho, KAPPA) + resolvent).max() < 1e-9 * np.abs(resolvent).max()
        # At the zone centre: the pseudo-inverse, of zero mean, on the neutralized load.
        phi = kernel.potential(rho)
        assert np.abs(kernel.weights @ phi).max() < 1e-10
        neutral = rho - np.outer(kernel.weights, rho.sum(axis=0)) / kernel.volume
        assert np.abs(cell.pencil()[0] @ phi - neutral).max() < 1e-10

    def test_the_energy_of_a_gaussian_charge_and_the_constant_on_a_bisected_mesh(self):
        """A narrow Gaussian charge at the ion: its energy on the bisected mesh
        is nearer the fine uniform mesh than that of the uniform mesh with more
        vertices. The zero-momentum constant tends to the Madelung constant
        whichever vertex carries the probe; the vertices differ at second order."""
        def energy(cell):
            kernel = coulomb.CoulombKernel.of_cell(cell, 8.0 * np.pi)
            radius2 = ((cell.positions - 3.0) ** 2).sum(axis=1)
            mass = cell.mass.dressed().real
            load = mass @ np.exp(-radius2 / (2.0 * 0.3 ** 2))
            return kernel.energy(load / load.sum()).real

        target = energy(CrystalCell(LATTICE, 24, kinetic_scale=1.0))
        uniform = energy(CrystalCell(LATTICE, 10, kinetic_scale=1.0))
        refined_cell = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, [(1.6, 1), (1.0, 2)]), kinetic_scale=1.0)
        assert refined_cell.size < 1000
        assert abs(energy(refined_cell) - target) < 0.5 * abs(uniform - target)

        madelung = 2.0 * coulomb.MADELUNG_SC / 6.0
        cell = GradedCell(LATTICE, 6, refinement=IonRefinement(CENTRE, [(1.6, 1)]), kinetic_scale=1.0)
        kernel = GradedCoulombKernel(cell, 8.0 * np.pi)
        far = kernel.zero_momentum_constant((2, 3, 4))
        kernel.probes = [int(np.argmin(np.linalg.norm(cell.positions - 3.0, axis=1)))]
        near = kernel.zero_momentum_constant((2, 3, 4))
        grid = coulomb.GridCoulombKernel(CrystalCell(LATTICE, 6, kinetic_scale=1.0), 8.0 * np.pi).zero_momentum_constant((2, 3, 4))
        assert abs(far - madelung) < 0.3 * abs(grid - madelung)
        assert abs(near - madelung) < 1.2 * abs(grid - madelung)


def soft_atom():
    from scipy.special import erf
    r = np.linspace(1e-4, 12.0, 2400)
    beta = np.exp(-(r / 0.9) ** 2)
    beta /= np.sqrt(np.trapezoid((r * beta) ** 2, r))
    density = 2.0 * (1.0 / (np.pi * 1.3 ** 2)) ** 1.5 * np.exp(-(r / 1.3) ** 2)
    return pp.Pseudopotential("X", 2.0, r, -4.0 * erf(r / 1.1) / r, [(0, r * beta)], np.array([[1.5]]),
                              4.0 * np.pi * r ** 2 * density)


def test_the_confined_pseudo_atom_on_a_bisected_mesh_against_the_radial_equation():
    atom, strength = soft_atom(), 0.4
    reference = pp.radial_levels(atom, 0, 1, strength)[0]
    lattice = 8.0 * np.eye(3)
    uniform = pseudo_atom_levels(CrystalCell(lattice, 8, kinetic_scale=1.0), atom, strength, 1)
    cell = GradedCell(lattice, 8, refinement=IonRefinement(CENTRE, [(2.6, 1)]), kinetic_scale=1.0)
    refined = pseudo_atom_levels(cell, atom, strength, 1)
    assert uniform[2] and refined[2] and refined[1] < 1e-8      # the shift is certified below the spectrum
    assert abs(refined[0][0] - reference) < 0.45 * abs(uniform[0][0] - reference)


def test_a_self_consistent_crystal_on_a_bisected_mesh_against_plane_waves():
    """The Hartree mean field of `MeshCrystal` with the mesh bisected about the
    ion: certified and converged, and nearer the plane-wave levels than the
    uniform mesh it refines."""
    crystal = abinitio.Crystal(LATTICE, [(soft_atom(), np.full(3, 0.5))])
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0)
    target = plane_waves.run(4)["levels"][0][:2] - plane_waves.alignment
    uniform = abinitio.MeshCrystal(crystal, 6).run(4)
    mesh = abinitio.MeshCrystal(crystal, 6, refinement=IonRefinement.of_crystal(crystal, [(2.4, 1)]))
    assert isinstance(mesh.kernel, GradedCoulombKernel)
    run = mesh.run(4)
    assert run["certified"] and run["converged"], run
    assert np.all(np.abs(run["levels"][:2] - target) < 0.6 * np.abs(uniform["levels"][:2] - target))


@pytest.mark.slow
def test_hartree_fock_at_the_zone_centre_and_at_a_finite_momentum_on_a_bisected_mesh():
    """Exchange through the sparse kernel with its zero-momentum constant, the
    pair loads with the links of the true displacements, and the projector loads
    at the true positions: the levels at the zone centre and at X continue the
    sequence of the uniform meshes toward lower energies."""
    crystal = abinitio.Crystal(LATTICE, [(soft_atom(), np.full(3, 0.5))])
    levels = {}
    for name, n, options in (("coarse", 6, {}), ("uniform", 8, {}),
                             ("bisected", 6, {"refinement": IonRefinement.of_crystal(crystal, [(2.4, 1)])})):
        mesh = abinitio.MeshCrystal(crystal, n, **options)
        run = mesh.run_hartree_fock(4)
        assert run["certified"] and run["converged"], (name, run)
        extended = mesh.extend_bands(run, 6)
        at_x = mesh.bands_at(extended, (0.5, 0.0, 0.0))
        assert at_x["converged"]
        levels[name] = np.array([extended["levels"][0], at_x["levels"][0], at_x["levels"][1]])
    assert np.all(levels["bisected"] < levels["uniform"]) and np.all(levels["uniform"] < levels["coarse"])
    assert np.abs(levels["bisected"] - levels["uniform"]).max() < 0.02


@pytest.mark.slow
def test_the_quasiparticle_step_at_the_zone_centre_on_a_bisected_mesh():
    """The random-phase screening with the closed-form head at vanishing
    momentum (the phase derivative of the pencil along the true edge
    displacements, the derivative of the sparse kernel) closes on a bisected
    mesh as it does on the grid, and gives the correction of the grid."""
    from tessera.drivers.bands.settings import Approximations
    crystal = abinitio.Crystal(LATTICE, [(soft_atom(), np.full(3, 0.5))])
    reads = {}
    for name, n, options in (("uniform", 8, {}), ("bisected", 6, {"refinement": IonRefinement.of_crystal(crystal, [(2.4, 1)])})):
        mesh = abinitio.MeshCrystal(crystal, n, approximations=Approximations(1, 1), **options)
        extended = mesh.extend_bands(mesh.run_hartree_fock(4), 10)
        reads[name] = mesh.quasiparticle_levels(extended, [0, 1])
    uniform, bisected = reads["uniform"], reads["bisected"]
    assert bisected["head_defect"] < 1e-10
    assert bisected["dielectric_constant"] - 1.0 == pytest.approx(uniform["dielectric_constant"] - 1.0, rel=0.15)
    for n in (0, 1):
        shift = lambda read: read["states"][n]["quasiparticle"] - read["states"][n]["mean_field"]
        assert shift(bisected) == pytest.approx(shift(uniform), abs=2e-3)
