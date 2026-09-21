# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Two sheets in the ab initio run (#1166): the published relativistic
pseudopotentials as closed forms, L . S on the real harmonics, the spin-orbit
block as a low-rank term of the two-sheet pencil, the pseudo-atom on the mesh
against the radial equation with the j-dependent coefficients, and
Hartree-Fock on two-component sections."""
import numpy as np
import pytest
import scipy.linalg

from tessera.drivers.bands import abinitio, spin
from tessera.drivers.bands import pseudopotential as pp
from tessera.drivers.bands import spinorbit as so


def gaussian_density(valence, width=1.3):
    r = np.linspace(1e-4, 12.0, 2400)
    return r, 4.0 * np.pi * r ** 2 * valence * (1.0 / (np.pi * width ** 2)) ** 1.5 * np.exp(-(r / width) ** 2)


def soft_ion(valence=2.0, coupling=0.2):
    """An ion of the published form that a coarse mesh resolves: s and p
    projectors one bohr wide and a spin-orbit coefficient on the p channel."""
    entry = {"valence": valence, "rloc": 1.1, "C": (-0.4, 0.05),
             "channels": ((0, 0.9, (0.75,), ()), (1, 1.0, (0.5, -0.2), (coupling, 0.15 * coupling)))}
    return so.from_parameters("X", entry, density=gaussian_density(valence))


class TestPublishedForm:
    def test_the_table_holds_the_published_rows_in_rydberg(self):
        gallium, arsenic = so.hgh("Ga", gaussian_density(3.0)), so.hgh("As", gaussian_density(5.0))
        assert (gallium.valence, arsenic.valence) == (3.0, 5.0)
        assert pp.Pseudopotential.from_hgh("As", gaussian_density(5.0)).K == pytest.approx(arsenic.K)
        assert [l for l, _ in arsenic.projectors] == [0, 0, 0, 1, 1, 2]
        assert arsenic.D[0, 0] == pytest.approx(2.0 * 4.560761) and arsenic.K[3, 3] == pytest.approx(2.0 * 0.052466)
        assert gallium.K[4, 4] == pytest.approx(2.0 * -0.000873) and gallium.K[5, 5] == pytest.approx(2.0 * 0.001486)
        assert np.abs(arsenic.K[:3]).max() == 0.0                            # no spin-orbit term at l = 0
        # The off-diagonal entries by the ratios of the paper (its equations 20, 22, 23).
        assert arsenic.D[0, 1] == pytest.approx(-0.5 * np.sqrt(3.0 / 5.0) * 2.0 * 1.692389)
        assert arsenic.D[1, 2] == pytest.approx(-0.5 * np.sqrt(100.0 / 63.0) * 2.0 * -1.373804)
        assert arsenic.K[3, 4] == pytest.approx(-0.5 * np.sqrt(5.0 / 7.0) * 2.0 * 0.020562)
        assert np.abs(arsenic.D - arsenic.D.T).max() == 0.0 and np.abs(arsenic.K - arsenic.K.T).max() == 0.0

    def test_an_element_that_is_not_held_is_refused(self):
        with pytest.raises(ValueError, match="copy its rows"):
            so.hgh("Pu")

    def test_the_radial_functions_are_the_closed_forms(self):
        ion = soft_ion()
        r = np.linspace(1e-5, 14.0, 140001)
        for index, (l, _) in enumerate(ion.projectors):
            assert np.trapezoid((r * ion.projector_at(index, r)) ** 2, r) == pytest.approx(1.0, abs=1e-8)
        far = np.array([15.0, 40.0])
        assert ion.local_at(far) == pytest.approx(-2.0 * ion.valence / far, rel=1e-12)
        # At the origin: -2 Z sqrt(2 / pi) / r_loc + 2 C_1, in rydberg.
        assert ion.local_at(np.array([0.0]))[0] == pytest.approx(-4.0 * np.sqrt(2.0 / np.pi) / 1.1 - 0.8)
        assert np.abs(ion.short_range_at(np.array([9.0, 12.0]), 1.2)).max() < 1e-9
        assert ion.local == pytest.approx(ion.local_at(ion.r)) and ion.projectors[1][1] == pytest.approx(
            ion.r * ion.projector_at(1, ion.r))

    def test_the_default_density_is_the_neutral_hartree_atom(self):
        ion = so.from_parameters("X", {"valence": 2.0, "rloc": 1.1, "C": (), "channels": ((0, 0.9, (0.75,), ()),)})
        assert np.trapezoid(ion.density, ion.r) == pytest.approx(2.0, abs=1e-4)


class TestAngularMomentum:
    @pytest.mark.parametrize("l", [1, 2])
    def test_the_matrices_are_an_angular_momentum_on_the_real_harmonics(self, l):
        L = so.angular_momentum(l)
        for a in range(3):
            assert np.abs(L[a] - L[a].conj().T).max() < 1e-14
            assert np.abs(L[a] @ L[(a + 1) % 3] - L[(a + 1) % 3] @ L[a] - 1j * L[(a + 2) % 3]).max() < 1e-14
        assert sum(M @ M for M in L) == pytest.approx(l * (l + 1) * np.eye(2 * l + 1), abs=1e-14)

    def test_the_matrices_act_as_minus_i_r_cross_grad(self):
        """L_z Y against the derivative of Y along the rotation about z."""
        rng = np.random.default_rng(0)
        points = rng.normal(size=(40, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        angle = 1e-5
        rotation = lambda t: np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
        for l in (1, 2):
            values = np.array(pp.real_harmonics(l, points))                              # (m, points)
            forward = np.array(pp.real_harmonics(l, points @ rotation(angle).T))
            backward = np.array(pp.real_harmonics(l, points @ rotation(-angle).T))
            derivative = (forward - backward) / (2.0 * angle)                            # d / d phi
            assert -1j * derivative == pytest.approx(so.angular_momentum(l)[2].T @ values, abs=1e-8)

    def test_l_dot_s_has_the_two_total_angular_momenta(self):
        assert so.l_dot_s(1) == pytest.approx(spin.l_dot_s(), abs=1e-15)
        for l in (1, 2):
            expected = np.sort([so.expectation(l, l - 0.5)] * (2 * l) + [so.expectation(l, l + 0.5)] * (2 * l + 2))
            assert np.linalg.eigvalsh(so.l_dot_s(l)) == pytest.approx(expected, abs=1e-13)
        with pytest.raises(ValueError):
            so.expectation(1, 2.5)


def test_the_radial_levels_of_the_two_total_angular_momenta_bracket_the_scalar_one():
    strength = 0.1
    shifts = []
    for coupling in (0.2, 0.1):
        ion = soft_ion(coupling=coupling)
        scalar = pp.radial_levels(ion, 1, 1, strength)[0]
        upper, lower = (so.radial_levels(ion, 1, j, 1, strength)[0] for j in (1.5, 0.5))
        assert lower < scalar < upper
        shifts.append((4.0 * upper + 2.0 * lower) / 6.0 - scalar)
    # The centre of gravity moves at second order in the coupling.
    assert shifts[0] / shifts[1] == pytest.approx(4.0, rel=0.1)
    assert so.radial_levels(ion, 0, 0.5, 1, strength)[0] == pytest.approx(pp.radial_levels(ion, 0, 1, strength)[0])


class TestTwoSheetPencil:
    def _mesh(self, n=6):
        crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_ion(), np.full(3, 0.5))])
        return abinitio.MeshCrystal(crystal, n)

    def test_the_core_is_hermitian_traceless_and_couples_the_sheets(self):
        ion = soft_ion()
        core = so.spin_orbit_core([ion, ion])
        rank = 2 * (1 + 3 + 3)
        assert core.shape == (2 * rank, 2 * rank)
        assert np.abs(core - core.conj().T).max() < 1e-15 and abs(np.trace(core)) < 1e-15
        assert np.abs(core[:rank, rank:]).max() > 0.0                       # the attachment block between the sheets
        assert np.abs(core[:7, 7:rank]).max() == 0.0                        # nothing between the two ions
        # On one ion: k (x) L . S, spin the slow index.
        one = so.spin_orbit_core([ion])
        block = one[np.ix_([1, 2, 3, 8, 9, 10], [1, 2, 3, 8, 9, 10])]
        assert block == pytest.approx(ion.K[1, 1] * so.l_dot_s(1), abs=1e-15)

    def test_against_the_dense_two_sheet_pencil(self):
        mesh = self._mesh()
        A = (mesh.stiffness + mesh.cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
        P2, D2 = so.two_sheet_term(mesh.P, mesh.D, so.spin_orbit_core([p for p, _ in mesh.crystal.ions]))
        values, vectors, residual, below = so.solve_two_sheets(A, mesh.mass, P2, D2, 12, -8.0)
        dense = scipy.linalg.block_diag(A.toarray(), A.toarray()) + P2 @ D2 @ P2.conj().T
        mass = scipy.linalg.block_diag(mesh.mass.toarray(), mesh.mass.toarray())
        assert values == pytest.approx(scipy.linalg.eigh(dense, mass, eigvals_only=True)[:12], abs=1e-9)
        assert residual < 1e-9 and below
        assert so.time_reversal_defect(values) < 1e-9                       # Kramers doublets
        # Without the block every level of the sheet appears twice.
        P0, D0 = so.two_sheet_term(mesh.P, mesh.D, np.zeros_like(D2))
        single = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 6, -8.0)[0]
        assert so.solve_two_sheets(A, mesh.mass, P0, D0, 12, -8.0)[0] == pytest.approx(np.repeat(single, 2), abs=1e-9)


def test_the_spin_orbit_splitting_of_the_pseudo_atom_on_the_mesh_tends_to_the_radial_one():
    """The p level of the screened, confined pseudo-atom on two sheets: a
    quartet and a doublet (the mesh, which keeps one threefold axis, splits the
    quartet further at the order of its error). The distance between their
    centres tends to the difference of the radial levels of j = 3/2 and 1/2:
    0.3811, 0.3898, 0.3944 Ry at 12, 16, 20 divisions against 0.4007."""
    from tessera.drivers.bands.crystal import richardson
    ion, side, strength = soft_ion(), 9.0, 0.4
    target = so.radial_levels(ion, 1, 1.5, 1, strength)[0] - so.radial_levels(ion, 1, 0.5, 1, strength)[0]
    spacings, splittings = [], []
    for n in (12, 16):
        levels, residual, below = so.pseudo_atom_levels(ion, side, n, strength, 8)
        assert residual < 1e-8 and below and so.time_reversal_defect(levels) < 1e-9
        assert levels[2] - levels[1] > levels[7] - levels[2]              # s, then the six p levels
        spacings.append(side / n)
        splittings.append(so.multiplet_splitting(levels[2:8], 4, 2))
    errors = np.abs(np.array(splittings) - target)
    assert errors[1] < 0.6 * errors[0] and errors[1] < 0.03 * target
    assert richardson(spacings, splittings)[0] == pytest.approx(target, rel=2e-3)
    # Without the block the six levels are the three of one sheet, twice.
    levels = so.pseudo_atom_levels(ion, side, 8, strength, 8, spin_orbit=False)[0]
    assert levels[0::2] == pytest.approx(levels[1::2], abs=1e-9)


@pytest.mark.slow
def test_the_published_arsenic_splitting_on_the_mesh_approaches_the_radial_one():
    """The published arsenic potential has Gaussians of 0.46 to 0.69 bohr. Its
    levels are far from resolved at these spacings (the s level is 3 eV high at
    0.56 bohr), but the splitting of the p level is within a tenth of the radial
    one and closes under refinement: 579, 481, 491, 501, 507 meV at 0.75, 0.56,
    0.45, 0.375, 0.32 bohr against 514 meV."""
    arsenic, side, strength = so.hgh("As"), 9.0, 0.2
    target = so.radial_levels(arsenic, 1, 1.5, 1, strength)[0] - so.radial_levels(arsenic, 1, 0.5, 1, strength)[0]
    levels = so.pseudo_atom_levels(arsenic, side, 16, strength, 8)[0]
    assert so.multiplet_splitting(levels[2:8], 4, 2) == pytest.approx(target, rel=0.08)


@pytest.fixture(scope="module")
def converged():
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_ion(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 8)
    run = mesh.run_hartree_fock(5)
    assert run["certified"], run
    return mesh, run


class TestHartreeFockOnTwoSheets:
    def test_exchange_of_doubled_orbitals_is_the_one_sheet_exchange_on_each_sheet(self, converged):
        mesh, run = converged
        n = mesh.cell.size
        single = mesh._exchange(run["vectors"][:, :1], run["vectors"])
        start = so.doubled(run)
        both = so._spinor_exchange(mesh, start["vectors"][:, :2], start["vectors"])
        assert both[:n, 0::2] == pytest.approx(single, abs=1e-12) and both[n:, 1::2] == pytest.approx(single, abs=1e-12)
        assert np.abs(both[:n, 1::2]).max() < 1e-14 and np.abs(both[n:, 0::2]).max() < 1e-14

    def test_exchange_is_covariant_under_a_rotation_of_the_spin_frame(self, converged):
        """K[U psi] (U phi) = U K[psi] phi for U in SU(2) acting on the sheets:
        a rotated frame fills the off-diagonal spin blocks of the density matrix."""
        mesh, run = converged
        n = mesh.cell.size
        rng = np.random.default_rng(1)
        spinors = rng.normal(size=(2 * n, 3)) + 1j * rng.normal(size=(2 * n, 3))
        a, b = 0.6 * np.exp(0.3j), 0.8 * np.exp(-1.1j)
        U = np.array([[a, -np.conj(b)], [b, np.conj(a)]])
        rotate = lambda X: np.vstack([U[0, 0] * X[:n] + U[0, 1] * X[n:], U[1, 0] * X[:n] + U[1, 1] * X[n:]])
        plain = so._spinor_exchange(mesh, spinors[:, :2], spinors)
        assert so._spinor_exchange(mesh, rotate(spinors[:, :2]), rotate(spinors)) == pytest.approx(rotate(plain), abs=1e-10)
        overlap = spinors.conj().T @ plain
        assert np.abs(overlap - overlap.conj().T).max() < 1e-10 and np.linalg.eigvalsh(overlap).max() < 0.0

    def test_without_the_block_the_two_sheet_loop_is_the_one_sheet_loop(self, converged):
        mesh, run = converged
        two = so.run_hartree_fock_two_sheets(mesh, 5, run, spin_orbit=False)
        assert two["certified"] and len(two["history"]) == 1
        assert two["levels"][0::2] == pytest.approx(run["levels"], abs=2e-5)
        assert two["levels"][1::2] == pytest.approx(run["levels"], abs=2e-5)

    def test_with_the_block_the_state_stays_time_reversal_invariant_and_an_s_level_does_not_move(self, converged):
        mesh, run = converged
        two = so.run_hartree_fock_two_sheets(mesh, 5, run)
        assert two["certified"] and two["time_reversal_defect"] < 1e-8
        # The first diagonalization is the two-sheet pencil of the converged one-sheet operator; L . S vanishes on s.
        first = two["history_levels"][0]
        assert first[:2] == pytest.approx(np.repeat(run["levels"][:1], 2), abs=2e-5)
        assert np.abs(first[2:] - np.repeat(run["levels"][1:], 2)).max() > 1e-3           # the empty levels do
        # The filled level is s-like, so the mean field does not feel the block. The empty levels move between the
        # first diagonalization and self-consistency: the compressed exchange is exact on the span of the sections it
        # was built from, which the block rotates out of the span of the one-sheet bands.
        assert two["levels"][:2] == pytest.approx(first[:2], abs=5e-4)
        n = mesh.cell.size
        filled = two["vectors"][:, :2]
        density = np.abs(filled[:n]) ** 2 + np.abs(filled[n:]) ** 2
        assert density.sum(axis=1) == pytest.approx(2.0 * run["vectors"][:, 0] ** 2, abs=1e-2 * density.max())
