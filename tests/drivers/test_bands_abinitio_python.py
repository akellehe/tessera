# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ab initio drivers (#1161): the pseudopotential reader, the screened
pseudo-atom on the mesh against the
radial equation, the low-rank shift-invert solve with its inertia certificate,
and a self-consistent crystal on the mesh against plane waves. Every fixture is
synthetic, so no pseudopotential file is needed."""
import functools

import numpy as np
import pytest
import scipy.linalg
from scipy.special import erf

from tessera.drivers.bands import abinitio, coulomb
from tessera.drivers.bands import pseudopotential as pp
from tessera.drivers.bands.crystal import CrystalCell, richardson


def soft_atom(valence=2.0, core=1.1, strength=1.5):
    """A smooth two-electron pseudo-atom: the local potential of a Gaussian
    charge, one s projector, and a Gaussian valence density."""
    r = np.linspace(1e-4, 12.0, 2400)
    local = -2.0 * valence * erf(r / core) / r
    beta = np.exp(-(r / 0.9) ** 2)
    beta /= np.sqrt(np.trapezoid((r * beta) ** 2, r))
    density = valence * (1.0 / (np.pi * 1.3 ** 2)) ** 1.5 * np.exp(-(r / 1.3) ** 2)
    return pp.Pseudopotential("X", valence, r, local, [(0, r * beta)], np.array([[strength]]),
                              4.0 * np.pi * r ** 2 * density)


UPF = """<UPF version="2.0.1">
<PP_HEADER element="Xx" pseudo_type="{kind}" z_valence="2.0" number_of_proj="1"/>
<PP_MESH><PP_R>{r}</PP_R></PP_MESH>
<PP_LOCAL>{local}</PP_LOCAL>
<PP_NONLOCAL><PP_BETA.1 angular_momentum="0">{beta}</PP_BETA.1><PP_DIJ>1.5</PP_DIJ></PP_NONLOCAL>
<PP_RHOATOM>{density}</PP_RHOATOM>
</UPF>"""


class TestReader:
    def test_round_trip_and_refusal(self, tmp_path):
        atom = soft_atom()
        text = lambda values: " ".join(f"{v:.12e}" for v in values)
        path = tmp_path / "soft.upf"
        path.write_text(UPF.format(kind="NC", r=text(atom.r), local=text(atom.local),
                                   beta=text(atom.projectors[0][1]), density=text(atom.density)))
        read = pp.Pseudopotential.from_upf(path)
        assert read.element == "Xx" and read.valence == 2.0 and read.D.shape == (1, 1)
        assert read.local == pytest.approx(atom.local) and read.projectors[0][0] == 0
        assert np.trapezoid(read.density, read.r) == pytest.approx(2.0, rel=1e-6)
        path.write_text(UPF.format(kind="US", r="1", local="1", beta="1", density="1"))
        with pytest.raises(ValueError, match="norm-conserving"):
            pp.Pseudopotential.from_upf(path)

    def test_radial_functions(self):
        atom = soft_atom()
        far = np.array([20.0, 50.0])
        assert atom.local_at(far) == pytest.approx(-4.0 / far)                  # the Coulomb tail
        assert np.abs(atom.short_range_at(np.array([8.0, 11.0]), 1.2)).max() < 1e-6
        assert atom.projector_at(0, np.array([0.0]))[0] == pytest.approx(atom.projector_at(0, np.array([1e-3]))[0], rel=1e-3)
        assert atom.density_at(np.array([0.0]))[0] == pytest.approx(2.0 * (1.0 / (np.pi * 1.69)) ** 1.5, rel=1e-3)


def test_real_harmonics_are_orthonormal():
    nodes, weights = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    cosine, azimuth = np.meshgrid(nodes, phi, indexing="ij")
    sine = np.sqrt(1.0 - cosine ** 2)
    vectors = np.stack([sine * np.cos(azimuth), sine * np.sin(azimuth), cosine], axis=-1).reshape(-1, 3)
    measure = (weights[:, None] * np.full_like(azimuth, 2.0 * np.pi / 48)).ravel()
    functions = [f for l in (0, 1, 2) for f in pp.real_harmonics(l, vectors)]
    gram = np.array([[np.sum(measure * f * g) for g in functions] for f in functions])
    assert gram == pytest.approx(np.eye(9), abs=1e-12)
    assert pp.real_harmonics(1, np.zeros((1, 3)))[0][0] == 0.0


def test_the_radial_solver_on_the_harmonic_oscillator():
    r = np.linspace(1e-4, 12.0, 1200)
    oscillator = pp.Pseudopotential("H", 0.0, r, r ** 2, [], np.zeros((0, 0)), np.zeros_like(r))
    # The box ends inside the tabulated potential (beyond it the Coulomb tail takes over).
    assert pp.radial_levels(oscillator, 0, 2, 0.0, radius=11.0) == pytest.approx([3.0, 7.0], rel=1e-4)
    assert pp.radial_levels(oscillator, 2, 1, 0.0, radius=11.0)[0] == pytest.approx(7.0, rel=1e-4)


class TestLowRankSolve:
    def _pencil(self, n=6):
        cell = CrystalCell.cubic(6.0, n, kinetic_scale=1.0)
        crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
        return abinitio.MeshCrystal(crystal, n)

    def test_against_the_dense_pencil_with_the_projector_added(self):
        mesh = self._pencil()
        A = (mesh.stiffness + mesh.cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
        values, vectors, residual, below = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 5, -8.0)
        dense = A.toarray() + mesh.P @ mesh.D @ mesh.P.T
        reference = scipy.linalg.eigh(dense, mesh.mass.toarray(), eigvals_only=True)[:5]
        assert values == pytest.approx(reference, abs=1e-9) and residual < 1e-9 and below
        assert np.abs(vectors.T @ (mesh.mass @ vectors) - np.eye(5)).max() < 1e-9

    def test_the_shifted_operator_is_the_library_low_rank_update_below_the_crossover(self):
        """The shift-invert operator of the sparse solve, B + P D P^T with
        B = A - sigma M, is `cobordism.LowRankUpdate` on the dense base: the
        factors span the whole change, and its certified Woodbury solve of
        (lambda - sigma) M z returns the eigenvector z of the sparse solve."""
        import tessera
        mesh = self._pencil()
        sigma = -8.0
        A = (mesh.stiffness + mesh.cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
        values, vectors, _, _ = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 3, sigma)
        base = (A - sigma * mesh.mass).toarray().astype(complex)
        n = base.shape[0]
        update = tessera.cobordism.LowRankUpdate(list(base.ravel()), n)
        left, right = (mesh.P @ mesh.D).astype(complex), mesh.P.T.astype(complex)
        update.setUpdate(list(left.ravel()), list(right.ravel()), left.shape[1])
        assert update.spansAffectedChange(list((base + left @ right).ravel()))
        for b in range(3):
            solved = update.solve(list(((values[b] - sigma) * (mesh.mass @ vectors[:, b])).astype(complex)))
            assert solved.certificate.holds()
            assert np.abs(np.array(solved.values) - vectors[:, b]).max() < 1e-8

    def test_the_inertia_certificate_fails_for_a_shift_inside_the_spectrum(self):
        mesh = self._pencil()
        A = (mesh.stiffness + mesh.cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
        lowest = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 2, -8.0)[0]
        _, _, _, below = abinitio.solve_with_projectors(A, mesh.mass, mesh.P, mesh.D, 2, 0.5 * (lowest[0] + lowest[1]))
        assert not below


def test_pulay_mixing_converges_where_plain_mixing_diverges():
    """x -> b - K x with an eigenvalue of K above 1 / step - 1: the plain update
    grows, the subspace update does not."""
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    K = Q @ np.diag([0.1, 0.4, 0.9, 2.0, 4.0, 7.0]) @ Q.T
    b = rng.normal(size=6)
    fixed = np.linalg.solve(np.eye(6) + K, b)
    plain = np.zeros(6)
    for _ in range(40):
        plain = 0.7 * plain + 0.3 * (b - K @ plain)
    assert np.linalg.norm(plain - fixed) > 1e3
    mixer, x = abinitio.PulayMixer(0.3, history=8), np.zeros(6)
    for _ in range(12):
        x = mixer.next(x, b - K @ x)
    assert np.linalg.norm(x - fixed) < 1e-8


def test_a_self_consistent_crystal_on_the_mesh_matches_plane_waves():
    """One soft two-electron ion in a simple cubic cell at the zone centre: the
    same ionic potential and Hartree kernel convention in both codes, in the
    Hartree mean field. The mesh levels extrapolate to the plane-wave levels."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    assert crystal.electrons == 2
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0)
    reference = plane_waves.run(4)
    assert reference["converged"]
    spacings, levels = [], []
    for n in (8, 12, 16):
        run = abinitio.MeshCrystal(crystal, n).run(4)
        assert run["certified"], run
        spacings.append(run["spacing"])
        levels.append(run["levels"])
    levels = np.array(levels)
    # The continuum kernel leaves a uniform remainder of the ionic charges that the mesh kernel does not carry.
    target = reference["levels"][0] - plane_waves.alignment
    coarse_error = np.abs(levels[-1][:2] - target[:2]).max()
    extrapolated = richardson(spacings, levels)[0]
    assert coarse_error > 5e-3
    assert np.abs(extrapolated[:2] - target[:2]).max() < 3e-3            # rydberg
    assert (extrapolated[1] - extrapolated[0]) == pytest.approx(target[1] - target[0], abs=3e-3)


@pytest.mark.slow
def test_hartree_fock_on_the_mesh_matches_hartree_fock_in_plane_waves():
    """The mean field of the Coulomb interaction. Two independent implementations of the same
    Hartree-Fock problem (exchange compressed onto the computed bands, the
    zero-momentum term restored by the probe-charge correction, the projector
    as a low-rank term): the mesh levels extrapolate to the plane-wave levels,
    filled and empty."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0)
    reference = plane_waves.run_hartree_fock(4, 6.0)
    assert reference["converged"]
    spacings, levels = [], []
    for n in (8, 12, 16):
        run = abinitio.MeshCrystal(crystal, n).run_hartree_fock(4)
        assert run["certified"], run
        spacings.append(run["spacing"])
        levels.append(run["levels"][:2])
    target = reference["levels"][0][:2] - plane_waves.alignment
    assert np.abs(np.array(levels[-1]) - target).max() > 1e-3
    assert np.abs(richardson(spacings, levels)[0] - target).max() < 1e-3           # rydberg
    # Exchange opens the gap well beyond that of the Hartree mean field.
    hartree = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0).run(4)["levels"][0]
    assert target[1] - target[0] > 1.3 * (hartree[1] - hartree[0])


def test_the_crystal_quasiparticle_step_agrees_with_the_full_tensor_route():
    """`quasiparticle_levels` forms only the Coulomb integrals it needs; on a
    crystal small enough to hold the full four-index tensor the two routes give
    the same quasiparticle levels, and the larger band set keeps the filled
    Hartree-Fock level where the smaller one put it."""
    from tessera.drivers.bands import coulomb, screening
    from tessera.drivers.bands.settings import Approximations
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 8, approximations=Approximations(1, 1))      # the orders that the full tensor has
    mean_field = mesh.run_hartree_fock(4)
    extended = mesh.extend_bands(mean_field, 10)
    assert mean_field["certified"] and extended["converged"] and extended["shift_below_spectrum"]
    assert extended["levels"][0] == pytest.approx(mean_field["levels"][0], abs=1e-6)
    assert extended["levels"][1] == pytest.approx(mean_field["levels"][1], abs=2e-4)

    lean = mesh.quasiparticle_levels(extended, [0, 1])
    T = coulomb.pair_densities(mesh.cell.complex, mesh.cell.squared_lengths, extended["vectors"])
    full = screening.RandomPhase(extended["levels"], coulomb.ModeInteraction(mesh.kernel, extended["levels"], T).W, 1)
    for n in (0, 1):
        energy, _ = full.quasiparticle(n)
        assert lean["states"][n]["body"] == pytest.approx(energy, abs=1e-9)
        assert lean["states"][n]["defect"] < 1e-9 and 0.5 < lean["states"][n]["renormalization"] < 1.0
    assert lean["correlation_energy"] == pytest.approx(full.correlation_energy(), abs=1e-10)
    # The zero-momentum constant is the kernel's own; the continuum kernel's is 2 MADELUNG / L.
    assert lean["head_constant"] == pytest.approx(mesh.kernel.zero_momentum_constant())
    assert lean["head_constant"] == pytest.approx(2.0 * coulomb.MADELUNG_SC / 6.0, rel=5e-3)
    # Without the zero-momentum term the correlation part moves this gap by a few per cent only.
    # Screening closes a Hartree-Fock gap through that term, by c (1 - 1/eps) on the energy shell.
    gap = lambda key: lean["states"][1][key] - lean["states"][0][key]
    assert abs(gap("body") / gap("mean_field") - 1.0) < 0.05
    eps, c = lean["dielectric_constant"], lean["head_constant"]
    # Local fields lower the dielectric constant below its independent-particle value.
    assert 1.0 < eps < lean["independent_particle_dielectric_constant"]
    assert lean["head_defect"] < 1e-10
    assert gap("quasiparticle") < gap("body")
    assert gap("body") - gap("quasiparticle") == pytest.approx(c * (1.0 - 1.0 / eps), rel=0.25)


def test_the_converged_crystal_is_a_stationary_covariance_state():
    """The Hartree-Fock state of the crystal as a `CovarianceState` on its own
    modes: pure, with the right particle number, and stationary under
    `meanFieldEvolve` with the Fock operator rebuilt as the Wick contraction of
    the Coulomb kernel, which reproduces the levels the pencil was solved with."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 8)
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-8), 8, tolerance=1e-8)
    read = mesh.covariance_certificate(extended, 8)
    assert read["purity_defect"] < 1e-12 and read["particles"] == pytest.approx(2.0, abs=1e-12)
    assert read["fock_defect"] < 1e-7 and read["stationarity_defect"] < 1e-7


def test_exchange_at_zero_momentum_transfer_is_the_zone_centre_exchange():
    """The contract a momentum set has to keep: the exchange operator with the
    momentum transfer in its kernel, at zero transfer, is the zone-centre
    operator, the entry at G = 0 included; and at a transfer q it is Hermitian
    and continuous in q."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    mesh = abinitio.MeshCrystal(crystal, 6)
    run = mesh.run(4)
    filled, orbitals = run["vectors"][:, :1], run["vectors"]
    centre = mesh._exchange(filled, orbitals)
    assert np.abs(mesh._exchange(filled, orbitals, (0.0, 0.0, 0.0)) - centre).max() < 1e-12
    trial = orbitals.astype(complex)
    near = trial.conj().T @ mesh._exchange(filled, trial, (1e-4, 0.0, 0.0))
    assert np.abs(near - near.conj().T).max() < 1e-12
    assert np.abs(near - orbitals.T @ centre).max() < 1e-5


def test_hartree_fock_on_a_momentum_set_is_hartree_fock_of_the_supercell():
    """A cell doubled along an axis and sampled at its zone centre is the single
    cell sampled at 0 and 1/2 along that axis: the same vertices, the same
    tetrahedra, the same filled determinant. The Hartree-Fock levels agree,
    filled and empty, which holds the exchange kernel at a momentum transfer,
    the pair loads between two momenta, and the zero-momentum constant of the
    set (that of the supercell) to the zone-centre code."""
    atom = soft_atom()
    single = abinitio.Crystal(6.0 * np.eye(3), [(atom, np.full(3, 0.5))])
    double = abinitio.Crystal(np.diag([12.0, 6.0, 6.0]), [(atom, np.array([0.25, 0.5, 0.5])),
                                                           (atom, np.array([0.75, 0.5, 0.5]))])
    supercell = abinitio.MeshCrystal(double, (12, 6, 6))
    reference = supercell.run_hartree_fock(8, tolerance=1e-8)
    mesh = abinitio.MeshCrystal(single, 6)
    run = mesh.run_hartree_fock_set(4, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], tolerance=1e-8)
    assert reference["certified"] and run["certified"]
    assert run["zero_momentum"] == pytest.approx(supercell.zero_momentum, abs=1e-6)
    union = np.sort(np.concatenate(run["levels"]))
    assert np.abs(reference["levels"][:4] - union[:4]).max() < 1e-5          # rydberg


def test_the_kinetic_eigenbasis_route_is_the_pole_exact_route():
    """Two routes to the correlation self-energy of the crystal, zero-momentum
    term included: the random-phase modes of the particle-hole pairs (poles in
    closed form), and the dielectric matrix in the eigenbasis of the kinetic
    pencil, inverted at imaginary frequencies, with the self-energy by contour
    deformation. With the whole basis they agree at every frequency, on and off
    the levels; with a truncated basis the second converges to the first."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    mesh = abinitio.MeshCrystal(crystal, 6)
    bands = 7
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 3, tolerance=1e-9)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    limits = mesh.vanishing_momentum_pairs(extended, coupling, bands)
    rpa.set_head(mesh.zero_momentum, limits)
    values, basis = scipy.linalg.eigh(mesh.stiffness.toarray(), mesh.mass.toarray())
    values, basis = values[1:], basis[:, 1:]                       # without the constant, where the kernel has no entry
    modes = extended["vectors"][:, :bands]
    pairs = np.vstack([(basis.T @ coulomb.pair_loads(mesh.cell, modes[:, i], modes[:, occupied:])).T for i in range(occupied)])
    kernel = abinitio.COULOMB_STRENGTH / values
    assert np.abs((pairs * kernel) @ pairs.T - coupling).max() < 1e-12
    head = [(np.imag(limit["charges"]), limit["entry"]) for limit in limits]      # real modes: the charges are i r
    frequencies = (levels[0] - 0.3, levels[0], 0.5 * (levels[0] + levels[1]), levels[1], levels[1] + 0.4)
    errors = []
    for size in (8, 30, len(values)):
        route = screening.KineticBasisScreening(levels, occupied, pairs[:, :size], kernel[:size], head,
                                                mesh.zero_momentum)
        errors.append(max(abs(route.correlation(n, (basis[:, :size].T @ coulomb.pair_loads(mesh.cell, modes[:, n], modes)).T, w)
                              - rpa.correlation(n, w)[0]) for n in (0, 1) for w in frequencies))
    assert errors[2] < 1e-9 and errors[2] < errors[1] < errors[0], errors


def test_quasiparticle_levels_agree_between_the_two_routes_with_a_sparse_kinetic_basis():
    """The production form of the second route: the kinetic basis from the
    sparse solver, truncated, and the quasiparticle equation solved on it. The
    levels converge to those of the pole-exact route."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    mesh = abinitio.MeshCrystal(crystal, 8)
    bands = 7
    extended = mesh.extend_bands(mesh.run_hartree_fock(4), bands + 3)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    limits = mesh.vanishing_momentum_pairs(extended, coupling, bands)
    rpa.set_head(mesh.zero_momentum, limits)
    head = [(np.imag(limit["charges"]), limit["entry"]) for limit in limits]
    reference = np.array([rpa.quasiparticle(n)[0] for n in (0, 1)])
    errors = []
    for size in (30, 120, 360):
        values, basis = mesh.kinetic_basis(size)
        assert values[0] > 1e-6 and np.abs(basis.T @ (mesh.mass @ basis) - np.eye(size)).max() < 1e-8
        pairs, states = mesh.basis_coefficients(extended, basis, bands, (0, 1))
        route = screening.KineticBasisScreening(levels, occupied, pairs, abinitio.COULOMB_STRENGTH / values, head,
                                                mesh.zero_momentum)
        errors.append(np.abs(np.array([route.quasiparticle(n, states[n])[0] for n in (0, 1)]) - reference).max())
    assert errors[2] < errors[1] < errors[0] and errors[2] < 2e-5, errors       # rydberg


@pytest.mark.parametrize("kappa", [None, (0.2, -0.1, 0.35)])
def test_the_kinetic_eigenbasis_of_the_grid_is_closed_form(kappa):
    """On the periodic grid the kinetic pencil at any crystal momentum is
    diagonal in lattice plane waves, with the eigenvalues a(G + k) / m(G + k)
    from the symbols of the stiffness and mass matrices: the dense spectrum, and
    the Coulomb energy of a load from its coefficients in those modes."""
    from tessera.drivers.bands import coulomb
    from tessera.drivers.bands.crystal import CrystalCell
    cell = CrystalCell(np.array([[5.0, 0.3, 0.0], [0.0, 6.0, 0.2], [0.1, 0.0, 7.0]]), (4, 5, 3), kinetic_scale=1.0)
    kernel = coulomb.GridCoulombKernel(cell, 8.0 * np.pi)
    A, M = cell.pencil(kappa or (0.0, 0.0, 0.0))
    dense = scipy.linalg.eigh(A.toarray(), M.toarray(), eigvals_only=True)
    values, indices, mass = kernel.kinetic_modes(20, kappa)
    assert np.abs(values - (dense[1:21] if kappa is None else dense[:20])).max() < 1e-12
    rng = np.random.default_rng(0)
    load = rng.standard_normal((cell.size, 2)) + 1j * rng.standard_normal((cell.size, 2))
    load -= load.mean(axis=0) if kappa is None else 0.0           # the zone-centre kernel has no entry on the constant
    values, indices, mass = kernel.kinetic_modes(cell.size, kappa)
    coefficients = kernel.mode_coefficients(load, indices, mass)
    energy = np.einsum("mi,m,mi->i", coefficients.conj(), kernel.strength / values, coefficients).real
    exact = np.einsum("vi,vi->i", load.conj(), kernel.potential(load, kappa)).real
    assert np.abs(energy - exact).max() < 1e-12 * np.abs(exact).max()


def test_a_converged_run_on_a_coarse_mesh_starts_the_next_mesh():
    """The orbitals of a coarse mesh are piecewise-linear functions; evaluated at
    the vertices of a mesh of twice the divisions they are the same functions
    (equal at the shared vertices, the same norm in the finer mass matrix), and
    started from them the finer run reaches the levels and the energy of a run
    from scratch in fewer solves of the pencil, the Hartree loop that a run from
    scratch starts with being what it saves."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    coarse, fine = abinitio.MeshCrystal(crystal, 6), abinitio.MeshCrystal(crystal, 12)
    run = coarse.run_hartree_fock(4)
    start = fine.prolonged(coarse, run)
    shared = (fine.cell.index % 2 == 0).all(axis=1)
    ids = (fine.cell.index[shared] // 2) @ np.array([36, 6, 1])
    assert np.abs(start["vectors"][shared] - run["vectors"][ids]).max() < 1e-14
    assert np.abs(np.diag(start["vectors"].T @ (fine.mass @ start["vectors"])) - 1.0).max() < 1e-10
    scratch, continued = fine.run_hartree_fock(4), fine.run_hartree_fock(4, start=start)
    assert continued["certified"] and continued["solves"] < scratch["solves"]
    assert np.abs(scratch["levels"] - continued["levels"]).max() < 1e-5
    assert continued["energy"] == pytest.approx(scratch["energy"], abs=1e-6)


@functools.lru_cache(maxsize=1)
def _doubled_cell_and_its_momentum_set():
    """A cell doubled along one axis at its zone centre, and the single cell on
    the momentum set {0, 1/2} that is equivalent to it, both with more bands
    than are filled and the same states kept in both (up to a gap of the
    spectrum)."""
    atom = soft_atom()
    single = abinitio.Crystal(6.0 * np.eye(3), [(atom, np.full(3, 0.5))])
    double = abinitio.Crystal(np.diag([12.0, 6.0, 6.0]), [(atom, np.array([0.25, 0.5, 0.5])),
                                                           (atom, np.array([0.75, 0.5, 0.5]))])
    supercell = abinitio.MeshCrystal(double, (12, 6, 6))
    reference = supercell.extend_bands(supercell.run_hartree_fock(8, tolerance=1e-8), 20, tolerance=1e-8)
    mesh = abinitio.MeshCrystal(single, 6)
    extended = mesh.extend_bands_set(mesh.run_hartree_fock_set(4, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], tolerance=1e-8),
                                     10, tolerance=1e-8)
    assert extended["converged"]
    union = np.sort(np.concatenate(extended["levels"]))
    assert np.abs(union[:14] - reference["levels"][:14]).max() < 1e-5
    cut = 5 + int(np.argmax(np.diff(reference["levels"][:16])[5:13]))
    threshold = 0.5 * (reference["levels"][cut] + reference["levels"][cut + 1])
    kept = [int(np.sum(levels < threshold)) for levels in extended["levels"]]
    assert sum(kept) == cut + 1
    return supercell, reference, mesh, extended, cut, kept


@pytest.mark.slow
def test_the_quasiparticle_equation_on_a_momentum_set_is_that_of_the_supercell():
    """The screened interaction at every momentum transfer of the set, pairs
    running from k to k + q, and the self-energy summed over the transfers: on
    the set {0, 1/2} of a cell they are the zone-centre calculation of the cell
    doubled along that axis, state by state; first without the entry of the
    kernel at zero transfer and G = 0, then with it (`RandomPhase.set_head` with
    the closed-form charges of the doubled cell against the charges of the
    pairs of zero transfer at both momenta of the set)."""
    from tessera.drivers.bands import screening
    supercell, reference, mesh, extended, cut, kept = _doubled_cell_and_its_momentum_set()
    levels, occupied, coupling, integrals = supercell.coulomb_integrals(reference, cut + 1)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    states = [(0, 0), (1, 0), (1, 1)]                                   # the three lowest modes of the supercell
    produced = mesh.quasiparticle_set(extended, states, kept, head=False)
    body = {}
    for n, state in enumerate(states):
        body[state] = rpa.quasiparticle(n)[0]
        assert produced[state][0] == pytest.approx(levels[n], abs=1e-5)
        assert produced[state][1] == pytest.approx(body[state], abs=1e-5)
        assert abs(produced[state][1] - produced[state][0]) > 1e-3       # the correction being compared is not small
    rpa.set_head(supercell.zero_momentum, supercell.vanishing_momentum_pairs(reference, coupling, cut + 1))
    assert extended["zero_momentum"] == pytest.approx(supercell.zero_momentum, abs=1e-9)
    produced = mesh.quasiparticle_set(extended, states, kept)
    for n, state in enumerate(states):
        with_head = rpa.quasiparticle(n)[0]
        assert produced[state][1] == pytest.approx(with_head, abs=1e-5)
        assert abs(with_head - body[state]) > 1e-3                       # the entry being compared is not small


@pytest.mark.slow
@pytest.mark.parametrize("order", [2, 3])
def test_the_zero_momentum_order_on_a_momentum_set_is_that_of_the_supercell(order):
    """The offsets of `zero_momentum_order` k around every transfer of the set
    {0, 1/2} are the midpoint grid of the doubled cell around its zone centre,
    so the averaged self-energy integrand is the same, state by state; and at
    order 1 the constant left to the zero-momentum term is the constant of the
    set."""
    from tessera.drivers.bands import screening
    from tessera.drivers.bands.momentum_set import SetScreening, set_nodes
    from tessera.drivers.bands.settings import Approximations
    supercell, reference, mesh, extended, cut, kept = _doubled_cell_and_its_momentum_set()
    settings = Approximations(1, order)
    levels, occupied, coupling, integrals = supercell.coulomb_integrals(reference, cut + 1)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    rpa.set_head(supercell.zero_momentum, supercell.vanishing_momentum_pairs(reference, coupling, cut + 1))
    first = [rpa.quasiparticle(n)[0] for n in range(3)]
    threshold = 0.5 * (reference["levels"][cut] + reference["levels"][cut + 1])
    screened = SetScreening(mesh, extended, threshold=threshold, nodes=set_nodes(settings, extended["momenta"]))
    # The same states in both: at a moved momentum the levels below the cut are fewer than at the zone centre.
    terms = [supercell.momentum_term(reference, kappa, range(3), sum(kept_there))
             for (kappa, _), (_, kept_there) in zip(settings.momentum_nodes, screened.moved)]
    assert all(term["converged"] for term in terms)
    rpa.set_momentum_terms(terms, [weight for _, weight in settings.momentum_nodes],
                           supercell.zero_momentum + supercell.kernel.auxiliary_function())
    for n, state in enumerate([(0, 0), (1, 0), (1, 1)]):
        expected = rpa.quasiparticle(n)[0]
        assert screened.quasiparticle(state)[1] == pytest.approx(expected, abs=2e-5)
        assert abs(expected - first[n]) > 1e-3                          # the order being compared is not small
    plain = SetScreening(mesh, extended, kept)
    average = mesh.zero_momentum + mesh.kernel.auxiliary_function()
    sampled = (mesh.kernel.auxiliary_function() + mesh.kernel.auxiliary_function((0.5, 0.0, 0.0))) / 2.0
    assert plain.effective_constant == pytest.approx(average - sampled, abs=1e-8)


@pytest.mark.slow
def test_the_diagrams_beyond_the_first_order_on_a_momentum_set_are_those_of_the_supercell():
    """The states of the set are the modes of the doubled cell, and a mode of
    the screened interaction of momentum q, whose couplings are complex, enters
    the engine as two bosons with Hermitian couplings. At first order the
    engine then gives the self-energy the set computes in closed form, and
    with the crossed diagram of second order the quasiparticle levels are
    those of the doubled cell, state by state."""
    from tessera.drivers.bands import screening
    from tessera.drivers.bands.momentum_set import SetScreening
    from tessera.drivers.bands.settings import Approximations
    supercell, reference, mesh, extended, cut, kept = _doubled_cell_and_its_momentum_set()
    states = [(0, 0), (1, 0), (1, 1)]
    levels, occupied, coupling, integrals = supercell.coulomb_integrals(reference, cut + 1)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    rpa.set_head(supercell.zero_momentum, supercell.vanishing_momentum_pairs(reference, coupling, cut + 1))
    first = [rpa.quasiparticle(n)[0] for n in range(3)]
    saved = supercell.approximations
    try:
        supercell.approximations = Approximations(2, 1, vertex_bands=cut + 1, vertex_poles=len(rpa.excitations))
        rpa.set_vertex(*supercell.vertex(reference, cut + 1), states=range(3))
    finally:
        supercell.approximations = saved
    screened = SetScreening(mesh, extended, kept)
    screened.set_vertex(2, cut + 1, len(rpa.excitations), states)
    assert len(screened.vertex) == cut + 1
    # First order through the engine against the closed form of the set, away from the poles.
    screened._vertex(states[0], 0.0, screened.mean_field)
    _, chemical_potential, engines = screened._engines
    for state in states:
        frequency = screened.mean_field[state[0]][state[1]] + 0.013
        terms = screened._terms(state, screened.mean_field)[:-len(screened.residues)]
        closed = sum(np.sum(w / (frequency - p)) for w, p in terms)
        through = np.mean([engine.evaluate(screened.vertex.index(state), frequency - chemical_potential, 1)
                           for engine in engines])
        assert abs(through.imag) < 1e-10 and through.real == pytest.approx(closed, abs=1e-9)
    for n, state in enumerate(states):
        expected = rpa.quasiparticle(n)[0]
        assert screened.quasiparticle(state)[1] == pytest.approx(expected, abs=2e-5)
        assert abs(expected - first[n]) > 1e-4                          # the diagram being compared is not small


@pytest.mark.slow
def test_eigenvalue_self_consistency_on_a_momentum_set_is_that_of_the_supercell():
    """The levels fed back into the propagator (GW0), and into the propagator
    and the screening (evGW), with the zero-transfer entry: on the set {0, 1/2}
    they are those of the doubled cell, every kept state."""
    from tessera.drivers.bands import screening
    from tessera.drivers.bands.momentum_set import SetScreening
    supercell, reference, mesh, extended, cut, kept = _doubled_cell_and_its_momentum_set()
    levels, occupied, coupling, integrals = supercell.coulomb_integrals(reference, cut + 1)
    heads = supercell.vanishing_momentum_pairs(reference, coupling, cut + 1)
    screened = SetScreening(mesh, extended, kept)
    for update in (False, True):
        expected, _, _ = screening.self_consistent_quasiparticles(
            levels, occupied, coupling, integrals, head=(supercell.zero_momentum, heads), update_screening=update,
            tolerance=1e-7)
        produced, history = screened.self_consistent(update_screening=update, tolerance=1e-7)
        screened.solve()
        assert history[-1] < 1e-7
        union = np.sort(np.concatenate(produced))
        assert np.abs(union - np.sort(expected)).max() < 1e-5
        assert np.abs(expected - levels).max() > 0.1                  # the corrections compared are not small


def test_a_pair_density_of_small_momentum_is_loaded_with_the_link_phases_of_that_momentum():
    """The load of conj(psi_i) psi_a for a section psi_a of crystal momentum
    kappa is the weighted mass matrix M_0^U[psi_i], dressed by the flat
    connection of that momentum, applied to the cell-periodic part; and the
    Coulomb kernel at that momentum inverts the dressed stiffness matrix."""
    from tessera.drivers.bands import coulomb
    from tessera.drivers.bands.crystal import CrystalCell
    cell = CrystalCell(np.array([[1.0, 0.1, 0.0], [0.0, 1.1, 0.05], [0.02, 0.0, 0.9]]), (4, 5, 3), kinetic_scale=1.0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(cell.size)
    Y = rng.standard_normal((cell.size, 3)) + 1j * rng.standard_normal((cell.size, 3))
    kappa = np.array([0.013, -0.2, 0.31])
    loads = coulomb.pair_loads(cell, x, Y, kappa)
    assert np.abs(loads - cell.weighted_mass(x).dressed(kappa) @ Y).max() < 1e-15
    kernel = coulomb.GridCoulombKernel(cell, 8.0 * np.pi)
    rho = Y[:, 0]
    assert np.abs(cell.stiffness.dressed(kappa) @ kernel.potential(rho, kappa) / kernel.strength - rho).max() < 1e-12
    # The entry at G = 0 tends to strength / (V q^2).
    small = np.array([1e-3, 0.0, 0.0])
    continuum = kernel.strength / (cell.volume * (cell.momentum(small) ** 2).sum())
    assert kernel.momentum_entry(small) == pytest.approx(continuum, rel=1e-5)


def test_the_zero_momentum_constant_of_the_mesh_kernel_tends_to_the_madelung_constant():
    from tessera.drivers.bands import coulomb
    from tessera.drivers.bands.crystal import CrystalCell
    side, target = 6.0, 2.0 * coulomb.MADELUNG_SC / 6.0
    errors = [abs(coulomb.GridCoulombKernel(CrystalCell.cubic(side, n, kinetic_scale=1.0), 8.0 * np.pi)
                  .zero_momentum_constant() - target) for n in (6, 12)]
    assert errors[1] < 0.3 * errors[0] < 0.01 * target


def test_the_closed_form_response_at_vanishing_momentum_is_the_limit_of_small_momenta():
    """`vanishing_momentum_pairs` differentiates the pair charge with respect to
    the momentum in closed form, exchange included; `bands_at` and
    `momentum_pairs` solve the Hartree-Fock pencil at a small momentum. The
    second converges to the first at second order, along every axis, for an ion
    placed off the symmetric positions; and the entry of the kernel at G = 0
    tends to strength / (V q^2) exactly."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.array([0.4, 0.45, 0.55]))])
    bands = 7
    mesh = abinitio.MeshCrystal(crystal, 8)
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 4, tolerance=1e-9)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    limits = mesh.vanishing_momentum_pairs(extended, coupling, bands)
    for axis in (0, 2):
        assert limits[axis]["entry"] == pytest.approx(abinitio.COULOMB_STRENGTH / crystal.volume, rel=1e-12)
        exact = rpa.set_head(mesh.zero_momentum, [limits[axis]]) - 1.0
        assert rpa.head_defect < 1e-12
        errors = []
        for delta in (0.02, 0.01):
            kappa = tuple(delta if a == axis else 0.0 for a in range(3))
            at_momentum = mesh.bands_at(extended, kappa, converge=bands, tolerance=1e-9)
            errors.append(rpa.set_head(mesh.zero_momentum, [mesh.momentum_pairs(extended, at_momentum, bands)]) - 1.0 - exact)
        assert abs(errors[0]) < 0.01 * exact and errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)


@pytest.mark.slow
def test_the_dielectric_response_at_vanishing_momentum_matches_plane_waves():
    """The independent-particle dielectric constant from the closed-form pair
    charges converges to that of plane waves, where the overlaps between the
    zone centre and a small momentum are exact."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    bands = 7                                   # closed at a gap, so every mesh and the plane waves hold the same states
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0)
    reference = plane_waves.dielectric_constant(plane_waves.run_hartree_fock(bands, 6.0),
                                                [0.002 * 2.0 * np.pi / 6.0, 0.0, 0.0]) - 1.0
    spacings, values = [], []
    for n in (8, 12, 16):
        mesh = abinitio.MeshCrystal(crystal, n)
        extended = mesh.extend_bands(mesh.run_hartree_fock(4), bands + 4)
        levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
        rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
        rpa.set_head(mesh.zero_momentum, mesh.vanishing_momentum_pairs(extended, coupling, bands))
        assert rpa.head_defect < 1e-10 and rpa.dielectric_constant < rpa.independent_particle_dielectric_constant
        spacings.append(mesh.cell.spacing)
        values.append(rpa.independent_particle_dielectric_constant - 1.0)
    assert values[0] < values[1] < values[2] < reference
    assert richardson(spacings, values)[0] == pytest.approx(reference, rel=0.01)


@pytest.mark.slow
def test_a_momentum_set_is_solved_on_one_momentum_of_every_orbit_of_the_operations_the_mesh_keeps():
    """Time reversal and the permutations of the axes carry the periodic Kuhn
    grid onto itself; for a crystal that has them the sections at a momentum
    are the images of those at another. Hartree-Fock on the 2 x 2 x 2 set with
    four momenta solved is the one with all eight solved (an insulating cell:
    in the 6 bohr cell of the other fixtures the filled band at the zone corner
    lies above the empty one at the face centre, and Hartree-Fock with one
    filled band per momentum breaks the symmetry). A crystal without the
    permutations keeps time reversal only."""
    from tessera.drivers.bands.momentum_set import SetSymmetry, uniform_set
    atom = soft_atom()
    mesh = abinitio.MeshCrystal(abinitio.Crystal(8.0 * np.eye(3), [(atom, np.full(3, 0.5))]), 6)
    momenta = uniform_set(2)
    orbits = SetSymmetry(mesh, momenta)
    assert len(orbits.operations) == 12 and len(orbits.representatives) == 4
    reduced = mesh.run_hartree_fock_set(2, momenta, tolerance=1e-8)
    full = mesh.run_hartree_fock_set(2, momenta, tolerance=1e-8, symmetry=False)
    assert reduced["converged"] and full["converged"]
    assert reduced["solved"] == orbits.representatives and len(full["solved"]) == 8
    for k in range(8):
        assert np.abs(reduced["levels"][k] - full["levels"][k]).max() < 1e-10
        # The filled section is the same line of the same pencil.
        M = mesh.cell.pencil(momenta[k])[1]
        assert abs(np.vdot(reduced["vectors"][k][:, 0], M @ full["vectors"][k][:, 0])) == pytest.approx(1.0, abs=1e-8)
    extended = mesh.extend_bands_set(reduced, 4, tolerance=1e-8)
    unreduced = mesh.extend_bands_set(full, 4, tolerance=1e-8)
    assert max(np.abs(a - b).max() for a, b in zip(extended["levels"], unreduced["levels"])) < 1e-7
    skew = abinitio.MeshCrystal(abinitio.Crystal(6.0 * np.eye(3), [(atom, np.array([0.5, 0.4, 0.3]))]), 6)
    orbits = SetSymmetry(skew, uniform_set(3))
    assert len(orbits.operations) == 2 and len(orbits.representatives) == 14
    assert sorted({orbits.images[k][0] for k in range(27)}) == orbits.representatives
