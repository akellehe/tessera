# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ab initio drivers (#1161): the pseudopotential reader, the screened
pseudo-atom on the mesh against the
radial equation, the low-rank shift-invert solve with its inertia certificate,
and a self-consistent crystal on the mesh against plane waves. Every fixture is
synthetic, so no pseudopotential file is needed."""
import numpy as np
import pytest
import scipy.linalg
from scipy.special import erf

from tessera.drivers.bands import abinitio
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
    reference = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0).run(4)
    assert reference["converged"]
    spacings, levels = [], []
    for n in (8, 12, 16):
        run = abinitio.MeshCrystal(crystal, n).run(4)
        assert run["certified"], run
        spacings.append(run["spacing"])
        levels.append(run["levels"])
    levels = np.array(levels)
    target = reference["levels"][0]
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
    reference = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0).run_hartree_fock(4, 6.0)
    assert reference["converged"]
    spacings, levels = [], []
    for n in (8, 12, 16):
        run = abinitio.MeshCrystal(crystal, n).run_hartree_fock(4)
        assert run["certified"], run
        spacings.append(run["spacing"])
        levels.append(run["levels"][:2])
    target = reference["levels"][0][:2]
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
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 8)
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
    triple = coulomb.TripleIntegrals(cell.complex, cell.squared_lengths)
    loads = triple.loads(x, Y, coulomb.bloch_twist(cell, triple.tops, kappa))
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
