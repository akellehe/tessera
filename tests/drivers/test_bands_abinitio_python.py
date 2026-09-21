# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The ab initio drivers (#1161): the pseudopotential reader, the local-density
exchange and correlation, the screened pseudo-atom on the mesh against the
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


class TestExchangeCorrelation:
    def test_the_potential_is_the_derivative_of_the_energy(self):
        for density in (1e-3, 0.02, 0.2387, 0.24, 3.0):          # both branches of the correlation fit
            step = 1e-6 * density
            energy = lambda n: n * pp.lda_energy_density(n)
            numerical = (energy(density + step) - energy(density - step)) / (2 * step)
            assert pp.lda_potential(density) == pytest.approx(numerical, rel=1e-6)

    def test_exchange_at_unit_wigner_seitz_radius(self):
        density = 3.0 / (4.0 * np.pi)
        # e_x = -0.9163 Ry / r_s; e_c(r_s = 1) = -0.1423 / (1 + 1.0529 + 0.3334) hartree.
        expected = -0.91633 + 2.0 * (-0.1423 / 2.3863)
        assert pp.lda_energy_density(density) == pytest.approx(expected, rel=1e-4)


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
    same ionic potential, Hartree kernel convention and exchange-correlation in
    both codes. The mesh levels extrapolate to the plane-wave levels."""
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
    # Exchange without correlation opens the gap well beyond the local-density one.
    local_density = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0).run(4)["levels"][0]
    assert target[1] - target[0] > 1.3 * (local_density[1] - local_density[0])


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
    assert lean["head_constant"] == pytest.approx(2.0 * coulomb.MADELUNG_SC / 6.0)
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


@pytest.mark.slow
def test_the_dielectric_response_at_vanishing_momentum_matches_plane_waves():
    """The dipoles of the particle-hole pairs come from the current operator of
    the pencil; the independent-particle dielectric constant they give converges
    to the plane-wave value computed with 2 (k + G) and the derivative of the
    projectors."""
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    bands = 11
    plane_waves = abinitio.PlaneWaveCrystal(crystal, [np.zeros(3)], [1.0], cutoff=16.0)
    reference = plane_waves.dielectric_constant(plane_waves.run_hartree_fock(bands, 6.0)) - 1.0
    spacings, values = [], []
    for n in (8, 12, 16):
        mesh = abinitio.MeshCrystal(crystal, n)
        mean_field = mesh.run_hartree_fock(4)
        levels, occupied, coupling, integrals, dipoles = mesh.coulomb_integrals(mesh.extend_bands(mean_field, bands))
        from tessera.drivers.bands import screening
        rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
        rpa.set_head(1.0, dipoles, crystal.volume, abinitio.COULOMB_STRENGTH)
        assert rpa.head_defect < 1e-10 and rpa.dielectric_constant < rpa.independent_particle_dielectric_constant
        spacings.append(mean_field["spacing"])
        values.append(rpa.independent_particle_dielectric_constant - 1.0)
    assert values[0] < values[1] < values[2] < 1.05 * reference
    assert richardson(spacings, values)[0] == pytest.approx(reference, rel=0.15)
