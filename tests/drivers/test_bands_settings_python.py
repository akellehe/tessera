# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The approximations made for the sake of cost (#1161): their ranges, their
refusal of an order that is not implemented, their way from the command line
into a run, and the run itself end to end on synthetic pseudopotential files."""
import argparse

import numpy as np
import pytest

from tessera.drivers.bands import abinitio, gaas
from tessera.drivers.bands.settings import IMPLEMENTED, Approximations
from tests.drivers.test_bands_abinitio_python import UPF, soft_atom


def test_ranges_and_defaults():
    defaults = Approximations()
    assert (defaults.self_energy_order, defaults.zero_momentum_order) == (3, 3)
    assert defaults.refinements == (2, 3, 4, 6, 8) and defaults.images == (-2, -1, 0, 1, 2)
    assert Approximations(refinement_terms=1).refinements == (8,) and Approximations(lattice_images=1).images == (0,)
    for bad in ({"self_energy_order": 0}, {"self_energy_order": 6}, {"zero_momentum_order": 6},
                {"refinement_terms": 0}, {"lattice_images": 4}, {"frequency_nodes": 4}):
        with pytest.raises(ValueError):
            Approximations(**bad)


def test_an_order_that_is_not_implemented_is_refused_by_name(monkeypatch):
    """Every order of both expansions exists (#1168 brought the fourth and the
    fifth of the self-energy); the refusal stays for whatever is added next."""
    assert IMPLEMENTED == {"self_energy_order": (1, 2, 3, 4, 5), "zero_momentum_order": (1, 2, 3, 4, 5)}
    for order in (1, 2, 3, 4, 5):
        Approximations(self_energy_order=order).require_implemented()
    monkeypatch.setitem(IMPLEMENTED, "self_energy_order", (1, 2, 3))
    Approximations().require_implemented()
    with pytest.raises(NotImplementedError, match="self_energy_order = 4"):
        Approximations(self_energy_order=4).require_implemented()


def test_the_memory_of_the_diagrams_is_a_flag():
    parser = argparse.ArgumentParser()
    Approximations.add_arguments(parser)
    assert Approximations.from_arguments(parser.parse_args([])).vertex_memory == 8.0
    parsed = Approximations.from_arguments(parser.parse_args(["--self-energy-order", "5", "--vertex-memory", "2.5"]))
    assert (parsed.self_energy_order, parsed.vertex_memory) == (5, 2.5) and parsed.record()["vertex_memory"] == 2.5
    with pytest.raises(ValueError):
        Approximations(vertex_memory=0.0)


def test_the_momentum_nodes_are_a_midpoint_grid_counted_once_per_time_reversed_pair():
    assert Approximations(1, 1).momentum_nodes == []
    for order in (2, 3, 4, 5):
        nodes = Approximations(1, order).momentum_nodes
        assert sum(weight for _, weight in nodes) == pytest.approx(1.0)
        assert all(max(abs(v) for v in kappa) < 0.5 for kappa, _ in nodes)
        assert not any(tuple(-v + 0.0 for v in kappa) in dict(nodes) for kappa, _ in nodes)
    assert len(Approximations(1, 2).momentum_nodes) == 4                  # (+-1/4)^3, a transfer and its opposite once
    assert len(Approximations(1, 3).momentum_nodes) == 13 + 3            # 26 / 2, and zero transfer as three small ones


def test_the_flags_reach_the_run():
    parser = argparse.ArgumentParser()
    Approximations.add_arguments(parser)
    parsed = Approximations.from_arguments(parser.parse_args(
        ["--self-energy-order", "1", "--zero-momentum-order", "1", "--refinement-terms", "3", "--lattice-images", "3",
         "--frequency-nodes", "32"]))
    assert parsed == Approximations(1, 1, 3, 3, 32, 12, 12)
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    coarse, fine = abinitio.MeshCrystal(crystal, 6, approximations=parsed), abinitio.MeshCrystal(crystal, 6)
    assert coarse.approximations.images == (-1, 0, 1)
    # Three terms of the refinement series against five: the same constant to the accuracy of the shorter series.
    assert coarse.zero_momentum == pytest.approx(fine.zero_momentum, abs=1e-5) and coarse.zero_momentum != fine.zero_momentum


def test_the_zone_average_of_the_self_energy_integrand_converges_with_the_grid():
    """Order 1 is the closed form at vanishing momentum; order k averages the
    integrand over a midpoint grid of k momentum transfers per axis, the
    singular part analytically. On a small cell the zone is large and order 1
    is far off: the filled level's correlation self-energy goes from -0.032 Ry
    to -0.056 and -0.064 Ry at orders 2 and 3 (-0.071 Ry at a grid of 6)."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    values = []
    for order in (1, 2, 3):
        mesh = abinitio.MeshCrystal(crystal, 6, approximations=Approximations(1, order))
        bands = 7
        extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 8, tolerance=1e-9)
        levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
        rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
        rpa.set_head(mesh.zero_momentum, mesh.vanishing_momentum_pairs(extended, coupling, bands))
        terms = mesh.momentum_terms(extended, (0, 1), bands)
        assert all(term["converged"] for term in terms[0])
        rpa.set_momentum_terms(*terms)
        values.append(rpa.correlation(0, 0.5 * (levels[0] + levels[1]))[0])
    print(values)
    assert values[0] == pytest.approx(-0.0320, abs=5e-4)
    assert values[2] < values[1] < values[0] < 0.0
    assert values[1] < 1.5 * values[0] and abs(values[2] - values[1]) < 0.5 * abs(values[1] - values[0])


def test_the_diagrams_beyond_the_first_order_enter_the_quasiparticle_equation():
    """With every mode and every pole kept, the first-order diagram of the
    evaluator is the self-energy `RandomPhase` already has, which holds the
    wiring (modes, couplings, chemical potential) to it; the second order then
    moves the quasiparticle levels, and the equation is still solved to its root."""
    from tessera.drivers.bands import screening
    from tessera.drivers.bands.diagrams import SkeletonSelfEnergy
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 6, approximations=Approximations(2, 1, vertex_bands=7, vertex_poles=6))
    bands = 7
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 3, tolerance=1e-9)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    first = [rpa.quasiparticle(n)[0] for n in (0, 1)]
    order, chosen, interaction, poles = mesh.vertex(extended, bands)
    assert (order, chosen, poles) == (2, list(range(7)), 6) and interaction.shape == (7,) * 4
    assert np.abs(interaction - interaction.transpose(1, 0, 2, 3)).max() < 1e-12
    rpa.set_vertex(order, chosen, interaction, poles)
    w = 0.5 * (levels[0] + levels[1])
    rpa._vertex(0, w, levels)                                                # builds the evaluator
    _, chemical_potential, engines = rpa._vertex_engines
    rpa.vertex_order = 1
    assert engines[0].evaluate(1, w - chemical_potential, 1).real == pytest.approx(rpa.correlation(1, w)[0], abs=1e-12)
    rpa.vertex_order = 2
    second = [rpa.quasiparticle(n) for n in (0, 1)]
    for n, (energy, weight) in enumerate(second):
        assert abs(levels[n] + rpa.correlation(n, energy)[0] - energy) < 1e-8 and 0.3 < weight < 1.2
    assert max(abs(second[n][0] - first[n]) for n in (0, 1)) > 1e-4


def _files(tmp_path):
    atom = soft_atom()
    text = lambda values: " ".join(f"{v:.12e}" for v in values)
    path = tmp_path / "soft.upf"
    path.write_text(UPF.format(kind="NC", r=text(atom.r), local=text(atom.local), beta=text(atom.projectors[0][1]),
                               density=text(atom.density)))
    return str(path)


def test_the_command_line_refuses_before_anything_runs(tmp_path, monkeypatch):
    path = _files(tmp_path)
    monkeypatch.setitem(IMPLEMENTED, "self_energy_order", (1, 2, 3))
    with pytest.raises(NotImplementedError, match="self_energy_order = 5"):
        gaas.main(["ab-initio", "--cation", path, "--anion", path, "--divisions", "6", "--self-energy-order", "5"])


@pytest.mark.slow
def test_the_run_end_to_end_on_synthetic_ions(tmp_path):
    path = _files(tmp_path)
    result = gaas.main(["ab-initio", "--cation", path, "--anion", path, "--divisions", "6", "8", "--bands", "12",
                        "--screening-bands", "24", "--lattice-constant", "5.0", "--self-energy-order", "1",
                        "--zero-momentum-order", "1"])
    assert result["approximations"] == Approximations(1, 1).record()
    # Eight soft ions are a fixture of the plumbing only: the exchange update of `run_hartree_fock` is a plain
    # fixed-point iteration, which contracts by a few per cent a step on this crystal (gallium arsenide with
    # published pseudopotentials converges in some twenty), so convergence is reported and not asserted here.
    assert isinstance(result["certified"], bool)
    for row in result["runs"]:
        assert row["covariance"]["purity_defect"] < 1e-10
        assert row["covariance"]["particles"] == pytest.approx(16.0, abs=1e-9)
        assert row["head_defect"] < 1e-10 and row["dielectric_constant"] > 1.0
        assert row["g0w0"] < row["g0w0_body"] < row["hartree_fock"]            # screening closes a Hartree-Fock gap
    assert set(result["extrapolated"]) >= {"hartree_fock", "g0w0", "gw0", "evgw"} and result["amplification"] > 1.0
