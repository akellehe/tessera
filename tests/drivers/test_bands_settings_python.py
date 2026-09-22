# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The approximations made for the sake of cost (#1161): their ranges, their
refusal of an order that is not implemented, their way from the command line
into a run, and the run itself end to end on synthetic pseudopotential files."""
import argparse
import os

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
    assert defaults.projector_quadrature == 6
    for bad in ({"self_energy_order": 0}, {"self_energy_order": 6}, {"zero_momentum_order": 6},
                {"refinement_terms": 0}, {"lattice_images": 4}, {"frequency_nodes": 4}, {"projector_quadrature": -1}):
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
    # On a momentum set the expansions exist at their first order, and the default orders are refused there.
    Approximations(1, 1, momenta=2).require_implemented()
    Approximations(3, 5, momenta=2).require_implemented()
    with pytest.raises(NotImplementedError, match="self_energy_order = 4 is not implemented on a momentum set"):
        Approximations(self_energy_order=4, momenta=2).require_implemented()
    with pytest.raises(ValueError):
        Approximations(momenta=0)


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
         "--frequency-nodes", "32", "--projector-quadrature", "4"]))
    assert parsed == Approximations(1, 1, 3, 3, 32, 12, 12, momenta=1, projector_quadrature=4)
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


def _screened(approximations, bands=7):
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 6, approximations=approximations)
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 3, tolerance=1e-9)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    return mesh, extended, levels, screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)


def test_the_diagrams_beyond_the_first_order_enter_the_quasiparticle_equation():
    """With every mode and every pole kept, the first-order diagram of the
    evaluator is the self-energy `RandomPhase` already has, which holds the
    wiring (modes, couplings, chemical potential) to it; the second order then
    moves the quasiparticle levels, and the equation is still solved to its root."""
    mesh, extended, levels, rpa = _screened(Approximations(2, 1, vertex_bands=7, vertex_poles=6))
    first = [rpa.quasiparticle(n)[0] for n in (0, 1)]
    order, chosen, interaction, poles = mesh.vertex(extended, 7)
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


def test_the_fourth_order_enters_the_self_energy():
    """`--self-energy-order` 4 through the run's own wiring: the diagrams that
    `RandomPhase` adds are those of the orders 2 to k of the evaluator it built
    (the loop over the orders is the same for 5), on 4 modes and 3 poles."""
    mesh, extended, levels, rpa = _screened(Approximations(4, 1, vertex_bands=4, vertex_poles=3))
    order, chosen, interaction, poles = mesh.vertex(extended, 7)
    assert (order, len(chosen), poles) == (4, 4, 3)
    rpa.set_vertex(order, chosen, interaction, poles)
    w = 0.5 * (levels[0] + levels[1])
    totals = []
    for k in (3, 4):
        rpa.vertex_order = k
        totals.append(rpa._vertex(chosen[1], w, levels)[0])
    _, chemical_potential, engines = rpa._vertex_engines
    fourth = np.mean([engine.evaluate(1, w - chemical_potential, 4).real for engine in engines])
    assert totals[1] - totals[0] == pytest.approx(fourth, abs=1e-9) and abs(fourth) > 1e-9


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
    # The plain exchange update contracts by a few per cent a step on these eight soft ions and stalls; with the
    # accelerator of the update (`acceleration`) every mesh converges, to a minimum of the energy on its span.
    assert result["certified"] is True
    for row in result["runs"]:
        assert row["exchange_updates"] <= 12 and row["hartree_fock_lowest_curvature"] > 0.0
        assert row["covariance"]["purity_defect"] < 1e-10
        assert row["covariance"]["particles"] == pytest.approx(16.0, abs=1e-9)
        assert row["head_defect"] < 1e-10 and row["dielectric_constant"] > 1.0
        assert row["g0w0"] < row["g0w0_body"] < row["hartree_fock"]            # screening closes a Hartree-Fock gap
    assert set(result["extrapolated"]) >= {"hartree_fock", "g0w0", "gw0", "evgw"} and result["amplification"] > 1.0


@pytest.mark.slow
def test_the_run_end_to_end_on_a_momentum_set(tmp_path):
    """`--momenta 2`: Hartree-Fock on the 2 x 2 x 2 set with one momentum of
    every orbit solved, and the quasiparticle equation with the screened
    interaction of every transfer of the set, the zero-transfer entry included."""
    path = _files(tmp_path)
    result = gaas.main(["ab-initio", "--cation", path, "--anion", path, "--divisions", "6", "--bands", "18",
                        "--screening-bands", "20", "--lattice-constant", "5.0", "--self-energy-order", "1",
                        "--zero-momentum-order", "1", "--momenta", "2"])
    assert result["approximations"]["momenta"] == 2
    row = result["runs"][0]
    assert row["solved_momenta"] == 4 and row["dielectric_constant"] > 1.0
    assert row["g0w0"] < row["hartree_fock"]                                     # screening closes a Hartree-Fock gap
    assert np.isfinite(row["gw0"])
    # On this fixture evGW closes a gap between momenta of the set (its levels fed back into the screening pull an
    # empty level below a filled one at the transfer (0, 0, 1/2)); the run records that and keeps its other results.
    assert (row["evgw"] is not None and np.isfinite(row["evgw"])) or "not positive" in row["evgw_failure"]


def test_a_method_without_a_solution_is_recorded_and_the_run_goes_on():
    rows, lines = {}, []

    def closes():
        raise ValueError("a level difference at the transfer (0, 0, 0.5) is not positive: the levels fed back have "
                         "closed a gap there")
    gaas._fed_back(rows, "evgw", closes, lambda levels: 1.0, lines.append, "N=6")
    assert rows["evgw"] is None and "not positive" in rows["evgw_failure"] and "no solution" in lines[0]
    gaas._fed_back(rows, "gw0", lambda: (np.zeros(3), [1e-6]), lambda levels: 2.5, lines.append, "N=6")
    assert rows["gw0"] == 2.5 and rows["gw0_residual"] == 1e-6

    def broken():
        raise ValueError("some other error")
    with pytest.raises(ValueError, match="some other error"):
        gaas._fed_back(rows, "g0w0", broken, lambda levels: 0.0, lines.append, "N=6")


def test_the_tables_of_transitions_change_where_the_numbers_live_and_not_the_numbers(tmp_path):
    """The momentum terms' transition amplitudes are kept for a whole loop,
    batched, and summed without a loop over the sections: against the plain
    per-section sums of the formula, and eigenvalue self-consistency with the
    tables on disk equal to the one without them."""
    from tessera.drivers.bands import screening
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    mesh = abinitio.MeshCrystal(crystal, 6, approximations=Approximations(1, 2))
    bands = 7
    extended = mesh.extend_bands(mesh.run_hartree_fock(4, tolerance=1e-9), bands + 8, tolerance=1e-9)
    levels, occupied, coupling, integrals = mesh.coulomb_integrals(extended, bands)
    heads = mesh.vanishing_momentum_pairs(extended, coupling, bands)
    terms = mesh.momentum_terms(extended, range(bands), bands)
    rpa = screening.RandomPhase.from_pieces(levels, occupied, coupling, integrals)
    rpa.set_head(mesh.zero_momentum, heads)
    rpa.set_momentum_terms(*terms, scratch=str(tmp_path))
    rpa.prefetch = True
    frequency = 0.5 * (levels[0] + levels[1])
    for n in (0, 3):
        for index, term in enumerate(rpa.momentum_terms):
            value, derivative = rpa._integrand(index, n, frequency)
            omega, modes, _ = rpa._momentum_modes[index]
            weights = 2.0 * np.abs(np.asarray(term["blocks"][n]) @ modes) ** 2
            expected = expected_derivative = 0.0
            for m, level in enumerate(np.asarray(term["levels"])):
                poles = level - omega if m < occupied else level + omega
                expected += np.sum(weights[m] / (frequency - poles))
                expected_derivative -= np.sum(weights[m] / (frequency - poles) ** 2)
            assert value == pytest.approx(expected, rel=1e-12, abs=1e-15)
            assert derivative == pytest.approx(expected_derivative, rel=1e-12, abs=1e-15)
    assert any(name.endswith(".npy") for name in os.listdir(tmp_path / "transitions"))
    kept, _, _ = screening.self_consistent_quasiparticles(levels, occupied, coupling, integrals,
                                                          head=(mesh.zero_momentum, heads), update_screening=True,
                                                          tolerance=1e-8, momentum_terms=terms,
                                                          scratch=str(tmp_path))
    plain, _, _ = screening.self_consistent_quasiparticles(levels, occupied, coupling, integrals,
                                                           head=(mesh.zero_momentum, heads), update_screening=True,
                                                           tolerance=1e-8, momentum_terms=terms)
    assert np.abs(kept - plain).max() < 1e-12
    # One screened interaction's tables at a time stay on disk.
    assert len({name.split("_")[0] for name in os.listdir(tmp_path / "transitions")}) == 1
