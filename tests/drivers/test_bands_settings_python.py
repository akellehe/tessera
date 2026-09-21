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


def test_an_order_that_is_not_implemented_is_refused_by_name():
    assert IMPLEMENTED == {"self_energy_order": (1,), "zero_momentum_order": (1,)}
    Approximations(self_energy_order=1, zero_momentum_order=1).require_implemented()
    with pytest.raises(NotImplementedError, match="self_energy_order = 3"):
        Approximations().require_implemented()
    with pytest.raises(NotImplementedError, match="zero_momentum_order = 2"):
        Approximations(self_energy_order=1, zero_momentum_order=2).require_implemented()


def test_the_flags_reach_the_run():
    parser = argparse.ArgumentParser()
    Approximations.add_arguments(parser)
    parsed = Approximations.from_arguments(parser.parse_args(
        ["--self-energy-order", "1", "--zero-momentum-order", "1", "--refinement-terms", "3", "--lattice-images", "3",
         "--frequency-nodes", "32"]))
    assert parsed == Approximations(1, 1, 3, 3, 32)
    crystal = abinitio.Crystal(6.0 * np.eye(3), [(soft_atom(), np.full(3, 0.5))])
    coarse, fine = abinitio.MeshCrystal(crystal, 6, approximations=parsed), abinitio.MeshCrystal(crystal, 6)
    assert coarse.approximations.images == (-1, 0, 1)
    # Three terms of the refinement series against five: the same constant to the accuracy of the shorter series.
    assert coarse.zero_momentum == pytest.approx(fine.zero_momentum, abs=1e-5) and coarse.zero_momentum != fine.zero_momentum


def _files(tmp_path):
    atom = soft_atom()
    text = lambda values: " ".join(f"{v:.12e}" for v in values)
    path = tmp_path / "soft.upf"
    path.write_text(UPF.format(kind="NC", r=text(atom.r), local=text(atom.local), beta=text(atom.projectors[0][1]),
                               density=text(atom.density)))
    return str(path)


def test_the_command_line_refuses_before_anything_runs(tmp_path):
    path = _files(tmp_path)
    with pytest.raises(NotImplementedError, match="self_energy_order = 3"):
        gaas.main(["ab-initio", "--cation", path, "--anion", path, "--divisions", "6"])


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
