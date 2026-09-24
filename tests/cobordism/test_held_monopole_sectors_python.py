# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Held monopole sectors in `HolomorphicRelaxation`: the declared monopole
number and the moduli of the cut faces' holonomies are kept, a sector declared
with the wrong number is refused, and on the symmetric unit-monopole
tetrahedron (stationary in its sector) the held solve converges."""
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R


def _sectors(number):
    return R.held_sectors([[0, 1, 2, 3]], [number], 4)


def test_a_wrong_declared_number_is_refused():
    spacetime = bp.build_host()
    config = bp.default_config([1.0], [1.0])
    config["held_sectors"] = _sectors(0)
    action = cob.JointAction(spacetime, bp.action_declaration(spacetime, 1.0, 1.0))
    with pytest.raises(ValueError, match="declared with monopole number 0"):
        cob.HolomorphicRelaxation(action, bp.relaxation_declaration(config))


def test_the_symmetric_host_relaxes_inside_its_sector():
    spacetime = bp.build_host()
    config = bp.default_config([1.0], [1.0])
    report = R.relax_level(spacetime, dict(config, kappa=1.0, beta=1.0), _sectors(1))
    assert report["converged"]
    assert report["sector_monopole_numbers"] == [1, 1, 1]
    assert report["held_modulus_drift"] < 1e-12
    assert report["sector_guard_damped_steps"] == 0


def test_the_hold_passes_through_the_mean_field_solve():
    config = bp.default_config([1.0], [1.0], selected_contents=[(1, 1, 1)])
    config["held_sectors"] = _sectors(1)
    geometry = bp.relaxation_declaration(config)
    assert [s.monopole_number for s in geometry.held_sectors] == [1, 1, 1]
    assert len(geometry.held_sectors[0].faces) == 4
