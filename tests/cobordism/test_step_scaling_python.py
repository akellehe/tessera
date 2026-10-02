"""The units of the linearized stationarity system.

The Jacobian's length rows and columns carry the unit of 1/z, so its rank is
read on the equation with every row and column multiplied by its variable's
scale (`HolomorphicRelaxation.variable_scales`): the modulus of the squared
length for a length coordinate, one for a link's increment and a multiplier.
The cell is the tick-1 host cell (0, 1, 2, 3) of the recursion run of
2026-10-01, whose squared lengths are of order 1e9, and the tick-0 host of
the run of 2026-09-23, whose squared lengths are 8.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs

from tests.drivers import _recursion_run_2026_09_23 as RUN

#: The tick-1 host cell (0, 1, 2, 3) of the run of 2026-10-01, as recorded.
DILATED_CELL = {
    "squared_lengths": [
        complex(2842442056.134229, -91405.50229423534),
        complex(3035544551.267494, -562355.1291879081),
        complex(4480356235.888196, 852532.8843812894),
        complex(5157071901.545203, -338028.86587001313),
        complex(5657491805.671762, 879501.240034435),
        complex(6154624323.67364, 432819.4583564187)],
    "links": [
        complex(-0.04225458178179704, 0.1097954824191668),
        complex(-0.12203545645129985, 0.04494497620869077),
        complex(-0.10648073217619775, -0.2029597947019897),
        complex(-0.5011746868080266, 0.9456600494109102),
        complex(0.9422698455536058, 1.7779568125833112),
        complex(-1.11479931570793, -1.590042265101136)],
}

CONTENT = (0, 0, 3)


def _system(cell):
    """The self-consistent system of `CONTENT` on a host cell, nothing held,
    as `baryon_poles.relax_content` poses it, with the base complex."""
    config = bp.default_config(kappas=[1.0], betas=[1.0],
                               selected_contents=[CONTENT])
    config["host_cell"] = cell
    config["held_sectors"] = []
    base = bp.build_base(config["edge_squared"], config["host_cell"])
    host = cs.sheeted_support(base, bp.SHEETS)
    order = bp.declared_villain_order(config)

    def declare(spacetime):
        return bp.action_declaration(spacetime, 1.0, 1.0,
                                     config["regge_hinges"],
                                     villain_order=order)

    moments = bp.fiber_moment_count(
        bp.mean_field_declaration(CONTENT, config),
        cob.JointAction(host.spacetime, declare(host.spacetime)),
        config.get("fiber_moments", bp.DECLARED_FIBER_MOMENTS),
        config.get("fiber_pinning", bp.DECLARED_FIBER_PINNING))

    def mean_field_of(support):
        declaration = bp.mean_field_declaration(CONTENT, config)
        declaration.geometry = bp.support_geometry(config, support, host)
        declaration.fiber_moments = moments
        return declaration

    return base, cs.ContentSystem(declare, mean_field_of, base, bp.SHEETS)


def _linear_algebra(point):
    """The residual, the Jacobian and the scales of a point, with the
    singular values of the Jacobian as it stands and scaled."""
    relaxation = point.relaxation
    residual = np.asarray(relaxation.residual(), dtype=complex)
    size = len(residual)
    jacobian = np.asarray(relaxation.jacobian(),
                          dtype=complex).reshape(size, size)
    scales = np.asarray(relaxation.variable_scales(), dtype=float)
    scaled = scales[:, None] * jacobian * scales[None, :]
    return (residual, jacobian, scales,
            np.linalg.svd(jacobian, compute_uv=False),
            np.linalg.svd(scaled, compute_uv=False))


@pytest.mark.parametrize("cell", [DILATED_CELL,
                                  RUN.HOST_CELLS[(0, 1, 2, 3)]],
                         ids=["dilated", "order-one"])
def test_the_scales_are_the_moduli_of_the_squared_lengths(cell):
    """One scale per variable in the Jacobian's column order: |z| of the
    coordinate for the six squared lengths, one for the six links and for
    every multiplier."""
    base, system = _system(cell)
    point = system.point(base)
    scales = np.asarray(point.relaxation.variable_scales())
    assert len(scales) == point.relaxation.variable_count()
    moduli = [abs(length * length)
              for _, _, length, _ in cs.edge_fields(base)]
    np.testing.assert_allclose(scales[:6], moduli, rtol=1e-15)
    np.testing.assert_array_equal(scales[6:], 1.0)


def test_the_rank_of_a_dilated_cell_keeps_the_length_equations():
    """On the dilated cell the Jacobian as it stands has every singular value
    that involves a squared length below 1e-15 of the largest, and the scaled
    one does not: the step's rank is the scaled Jacobian's, read at the
    declared rank tolerance."""
    base, system = _system(DILATED_CELL)
    point = system.point(base)
    _, _, _, raw, scaled = _linear_algebra(point)
    newton = point.relaxation.newton_step()
    tolerance = point.geometry.rank_tolerance
    raw_rank = int(np.sum(raw > tolerance * raw[0]))
    scaled_rank = int(np.sum(scaled > tolerance * scaled[0]))
    assert newton.jacobian_rank == scaled_rank
    assert scaled_rank > raw_rank
    assert newton.largest_singular_value == pytest.approx(scaled[0],
                                                          rel=1e-12)


@pytest.mark.parametrize("cell", [DILATED_CELL,
                                  RUN.HOST_CELLS[(0, 1, 2, 3)]],
                         ids=["dilated", "order-one"])
def test_the_step_is_the_minimum_norm_solution_of_the_scaled_equation(cell):
    """The step d solves (D J D) y = -D R in the minimum-norm least-squares
    sense at the rank the library reports, with d = D y: it agrees with that
    solution formed here from the bound Jacobian, residual and scales."""
    base, system = _system(cell)
    point = system.point(base)
    residual, jacobian, scales, _, _ = _linear_algebra(point)
    newton = point.relaxation.newton_step()
    scaled = scales[:, None] * jacobian * scales[None, :]
    left, singular, right = np.linalg.svd(scaled)
    rank = newton.jacobian_rank
    solution = right[:rank].conj().T @ (
        (left[:, :rank].conj().T @ (-(scales * residual))) / singular[:rank])
    expected = scales * solution
    step = np.asarray(newton.step, dtype=complex)
    # the two solutions differ by the rounding of two decompositions,
    # amplified by the smallest singular value kept
    conditioning = singular[0] / singular[rank - 1]
    assert np.linalg.norm((step - expected) / scales) <= \
        1e-13 * conditioning * np.linalg.norm(solution)


def test_the_step_lowers_every_block_of_the_dilated_cell_s_residual():
    """On the dilated cell the full step moves the squared lengths, and the
    linearized equation it solves leaves a remainder that is small against
    the residual in the scaled norm, the length block included."""
    base, system = _system(DILATED_CELL)
    point = system.point(base)
    residual, jacobian, scales, _, _ = _linear_algebra(point)
    step = np.asarray(point.relaxation.newton_step().step, dtype=complex)
    assert np.linalg.norm(step[:6] / scales[:6]) > 1e-3
    remainder = scales * (jacobian @ step + residual)
    assert np.linalg.norm(remainder[:6]) <= \
        1e-6 * np.linalg.norm((scales * residual)[:6])
