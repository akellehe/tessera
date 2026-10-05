# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The first tick of the cosmological term (`recursion.tick_config`,
``--cosmological-constant-from-tick``): the actions of a tick before it carry
no cosmological constant, and from it on the level's relaxation and the
growth step carry the declared one; the cells' reads never do
(`recursion.cell_reads_config`).

The level relaxation of a tick is replaced by a function that records the
cosmological constant the tick's config declares and then has no value, so
the tick stops there with its record (`recursion.tick`) and no solve is run.
"""
import pytest

from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R


def test_the_declared_first_tick_is_every_tick():
    assert R.DECLARED_COSMOLOGICAL_CONSTANT_FROM_TICK == 0
    args = R.build_parser().parse_args(["run"])
    assert args.cosmological_constant_from_tick == 0
    args = R.build_parser().parse_args(
        ["run", "--cosmological-constant", "1",
         "--cosmological-constant-from-tick", "1"])
    assert (args.cosmological_constant,
            args.cosmological_constant_from_tick) == (1.0, 1)


def test_the_config_records_the_first_tick_with_a_constant_only():
    with_constant = R.default_config(cosmological_constant=1.0,
                                     cosmological_constant_from_tick=1)
    assert with_constant["cosmological_constant"] == 1.0
    assert with_constant["cosmological_constant_from_tick"] == 1
    without = R.default_config(cosmological_constant_from_tick=1)
    assert "cosmological_constant" not in without
    assert "cosmological_constant_from_tick" not in without


def test_a_negative_first_tick_is_named():
    with pytest.raises(ValueError, match="the first tick of the cosmological "
                                         "term"):
        R.default_config(cosmological_constant=1.0,
                         cosmological_constant_from_tick=-1)


@pytest.mark.parametrize("first, carried", [(0, [1.0, 1.0, 1.0]),
                                            (1, [0.0, 1.0, 1.0]),
                                            (2, [0.0, 0.0, 1.0])])
def test_a_tick_before_the_first_carries_no_constant(first, carried):
    config = R.default_config(cosmological_constant=1.0,
                              cosmological_constant_from_tick=first)
    assert [bp.declared_cosmological_constant(R.tick_config(config, index))
            for index in range(3)] == carried
    # the run's own config keeps the declared constant
    assert config["cosmological_constant"] == 1.0


@pytest.mark.parametrize("index, carried", [(0, 0.0), (1, 2.5)])
def test_each_tick_relaxes_and_records_with_its_constant(monkeypatch, index,
                                                         carried):
    seen = []

    def relaxation(spacetime, config, *args, **kwargs):
        seen.append(bp.declared_cosmological_constant(config))
        raise ValueError("stopped by the test")

    monkeypatch.setattr(R, "relax_level", relaxation)
    config = R.default_config(cosmological_constant=2.5,
                              cosmological_constant_from_tick=1)
    cells, z, links, _ = R.level_zero(config)
    record, following = R.tick(index, cells, z, links, config)
    assert seen == [carried]
    assert record["cosmological_constant"] == carried
    assert following is None


def test_a_run_without_a_constant_records_none_per_tick(monkeypatch):
    def relaxation(spacetime, config, *args, **kwargs):
        raise ValueError("stopped by the test")

    monkeypatch.setattr(R, "relax_level", relaxation)
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    record, _ = R.tick(0, cells, z, links, config)
    assert "cosmological_constant" not in record


@pytest.mark.parametrize("index", [0, 1, 2])
def test_the_cells_are_read_without_the_constant(index):
    """From the first tick on the level carries the constant and the cells'
    reads do not: a cell is one tetrahedron with no interior hinge."""
    config = R.default_config(cosmological_constant=1.0,
                              cosmological_constant_from_tick=1)
    level = R.tick_config(config, index)
    reads = R.cell_reads_config(level, index == 0)
    assert bp.declared_cosmological_constant(level) == (
        0.0 if index == 0 else 1.0)
    assert bp.declared_cosmological_constant(reads) == 0.0
    assert "cosmological_constant" not in reads
    assert reads["hold_cell_sectors"] is (index == 0)
    # everything else of the tick's config reaches the cells
    assert {k: v for k, v in reads.items() if k != "hold_cell_sectors"} == {
        k: v for k, v in level.items() if k != "cosmological_constant"}


def test_the_help_names_the_actions_that_carry_the_constant():
    run = R.build_parser()._subparsers._group_actions[0].choices["run"]
    helps = {action.option_strings[0]: action.help for action in run._actions
             if action.option_strings}
    assert "never carry" in helps["--cosmological-constant"]
    assert "the cells' reads never do" in helps[
        "--cosmological-constant-from-tick"]
