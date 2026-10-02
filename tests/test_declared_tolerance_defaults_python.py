# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The library's declared tolerances: every tolerance and rank threshold of a
declaration, a configuration, an options record or a method on the path of the
recursion and baryon drivers defaults to 1e-15, and each is the caller's to
set.

A *tolerance* here is a number a residual, a separation or a singular-value
ratio is compared with; a *default* is the value the library uses when the
caller declares none. The defaults of fields are read off a default-built
record; the defaults of arguments are read off the signature the binding
publishes in the method's docstring.
"""
import re

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import recursion as R

DECLARED = 1e-15


def _argument_defaults(method):
    """The default of every argument of a bound method whose default is a
    number, by name, from the signature line of its docstring."""
    signature = method.__doc__.splitlines()[0]
    return {name: float(value) for name, value in re.findall(
        r"(\w+): [^,=]*= ([-+0-9.eE]+)(?=[,)])", signature)}


@pytest.mark.parametrize("record, fields", [
    (cob.HolomorphicRelaxationDeclaration, ("rank_tolerance",)),
    (cob.SelfConsistentMeanFieldDeclaration, ("band_tolerance", "tolerance")),
    (cob.BoundStatePoleConfig, ("rank_tolerance",)),
    (cob.DressedFluctuationDeclaration, ("tolerance",)),
    (cob.LevelRecursionDeclaration, ("tolerance", "rank_tolerance")),
    (cob.RecursiveQuotient.Options,
     ("tolerance", "rankTolerance", "nearIsometryEpsilon")),
    (obs.IsospinDoubletConfig,
     ("grouping_tolerance", "min_relative_gap", "projector_tolerance",
      "invariance_tolerance", "commutant_tolerance", "isotypic_tolerance",
      "hermiticity_tolerance", "transport_leakage_tolerance",
      "intertwining_tolerance", "span_tolerance", "transport_rank_tolerance",
      "singular_value_grouping_tolerance", "member_splitting_tolerance",
      "occupation_tolerance")),
])
def test_every_tolerance_field_defaults_to_1e_15(record, fields):
    built = record()
    for field in fields:
        assert getattr(built, field) == DECLARED, field
        setattr(built, field, 1e-7)
        assert getattr(built, field) == 1e-7


@pytest.mark.parametrize("method, arguments", [
    (obs.MonopoleSupport.__init__, ("unitModulusTolerance",)),
    (obs.MonopoleSupport.monopoleNumber, ("tolerance",)),
    (obs.MonopoleSupport.coexactProjector, ("tolerance",)),
    (obs.MonopoleSupport.gaugeCompensation, ("tolerance",)),
    (obs.MonopoleSupport.cocycle, ("tolerance",)),
    (obs.MonopoleSupport.spinorBands, ("degeneracyTolerance", "tolerance")),
    (obs.MonopoleSupport.spinRead, ("degeneracyTolerance", "tolerance")),
    (obs.SharpSpin.read, ("tolerance",)),
    (obs.SharpSpin.isotypicRead, ("tolerance",)),
    (obs.SheetedSupport.certifyIsomorphism, ("tolerance",)),
    (obs.SheetAttachment.attachmentMatrix, ("fullRankTolerance",)),
    (ch.PencilSchur.feshbach, ("rank_tolerance", "resonance_radius")),
    (ch.DressedAnchor.profile, ("tolerance",)),
    (ch.GrownCellRule.invertWhitneyBlock, ("rank_tolerance",)),
    (ch.GrownCellRule.invertVertexPairing, ("rank_tolerance",)),
    (cob.BoundStatePole.response, ("rank_tolerance",)),
    (cob.BoundStatePole.response_derivative, ("rank_tolerance",)),
])
def test_every_tolerance_argument_defaults_to_1e_15(method, arguments):
    defaults = _argument_defaults(method)
    for argument in arguments:
        assert defaults[argument] == DECLARED, (argument, defaults)


def test_the_isospin_detector_keeps_its_criteria():
    """The detector's thresholds that are criteria of the read, not
    tolerances, are the library's declared values, and its cap and its
    limits are options that are not declared as built."""
    config = obs.IsospinDoubletConfig()
    assert config.contour_nodes == 64
    assert config.track_overlap_threshold == 0.5
    assert config.min_frames == 2
    assert config.condition_number_cap is None
    assert config.decomposed_rank_limit is None
    assert config.decomposed_commutant_limit is None
    config.condition_number_cap = 1e8
    config.decomposed_rank_limit = 24
    config.decomposed_commutant_limit = 100
    assert (config.condition_number_cap, config.decomposed_rank_limit,
            config.decomposed_commutant_limit) == (1e8, 24, 100)


def test_the_size_limit_of_a_level_is_not_declared():
    """The level recursion refuses a level at or above a dense crossover only
    when the caller declares one; none is declared as built."""
    declaration = cob.LevelRecursionDeclaration()
    assert declaration.dense_crossover is None
    declaration.dense_crossover = 512
    assert declaration.dense_crossover == 512
    declaration.dense_crossover = None
    assert declaration.dense_crossover is None


def test_the_move_and_admissibility_tolerances_of_a_node():
    """A `MultiCobordism` node commits a move that lowers the objective by
    more than its move tolerance and admits a geometry whose
    Kontsevich-Segal margin is at least minus its admissibility tolerance;
    both are 1e-15 as built, both are the caller's to set, and a negative
    value is refused by name."""
    config = R.default_config()
    cells, z, links, _ = R.level_zero(config)
    spacetime, _ = R.build_level(cells, z, links, sheets=1)
    node = cob.MultiCobordism(spacetime, [], [],
                              list(R.PACHNER_REGISTER_DEGREES), 1.0, 0, 0,
                              False)
    assert node.move_tolerance == DECLARED
    assert node.admissibility_tolerance == DECLARED
    node.move_tolerance = 1e-9
    node.admissibility_tolerance = 1e-12
    assert node.move_tolerance == 1e-9
    assert node.admissibility_tolerance == 1e-12
    node.move_tolerance = 0.0
    assert node.move_tolerance == 0.0
    with pytest.raises(ValueError, match="move tolerance must be zero or "
                                         "positive"):
        node.move_tolerance = -1e-9
    with pytest.raises(ValueError, match="admissibility tolerance must be "
                                         "zero or positive"):
        node.admissibility_tolerance = float("nan")


def test_the_unit_modulus_tolerance_is_the_callers():
    """`MonopoleSupport` refuses a connection value whose modulus departs
    from one by more than the tolerance its caller declares: a value 1e-12
    off the unit circle is refused at the declared 1e-15 and taken at 1e-9,
    and a tolerance that is not positive is refused by name."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
    connection = [complex(u) for u in fixture.connection]
    connection[3] *= 1.0 + 1e-12
    with pytest.raises(ValueError, match="every value must have unit "
                                         "modulus"):
        obs.MonopoleSupport(4, fixture.edges, fixture.faces, connection)
    loose = obs.MonopoleSupport(4, fixture.edges, fixture.faces, connection,
                                1e-9)
    assert loose.monopoleNumber(1e-9).monopole_number == 1
    with pytest.raises(ValueError, match="unit-modulus tolerance must be "
                                         "positive"):
        obs.MonopoleSupport(4, fixture.edges, fixture.faces,
                            fixture.connection, 0.0)


def test_the_branch_cut_is_read_at_the_callers_tolerance():
    """A face flux within the caller's tolerance of the ends of the principal
    interval sits on the branch cut. On the unit-monopole tetrahedron every
    face carries pi / 2, a margin of pi / 2 from the cut; with one link
    turned so that a face carries pi - 1e-12, that face is on the cut at a
    tolerance of 1e-9 and off it at the declared 1e-15."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
    read = fixture.monopoleNumber()
    assert not read.on_branch_cut
    assert read.branch_margin == pytest.approx(np.pi / 2, abs=1e-12)
    # the face (0, 1, 3) carries the holonomy U_13 alone (U_01 = U_03 = 1)
    connection = [complex(u) for u in fixture.connection]
    connection[4] = np.exp(1j * (np.pi - 1e-12))
    turned = obs.MonopoleSupport(4, fixture.edges, fixture.faces, connection)
    assert turned.monopoleNumber(1e-9).on_branch_cut
    declared = turned.monopoleNumber()
    assert not declared.on_branch_cut
    assert declared.branch_margin == pytest.approx(1e-12, rel=1e-3)
