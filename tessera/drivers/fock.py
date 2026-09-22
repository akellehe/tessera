# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The inductive limit of the Fock stages over a refinement sequence.

What is measured
----------------
The infinite Fock space is the direct limit ``F = lim(H_M, iota_M)`` of the
finite stages under the vacuum embedding ``iota_M(psi) = psi (x) |0>``, which
adds modes and changes no amplitude. The maps carried along the sequence are
consistent only when

    ||iota_M V_M - V_{M+1} iota_M|| -> 0

over a refinement sequence. That is a numerical certificate on a SEQUENCE. One
pair of stages gives one number, which establishes a limit no more than one
term establishes a series, so this driver measures the defect at every adjacent
pair of a declared sequence, on one and the same carried subspace, and reports
whether it falls and by what worst ratio.

`quantum.LazyFockEngine.inductiveLimit` does the measuring; this module builds
the stages and states the fixture.

The fixture, stated in full
---------------------------
The default sequence is a one-particle model in which every mode couples to
every other, and the coupling of a mode is set by its own index:

    h_ij = t * r ** max(i, j)   (i != j),      h_ii = e + i * d

with ``0 < r < 1``. Stage M carries the first M modes and the map
``V_M = dGamma(h_M)``, the second quantization of that stage's one-particle
operator (`quantum.ExteriorAlgebra.dGamma`). Because ``h_ij`` does not depend
on M, ``h_M`` is the leading principal submatrix of ``h_{M+1}`` and the stages
are nested exactly as the vacuum embedding requires.

The defect a step measures is then the coupling of the NEWLY ADDED mode into
the carried subspace: ``dGamma(h_{M+1})`` sends a carried state to the states
``dGamma(h_M)`` sends it to, plus a component on the new mode of size
``t * r ** M``. The defect therefore falls by the factor ``r`` at every step,
which is the whitepaper's "at any finite stage only finitely many modes have
interacted" made measurable. A refinement whose added modes did NOT decouple
would show it here: the defect is measured, never assumed.

The model is a fixture, not a claim. Any nested sequence of one-particle
operators -- the compressions of a real refinement, for instance -- is measured
by the same code: pass it to `measure` as `operators`.

Running it
----------
::

    python -m tessera.drivers.fock --stages 6 --active-modes 2
"""

import argparse
import json
import sys

import numpy as np

from tessera import quantum as qu

#: Number of stages in the refinement sequence. Two stages measure one defect
#: and establish nothing; the acceptance is that the defect FALLS, which needs
#: at least three.
DECLARED_STAGES = 6
#: Modes of the first stage. Every later stage adds one mode to the one before.
DECLARED_FIRST_STAGE_MODES = 3
#: Modes spanning the carried subspace the comparison is made on. The active
#: basis is every occupation state over these modes, so the subspace has
#: dimension 2 ** this, and it must lie inside the first stage.
DECLARED_ACTIVE_MODES = 2
#: The hopping amplitude t of the fixture's one-particle model.
DECLARED_HOPPING = 1.0
#: The decay ratio r of the fixture: the coupling of mode i is t * r ** i, so
#: a mode added at stage M disturbs the carried subspace by t * r ** M.
DECLARED_DECAY = 0.25
#: The on-site energy e and its per-mode increment d.
DECLARED_ONSITE = 1.0
DECLARED_ONSITE_STEP = 0.5


def build_config(stages=DECLARED_STAGES,
                 first_stage_modes=DECLARED_FIRST_STAGE_MODES,
                 active_modes=DECLARED_ACTIVE_MODES,
                 hopping=DECLARED_HOPPING,
                 decay=DECLARED_DECAY,
                 onsite=DECLARED_ONSITE,
                 onsite_step=DECLARED_ONSITE_STEP):
    """The configuration of one inductive-limit measurement.

    Raises ValueError on a sequence too short to show a fall, a carried
    subspace that does not fit inside the first stage, or a decay ratio outside
    the open unit interval, where the fixture's couplings would not decay at
    all.
    """
    stages = int(stages)
    first_stage_modes = int(first_stage_modes)
    active_modes = int(active_modes)
    if stages < 3:
        raise ValueError(
            "a refinement sequence that can show a defect FALLING is at least "
            "three stages, which is two defects to compare; got %r" % stages)
    if first_stage_modes < 1:
        raise ValueError("the first stage carries at least one mode, got %r"
                         % first_stage_modes)
    if not 1 <= active_modes <= first_stage_modes:
        raise ValueError(
            "the carried subspace must lie inside the first stage: its %r "
            "active modes do not fit in the first stage's %r"
            % (active_modes, first_stage_modes))
    if not 0.0 < float(decay) < 1.0:
        raise ValueError(
            "the fixture's decay ratio r sets the coupling of mode i to "
            "t * r ** i and must lie in (0, 1) for the couplings to decay at "
            "all, got %r" % (decay,))
    return {"stages": stages,
            "first_stage_modes": first_stage_modes,
            "active_modes": active_modes,
            "hopping": float(hopping),
            "decay": float(decay),
            "onsite": float(onsite),
            "onsite_step": float(onsite_step)}


def one_particle_stages(config):
    """The fixture's nested one-particle operators, one per stage.

    Entry (i, j) does not depend on the stage, so each operator is the leading
    principal submatrix of the next and the stages are nested exactly as the
    vacuum embedding requires.
    """
    largest = config["first_stage_modes"] + config["stages"] - 1
    index = np.arange(largest)
    h = config["hopping"] * config["decay"] ** np.maximum(
        index[:, None], index[None, :])
    np.fill_diagonal(h, config["onsite"] + config["onsite_step"] * index)
    h = h.astype(complex)
    return [h[:m, :m].copy()
            for m in range(config["first_stage_modes"], largest + 1)]


def fock_stages(operators):
    """The refinement stages of a nested sequence of one-particle operators:
    stage M carries modes 0..M-1 and the map dGamma(h_M), dense over the
    2 ** M support Fock basis in the ascending mode order both
    `ExteriorAlgebra` and `LazyFockEngine.applyLocalMapDense` use.

    Raises ValueError when the operators are not square, or not nested (each
    the leading principal submatrix of the next), since the vacuum embedding
    between two stages that disagree on their common modes is not an embedding
    of the one into the other.
    """
    stages = []
    previous = None
    for h in operators:
        h = np.asarray(h, dtype=complex)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0:
            raise ValueError("every stage's one-particle operator is a "
                             "nonempty square matrix, got shape %r"
                             % (h.shape,))
        if previous is not None:
            if h.shape[0] <= previous.shape[0]:
                raise ValueError(
                    "a refinement sequence grows: stage of %d modes follows "
                    "one of %d" % (h.shape[0], previous.shape[0]))
            kept = previous.shape[0]
            if not np.allclose(h[:kept, :kept], previous, atol=0.0, rtol=0.0):
                raise ValueError(
                    "the stages are not nested: the %d-mode operator is not "
                    "the leading principal submatrix of the %d-mode one"
                    % (kept, h.shape[0]))
        modes = list(range(h.shape[0]))
        rows, columns, values, size = qu.ExteriorAlgebra(
            h.shape[0]).dGammaCOO(h)
        dense = np.zeros((size, size), dtype=complex)
        dense[np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)] = \
            np.asarray(values, dtype=complex)
        stages.append(qu.FockRefinementStage(modes, modes, dense))
        previous = h
    return stages


def active_basis(active_modes):
    """Every occupation state over the first `active_modes` modes: the carried
    subspace the compatibility is measured on, the same at every stage."""
    return [[mode for mode in range(active_modes) if word >> mode & 1]
            for word in range(1 << active_modes)]


def measure(config=None, operators=None):
    """Measure the inductive compatibility along a refinement sequence.

    `operators` is a nested sequence of one-particle operators; the fixture of
    `one_particle_stages` is used when it is not supplied. Returns
    (read, record): the `LazyInductiveLimitRead` and a JSON-able summary.
    """
    config = build_config() if config is None else config
    operators = (one_particle_stages(config) if operators is None
                 else list(operators))
    stages = fock_stages(operators)
    modes = [len(stage.modes) for stage in stages]
    if config["active_modes"] > modes[0]:
        raise ValueError(
            "the carried subspace must lie inside the first stage: its %d "
            "active modes do not fit in the first stage's %d"
            % (config["active_modes"], modes[0]))
    engine = qu.LazyFockEngine(modes[-1])
    read = engine.inductiveLimit(stages, active_basis(config["active_modes"]))
    record = {
        "stageModes": modes,
        "activeModes": config["active_modes"],
        "activeDimension": int(read.activeDimension),
        "defects": [float(d) for d in read.defects],
        "lastDefect": float(read.lastDefect),
        "largestRatio": float(read.largestRatio),
        "falls": bool(read.falls),
        "certified": bool(read.certificate.holds()),
        "residual": float(read.certificate.residual),
    }
    return read, record


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tessera.drivers.fock",
        description="Measure the inductive compatibility of the Fock stages "
                    "along a refinement sequence.")
    parser.add_argument("--stages", type=int, default=DECLARED_STAGES)
    parser.add_argument("--first-stage-modes", type=int,
                        default=DECLARED_FIRST_STAGE_MODES)
    parser.add_argument("--active-modes", type=int,
                        default=DECLARED_ACTIVE_MODES)
    parser.add_argument("--hopping", type=float, default=DECLARED_HOPPING)
    parser.add_argument("--decay", type=float, default=DECLARED_DECAY)
    parser.add_argument("--onsite", type=float, default=DECLARED_ONSITE)
    parser.add_argument("--onsite-step", type=float,
                        default=DECLARED_ONSITE_STEP)
    parser.add_argument("--json", action="store_true",
                        help="write the record as JSON on standard output")
    args = parser.parse_args(argv)
    config = build_config(stages=args.stages,
                          first_stage_modes=args.first_stage_modes,
                          active_modes=args.active_modes,
                          hopping=args.hopping, decay=args.decay,
                          onsite=args.onsite, onsite_step=args.onsite_step)
    _, record = measure(config)
    if args.json:
        json.dump(record, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    print("stages (modes):      %s" % record["stageModes"])
    print("carried subspace:    %d modes, dimension %d"
          % (record["activeModes"], record["activeDimension"]))
    for step, defect in enumerate(record["defects"]):
        print("  defect %d -> %d:     %.6e"
              % (record["stageModes"][step], record["stageModes"][step + 1],
                 defect))
    print("worst step ratio:    %.6e" % record["largestRatio"])
    print("the defect falls:    %s" % record["falls"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
