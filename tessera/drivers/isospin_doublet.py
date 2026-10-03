# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The isospin doublet observed on the three-sheeted unit-monopole host.

Flavour is observed, never declared (whitepaper v16, Section 10: "Flavor and
electric charge are not assumed as hidden labels"). This driver hands the host
of `tessera.drivers.baryon_poles` to `IsospinDoublet` and records what it finds,
including "no doublet".

The host
--------
Three isomorphic, disjoint sheets of the regular tetrahedron, each carrying the
symmetric unit Dirac monopole (`baryon_poles.build_host`). The certified
sheeting (sheet-major cells, six base edges per sheet) is declared as the
colour multiplicity; the projective rotation action D_1(g) of T = A_4 on each
sheet's edges (`MonopoleSupport.edgeRepresentation`) is declared as the
support's symmetry, with the projective class read from
`MonopoleSupport.cocycle`.

The operators read
------------------
Every read is on the covariant operator h_1(z, U) (WP §3.2). The spin content
is read on its T-average (WP §11.1, line 497), so the detector is run on both:

* ``covariant``: h_1(z, U) itself, the carrier operator of the joint action;
* ``t_averaged``: (1/|T|) sum_g D_1(g)^{-1} h_1 D_1(g).

Frames and resolutions
----------------------
A single-level synthesis spans one cobordism frame and one resolution (a
tetrahedral support is never refined here), so frame persistence, refinement
persistence and transport over the lifetime are unmeasured and read "not
evaluable" wherever a candidate exists. A lineage needs a cobordism history,
so the baryon number is unmeasured too. The third condition, agreement with the
Ward flux, is not evaluable because Section 13.4 is deferred.

Running it
----------
::

    python -m tessera.drivers.isospin_doublet run --json doublet.json \\
        [--kappa 1 --beta 1 --content 1 1 1] [--villain-order 10]

Without ``--kappa`` the declared host is read (its carrier operator does not
depend on kappa or beta). With it, each (kappa, beta) and each content is first
relaxed to self-consistency exactly as `baryon_poles` does, and the relaxed
host is read. The baryon driver's ``--isospin-doublet`` option adds the same
record to every content of its own scan. ``--villain-order`` is the order the
Villain weight of the action's holonomy term is summed to
(`baryon_poles.DECLARED_VILLAIN_ORDER`, ten by default); the result records
it.

Every tolerance of the stack (`baryon_poles.TOLERANCES`) and every limit a
user may declare on a solve (`baryon_poles.LIMITS`) is an option here as it is
of the baryon driver: each tolerance defaults to 1e-15, no limit is declared
by default, and the values the run was made at are recorded in its result.
The detector's own tolerances are the ``--isospin-...`` options
(`baryon_poles.ISOSPIN_TOLERANCES`).
"""

import argparse
import json
import sys

import numpy as np

from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp


def sheet_layout():
    """The sheet and base edge of each of the host's 18 edge cells, in the
    canonical sheet-major order (`baryon_poles.canonical_edges`)."""
    cells = bp.canonical_edges()
    sheet_of = [a // 4 for a, _ in cells]
    base_of = [i % bp.BASE_EDGES for i in range(len(cells))]
    return cells, sheet_of, base_of


def symmetry(tolerance=bp.DECLARED_CERTIFICATE_TOLERANCE):
    """D_1(g) on the 18 cells for the twelve rotations, and whether the
    projective class is nontrivial, the cocycle read at ``tolerance``."""
    support = bp.monopole_support()
    group = bp.rotation_group()
    actions = bp.rotation_action([support] * bp.SHEETS)
    return actions, bool(support.cocycle(group, 1, tolerance).nontrivial)


def declaration(operator, name, actions, spinorial):
    """The detector's declaration for one operator of the host, one frame."""
    cells, sheet_of, base_of = sheet_layout()
    d = obs.IsospinDoubletDeclaration()
    d.operator_name = name
    d.degree = 1
    d.cells = [list(c) for c in cells]
    d.sheet_of_cell = sheet_of
    d.base_cell_of_cell = base_of
    d.symmetry = [np.asarray(a) for a in actions]
    d.symmetry_name = "T = A_4, projective action D_1(g) of the unit monopole"
    d.spinorial = spinorial
    d.frames = [obs.IsospinFrame("single-level synthesis", np.asarray(operator))]
    return d


def _status(read):
    return {
        "number": int(read.number), "name": read.name,
        "status": read.status.name,
        "missing": list(read.missing), "failing": list(read.failing),
        "evidence": [{"name": e.name, "held": e.held, "detail": e.detail}
                     for e in read.evidence],
    }


def unread_tolerances(read):
    """The tolerances of the detector that decided nothing in ``read``
    (`baryon_poles.ISOSPIN_NEEDS`, by config key), each with what it needs
    and the read did not have: the span tolerance without a doublet
    candidate followed to a further frame or resolution, the transport
    tolerances without a transport between two frames, and the member
    splitting and occupation tolerances without an observed doublet. On
    the one frame `declaration` gives the detector it is all seven."""
    followed = bool(read.candidates) and (len(read.frames) > 1
                                          or len(read.resolutions) > 0)
    transported = any(len(c.transports) > 0 for c in read.candidates)
    observed = any(c.observed for c in read.candidates)
    had = {bp.ISOSPIN_NEEDS["isospin_span_tolerance"]: followed,
           bp.ISOSPIN_NEEDS["isospin_transport_leakage_tolerance"]:
               transported,
           bp.ISOSPIN_NEEDS["isospin_occupation_tolerance"]: observed}
    return {key: "no %s in this read" % need
            for key, need in bp.ISOSPIN_NEEDS.items() if not had[need]}


def record(read):
    """A JSON-able record of one `IsospinDoubletRead`, with the tolerances
    of the detector that decided nothing in it (`unread_tolerances`)."""
    frames = []
    for frame in read.frames:
        frames.append({
            "label": frame.label,
            "spectrum": [complex(v) for v in frame.spectrum],
            "bands": [{
                "index": int(b.index),
                "center": complex(b.center),
                "rank": int(b.rank),
                "gap": float(b.gap),
                "isolated": bool(b.isolated),
                "projector_residual": float(b.projector_residual),
                "colour_acts": bool(b.colour_acts),
                "sheet_invariance_residual": float(b.sheet_invariance_residual),
                "symmetry_acts": bool(b.symmetry_acts),
                "symmetry_invariance_residual":
                    float(b.symmetry_invariance_residual),
                "commutant_dimension": int(b.commutant_dimension),
                "content": b.content,
                "spin_doublet": bool(b.spin_doublet),
                "doublet_candidate": bool(b.doublet_candidate),
                "unexplained_multiplicity": bool(b.unexplained_multiplicity),
                "classification": b.classification,
            } for b in frame.bands],
        })
    candidates = []
    for c in read.candidates:
        entry = {
            "band_index": int(c.band_index),
            "tracked_bands": [int(x) for x in c.tracked_bands],
            "conditions": [_status(x) for x in c.conditions],
            "observed": bool(c.observed),
        }
        if c.charges is not None:
            entry["charges"] = {
                "member_source": c.charges.member_source,
                "isospin": list(c.charges.isospin),
                "baryon_number": c.charges.baryon_number,
                "charges": list(c.charges.charges),
                "occupation_pattern": c.charges.occupation_pattern,
                "notes": list(c.charges.notes),
            }
        candidates.append(entry)
    return {
        "operator": read.operator_name,
        "symmetry": read.symmetry_name,
        "frames": frames,
        "candidates": candidates,
        "conditions": [_status(x) for x in read.conditions],
        "doublet_observed": bool(read.doublet_observed),
        "falsifier_8_no_isospin_doublet": bool(read.no_isospin_doublet),
        "falsifier_10_unexplained_multiplicities":
            list(read.unexplained_multiplicities),
        "falsifier_10_refinement_measured":
            bool(read.multiplicity_refinement_measured),
        "summary": read.summary,
        "unread_tolerances": unread_tolerances(read),
    }


def observe_host(carrier, actions=None, spinorial=None, config=None):
    """The detector on the covariant carrier operator h_1(z, U) of the host and
    on its T-average, at the tolerances of ``config``
    (`baryon_poles.isospin_doublet_config`; the declared values when there is
    no config)."""
    if actions is None:
        actions, spinorial = symmetry(
            bp.declared_tolerance(config, "certificate_tolerance"))
    carrier = np.asarray(carrier)
    averaged = bp.rotation_averaged(carrier, actions)
    detector = bp.isospin_doublet_config(config)
    out = {}
    for key, name, operator in (
            ("covariant", "covariant h_1(z, U)", carrier),
            ("t_averaged", "T-averaged h_1(z, U) (WP line 497)", averaged)):
        read = obs.IsospinDoublet.observe(
            declaration(operator, name, actions, spinorial), detector)
        out[key] = record(read)
    return out


def declared_carrier(edge_squared=bp.DECLARED_EDGE_SQUARED,
                     villain_order=bp.DECLARED_VILLAIN_ORDER):
    """h_1(z, U) of the declared (unrelaxed) host. The carrier operator does
    not depend on the action's weights or on the order of its Villain weight,
    so the declaration's kappa and beta are placeholders."""
    spacetime = bp.build_host(edge_squared)
    action = cob.JointAction(
        spacetime, bp.action_declaration(spacetime, 1.0, 1.0,
                                         villain_order=villain_order))
    return bp.matrix(action.carrier_operator())


def relaxed_carrier(content, kappa, beta, config):
    """h_1(z, U) of the host relaxed to self-consistency for one content, as
    `baryon_poles.relax_content` does it."""
    _, action, report, _ = bp.relax_content(content, kappa, beta, config)
    return bp.matrix(action.carrier_operator()), report


def drive(kappas=None, betas=None, contents=None,
          edge_squared=bp.DECLARED_EDGE_SQUARED, progress=False,
          villain_order=bp.DECLARED_VILLAIN_ORDER, tolerances=None,
          limits=None):
    """The declared host's read and, with ``kappas``, the read of every
    relaxed host. ``villain_order`` is the order the Villain weight is summed
    to; ``tolerances`` sets any of `baryon_poles.TOLERANCES` by key and
    ``limits`` declares any of `baryon_poles.LIMITS` by key; the values of
    every one are recorded in the result. A content that is not one of the
    ten (`baryon_poles.checked_contents`) is refused with the config,
    before anything is computed."""
    villain_order = bp.checked_villain_order(villain_order)
    config = bp.default_config(kappas or list(bp.DECLARED_KAPPAS),
                               betas or list(bp.DECLARED_BETAS),
                               edge_squared, tolerances=tolerances,
                               limits=limits, villain_order=villain_order,
                               selected_contents=contents)
    actions, spinorial = symmetry(
        bp.declared_tolerance(config, "certificate_tolerance"))
    result = {
        "declared_host": observe_host(
            declared_carrier(edge_squared, villain_order), actions,
            spinorial, config),
        "spinorial": spinorial,
        "villain_order": villain_order,
        "relaxed": [],
        "tolerances": {key: config[key] for key, _ in bp.TOLERANCES},
        "limits": {key: config[key] for key, _, _ in bp.LIMITS},
    }
    if progress:
        _print("declared host", result["declared_host"])
    if kappas:
        for kappa in kappas:
            for beta in (betas or list(bp.DECLARED_BETAS)):
                for content in config["contents"]:
                    carrier, report = relaxed_carrier(content, kappa, beta,
                                                      config)
                    entry = {"kappa": kappa, "beta": beta,
                             "content": list(content),
                             "relaxation_converged": bool(report.converged),
                             "reads": observe_host(carrier, actions,
                                                   spinorial, config)}
                    result["relaxed"].append(entry)
                    if progress:
                        _print("kappa=%g beta=%g content=%s"
                               % (kappa, beta, list(content)), entry["reads"])
    return result


def _print(head, reads):
    for key in ("covariant", "t_averaged"):
        r = reads[key]
        conditions = ", ".join("%s %s" % (c["name"], c["status"])
                               for c in r["conditions"])
        sys.stdout.write("%s %s: %s\n  %s\n" % (head, key, conditions,
                                                r["summary"]))
        for b in r["frames"][0]["bands"]:
            sys.stdout.write("    band %d at %s rank %d: %s\n"
                             % (b["index"], b["center"], b["rank"],
                                b["classification"]))
    sys.stdout.flush()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tessera.drivers.isospin_doublet",
        description="Observe the isospin doublet (WP v16 Section 10) on the "
                    "three-sheeted unit-monopole host.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="read the declared host, and "
                                          "optionally relaxed hosts")
    run.add_argument("--kappa", type=float, nargs="+", default=None,
                     help="kappa = 8 pi G of each relaxed host; without it "
                          "only the declared host is read")
    run.add_argument("--beta", type=float, nargs="+", default=None,
                     help="beta of the holonomy term of each relaxed host "
                          "(default %s)" % (list(bp.DECLARED_BETAS),))
    run.add_argument("--content", type=int, nargs=3, action="append",
                     default=None, help="a content (quarks per band); "
                                        "repeatable; default all ten")
    run.add_argument("--edge-squared", type=float,
                     default=bp.DECLARED_EDGE_SQUARED,
                     help="the squared edge length of the declared host "
                          "(default %g)" % bp.DECLARED_EDGE_SQUARED)
    bp.add_action_arguments(run)
    run.add_argument("--json", default=None,
                     help="write the record here, with the environment of "
                          "the run")
    bp.add_tolerance_arguments(run)
    bp.add_limit_arguments(run)
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = drive(args.kappa, args.beta, args.content, args.edge_squared,
                   progress=not args.quiet, villain_order=args.villain_order,
                   tolerances=bp.tolerances_from(args),
                   limits=bp.limits_from(args))
    # what the read depends on beside its declarations: the command line,
    # the commit, the thread counts and the linear algebra libraries
    result["environment"] = bp.environment_record(argv)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(bp._jsonable(result), handle, indent=1,
                      allow_nan=False)
    return result


if __name__ == "__main__":
    main()
