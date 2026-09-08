# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Does a solved bulk act algebraically when the input states are swapped?

The question (#999). A qubit cobordism is driven until the transfer
``T = (Z_A^v)^T A~_1 Z_B`` -- the coupling block of the WHOLE complex's
degree-1 Laplacian, read in the two boundary tori's period frames -- sits
close to a target ``chi``. The target is a BILINEAR function of the two input
states: for the XY flip-flop, ``chi(psi, phi) = (s- psi)(s+ phi)^T +
(s+ psi)(s- phi)^T``. So a solved bulk has fitted one VALUE of a bilinear
map, and the sharp question is whether it carries the map itself.

The test. Take the solved complex. Hold every bulk edge at the length and
phase the relaxation left it. Replace the boundary tori with new ones, which
in this construction means new edge lengths, since a state IS a conformal
structure. Relax nothing, move nothing else. Recompute the transfer in the
new period frames and ask whether it matches ``chi`` of the NEW pair.

Why this is not trivially true. The boundary lengths enter the mass matrices,
so replacing them changes ``A~_1`` and therefore changes ``T``. ``T'`` is not
related to ``T`` by any linear substitution. The hypothesis is that the
period frames absorb exactly the boundary's change and leave the bulk acting
as the same bilinear map; the measurement is the deviation from that.

Two controls, because "swap the states" hides a choice:

* **replay** -- put back the solved boundary lengths, edge for edge, out of
  the dump. This must reproduce the solved residual to rounding. It checks
  the harness, not the physics.
* **reflat** -- install a FRESH flat torus at the ORIGINAL tau. Its state is
  the one the bulk was solved against, but its lengths are not: the D2
  residual holds tau_hat, and the individual lengths drift under it. The gap
  between `replay` and `reflat` is how much the bulk depends on the boundary's
  conformal REPRESENTATIVE rather than on its state, and it bounds what any
  swap can be expected to achieve.

And the attachments. A torus is glued to the bulk by a map from its vertices
to host vertex ids. The flat grid's translations carry the marking to itself,
so each of them is a different gluing of the SAME torus to the SAME bulk. If
the answer depends on which was used, the read is an artifact of the gluing
and not a property of the geometry, which is why every one is measured.

Nothing here relaxes, drives or fits. It loads a solved geometry, installs
boundary lengths, and reads.
"""
import argparse
import cmath
import itertools
import json
import sys
import warnings

import numpy as np

import tessera
from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
Spacetime = tessera.spacetime.Spacetime

#: The lowering operator; its transpose raises. `s-|0> = |1>`.
LOWERING = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


def state_of(tau):
    """`(1, tau)/sqrt(1+|tau|^2)`: the qubit a torus of modulus tau is."""
    tau = complex(tau)
    return np.array([1.0, tau], dtype=complex) / np.sqrt(1.0 + abs(tau) ** 2)


def flip_flop(psi, phi):
    """chi of spec S5: the XY flip-flop's image of the product state."""
    raising = LOWERING.T
    return (np.outer(LOWERING @ psi, raising @ phi)
            + np.outer(raising @ psi, LOWERING @ phi))


def ising(psi, phi):
    """`s1z s2z`'s image: diagonal, Schmidt rank one."""
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.outer(sigma_z @ psi, sigma_z @ phi)


def exchange(psi, phi):
    """`s1 . s2`'s image: the flip-flop plus the Ising term."""
    return flip_flop(psi, phi) + ising(psi, phi)


def product(psi, phi):
    """The quasi-free map: the product state itself, Schmidt rank one."""
    return np.outer(psi, phi)


#: The operators an experiment may fit, each a bilinear map of the two states.
OPERATORS = {"flip_flop": flip_flop, "ising": ising, "exchange": exchange,
             "product": product}


def projective_residual(transfer, target):
    """`min_c ||c T - chi||^2 / ||chi||^2`, the engine's own scoring: the
    squared sine of the angle between the two matrices as vectors under the
    Hilbert-Schmidt inner product, so the scale of T is not scored."""
    transfer = np.asarray(transfer, dtype=complex)
    target = np.asarray(target, dtype=complex)
    tt = float(np.vdot(transfer, transfer).real)
    cc = float(np.vdot(target, target).real)
    if not (tt > 0.0) or not (cc > 0.0):
        return 1.0
    overlap = np.vdot(transfer, target)
    return max(0.0, 1.0 - abs(overlap) ** 2 / (tt * cc))


def load(path):
    with open(path) as handle:
        return json.load(handle)


def rebuild_spacetime(document):
    """The dump's own rebuild path: cells, then times, lengths and phases."""
    spacetime = Spacetime.fromCells(document["dimensions"], document["cells"])
    vertices = spacetime.getVertexList()
    for vid, t in document["vertex_times"]:
        vertices.get(int(vid)).setTime(float(t))
    by_pair = edges_by_pair(spacetime)
    for u, v, re_l2, im_l2 in document["edges"]:
        by_pair[key_of(u, v)].setLength(cmath.sqrt(complex(re_l2, im_l2)))
    for u, v, re_p, im_p in document.get("edge_phases", []):
        by_pair[key_of(u, v)].setPhase(complex(re_p, im_p))
    spacetime.materializeFacets()
    return spacetime


def key_of(u, v):
    return (min(int(u), int(v)), max(int(u), int(v)))


def edges_by_pair(spacetime):
    out = {}
    for edge in spacetime.getEdgeList().toVector():
        out[key_of(edge.getSource().getId(), edge.getTarget().getId())] = edge
    return out


def torus_vertex_order(block):
    """The torus's own vertex indices in the order `SimplicialQubit` uses,
    paired with the host ids they are glued to.

    A torus is built by `SimplicialQubit.flat_torus`, whose vertices are
    `0 .. n*n-1`; the collar glues vertex `i` to a host id. The dump records
    the host ids of the block, sorted, and the flat torus indexes its own
    vertices in the same ascending order, so position in the sorted host list
    IS the torus's vertex index. That is the identity attachment; a
    permutation composes with it.
    """
    return sorted(int(v) for v in block["vertices"])


def flat_lengths(tau, grid, texture=0.0):
    """A torus of modulus `tau`: its edges as (i, j) index pairs with their
    lengths, in the torus's own vertex indexing.

    `texture` breaks the lattice symmetry WITHOUT moving the state. A flat
    torus gives every edge of a lattice direction the same length, so a
    lattice translation maps its length assignment to itself: nine gluings
    of it are nine relabelings of an object that cannot tell them apart, and
    measuring the attachment against it measures nothing. Scaling the
    lengths by `exp(texture * f(midpoint))` for a mean-zero f is a discrete
    conformal factor: it varies edge to edge, so the translations become
    genuinely different gluings, while tau is a conformal invariant and the
    state is unchanged to the construction's mesh order. The residual it
    leaves is reported, so the reader can see how far "unchanged" holds.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torus = obs.SimplicialQubit.flat_torus(complex(tau), grid, grid)
    lengths = {(int(i), int(j)): complex(length)
               for (i, j), length in zip(torus.edges(), torus.lengths())}
    if texture == 0.0:
        return lengths, torus
    def wrapped(delta):
        return -1 if delta == grid - 1 else (1 if delta == -(grid - 1) else delta)
    scaled = {}
    for (i, j), length in lengths.items():
        ri, ci = divmod(i, grid)
        rj, cj = divmod(j, grid)
        x = (ri + 0.5 * wrapped(rj - ri)) / grid
        y = (ci + 0.5 * wrapped(cj - ci)) / grid
        factor = np.exp(texture * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
        scaled[(i, j)] = length * factor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        textured = obs.SimplicialQubit(
            list(torus.vertices()), [tuple(e) for e in torus.edges()],
            [tuple(f) for f in torus.faces()],
            [scaled[(int(i), int(j))] for i, j in torus.edges()],
            list(torus.cycle_A()), list(torus.cycle_B()))
    return scaled, textured


def install(spacetime, block, tau, grid, permutation=None, texture=0.0):
    """Put a fresh flat torus of modulus `tau` onto this block's edges.

    `permutation` maps a torus vertex index to a torus vertex index; it is the
    gluing, composed with the identity attachment of `torus_vertex_order`. The
    bulk is never touched: only edges whose two ends are both inside the
    block's vertex set are written, which is exactly the block's surface.
    """
    hosts = torus_vertex_order(block)
    permutation = list(range(len(hosts))) if permutation is None else list(permutation)
    host_of = {index: hosts[permutation[index]] for index in range(len(hosts))}
    lengths, torus = flat_lengths(tau, grid, texture)
    by_pair = edges_by_pair(spacetime)
    written = 0
    for (i, j), length in lengths.items():
        pair = key_of(host_of[i], host_of[j])
        edge = by_pair.get(pair)
        if edge is None:
            raise KeyError("the permuted torus needs a host edge %s that the "
                           "solved complex does not have" % (pair,))
        edge.setLength(length)
        written += 1
    if written != len(lengths):
        raise RuntimeError("wrote %d of %d torus edges" % (written, len(lengths)))
    return torus


def replay(spacetime, block, permutation=None):
    """Put the SOLVED boundary lengths back, out of the dump.

    With `permutation` the same lengths are laid down under a different
    gluing: each recorded host edge is read back as a pair of TORUS vertex
    indices (position in the block's sorted host ids, which is how the torus
    was attached), permuted, and written to the host edge that pair now names.
    The torus is intrinsically identical either way -- same lengths, same
    tau -- so this isolates the attachment from everything else.
    """
    by_pair = edges_by_pair(spacetime)
    if permutation is None:
        for u, v, re_l2, im_l2 in block["surface"]["edges"]:
            by_pair[key_of(u, v)].setLength(cmath.sqrt(complex(re_l2, im_l2)))
        for u, v, re_p, im_p in block["surface"].get("edge_phases", []):
            by_pair[key_of(u, v)].setPhase(complex(re_p, im_p))
        return
    hosts = torus_vertex_order(block)
    torus_of = {host: n for n, host in enumerate(hosts)}
    host_of = {n: hosts[permutation[n]] for n in range(len(hosts))}
    for u, v, re_l2, im_l2 in block["surface"]["edges"]:
        pair = key_of(host_of[torus_of[int(u)]], host_of[torus_of[int(v)]])
        edge = by_pair.get(pair)
        if edge is None:
            raise KeyError("the permuted replay needs a host edge %s the solved "
                           "complex does not have" % (pair,))
        edge.setLength(cmath.sqrt(complex(re_l2, im_l2)))
    for u, v, re_p, im_p in block["surface"].get("edge_phases", []):
        pair = key_of(host_of[torus_of[int(u)]], host_of[torus_of[int(v)]])
        by_pair[pair].setPhase(complex(re_p, im_p))


def node_on(spacetime, document, weight=1e4):
    """A node on this complex carrying the dump's blocks and markings.

    Nothing is driven: the node exists to READ, which needs the input blocks,
    their state fibers and their markings, because the transfer is taken in
    the frames those markings normalize.
    """
    node = MC(spacetime, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=0,
              einstein_hilbert=False, real_squared_lengths_only=False,
              metric_source=cob.HodgeMetricSource.WhitneyPencil)
    node.seed_inputs([block["vertices"] for block in document["blocks"]])
    node.use_fiber_residuals(True)
    node.set_input_residual_weight(weight)
    for index, block in enumerate(document["blocks"]):
        cells = sorted(key_of(u, v) for u, v, _re, _im in block["surface"]["edges"])
        fiber = cob.BoundaryFiber()
        fiber.degree = 1
        fiber.cells = [list(cell) for cell in cells]
        # One column, any nonzero: the state a block represents is its
        # marking's coefficients, and the fiber's images are not read once a
        # marking is set. It is attached because a marking requires one.
        fiber.images = np.ones((len(cells), 1), dtype=complex)
        fiber.contour = cob.PencilLayer.harmonic_contour(
            cob.PencilLayer.assemble([spacetime]), 1)
        node.attach_input_fiber(index, fiber, fiber.cells)
    return node


def mark(node, document, taus):
    """Set each block's marking with the input coefficients (1, tau)."""
    for index, block in enumerate(document["blocks"]):
        cycles = [[tuple(int(v) for v in step) for step in cycle]
                  for cycle in block["marking"]]
        node.set_input_marking(index, cycles, [1.0 + 0j, complex(taus[index])])


def read(document, taus, operator, grid, permutations=(None, None),
         mode="flat", weight=1e4, texture=0.0, derive_states=False,
         frames=True):
    """One reading: install the boundary, take the transfer, score it.

    `mode` is `flat` (fresh tori at `taus`) or `replay` (the solved lengths
    back, in which case `taus` and `permutations` are ignored for the
    install and used only for the marking).
    """
    spacetime = rebuild_spacetime(document)
    tori = []
    for index, block in enumerate(document["blocks"]):
        if mode == "replay":
            replay(spacetime, block, permutations[index])
            tori.append(None)
        else:
            tori.append(install(spacetime, block, taus[index], grid,
                                permutations[index], texture))
    spacetime.materializeFacets()
    node = node_on(spacetime, document, weight)
    if not frames:
        # No marking and no frame on either block: `read_two_body` then takes
        # unit images on the attached cells, so the transfer IS the raw
        # coupling block of the whole's degree-1 Laplacian between the two
        # tori's edge sets -- the geometry's coupling with no readout
        # convention on it. It has no 2x2 target; it is for comparing a
        # geometry with itself across a change.
        out = {"mode": mode, "frames": False,
               "permutations": [None if p is None else list(p) for p in permutations]}
        try:
            read = node.read_two_body()
            out["transfer"] = np.asarray(read.transfer, dtype=complex)
            out["in_frames"] = bool(read.in_frames)
            out["cells_a"] = [list(c) for c in read.cells_a]
            out["cells_b"] = [list(c) for c in read.cells_b]
        except Exception as error:                        # noqa: BLE001
            out["refused"] = str(error)
        return out
    mark(node, document, taus)
    if derive_states:
        # The state the ATTACHMENT presents, not the one we asked for. A
        # gluing that carries the marked homology classes to a different pair
        # makes the host's marking read a modular image of tau, so asserting
        # the original would compare a transfer taken in one marking against
        # a target written in another. Reading tau back and rebuilding both
        # the coefficients and the target from it keeps the two in the same
        # marking, and leaves the own-state residuals at zero, which is the
        # check that it worked.
        taus = [complex(node.block_qubit(i).tau()) for i in range(2)]
        mark(node, document, taus)
    out = {"mode": mode, "taus": [complex(t) for t in taus], "texture": float(texture),
           "derive_states": bool(derive_states),
           "permutations": [None if p is None else list(p) for p in permutations]}
    out["installed_tau"] = [None if t is None else complex(t.tau()) for t in tori]
    try:
        transfer = np.asarray(node.read_two_body().transfer, dtype=complex)
    except Exception as error:                            # noqa: BLE001
        out["refused"] = str(error)
        return out
    psi, phi = state_of(taus[0]), state_of(taus[1])
    target = OPERATORS[operator](psi, phi)
    out["transfer"] = transfer
    out["target"] = target
    out["residual"] = projective_residual(transfer, target)
    out["singular_values"] = list(np.linalg.svd(transfer, compute_uv=False))
    out["target_singular_values"] = list(np.linalg.svd(target, compute_uv=False))
    out["own_state_residuals"] = [node.own_state_residual(i) for i in range(2)]
    out["tau_hat"] = [complex(node.block_qubit(i).tau()) for i in range(2)]
    return out


def translations(grid):
    """The gluings a flat `grid x grid` torus admits that carry its marking
    to itself: the lattice translations, `grid^2` of them. Vertex `(r, c)` of
    the grid is index `r * grid + c`, and a translation shifts both."""
    out = []
    for dr in range(grid):
        for dc in range(grid):
            out.append(tuple(((r + dr) % grid) * grid + ((c + dc) % grid)
                             for r in range(grid) for c in range(grid)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("geometry", help="the solved geometry dump (--geometry)")
    parser.add_argument("--operator", default="flip_flop", choices=sorted(OPERATORS))
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--weight", type=float, default=1e4)
    parser.add_argument("--tau-a", type=complex, default=None,
                        help="the new modulus for torus A (default: its own)")
    parser.add_argument("--tau-b", type=complex, default=None)
    parser.add_argument("--permutations", action="store_true",
                        help="repeat over every lattice translation of torus A")
    parser.add_argument("--texture", type=float, default=0.0,
                        help="a discrete conformal factor on the installed "
                             "tori, breaking the lattice symmetry so the "
                             "translations become different gluings; tau is "
                             "unchanged, and the installed tau is reported")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    document = load(args.geometry)
    original = [complex(*block["tau_in"]) if isinstance(block["tau_in"], list)
                else complex(block["tau_in"]) for block in document["blocks"]]
    new = [args.tau_a if args.tau_a is not None else original[0],
           args.tau_b if args.tau_b is not None else original[1]]

    results = [read(document, original, args.operator, args.grid, mode="replay",
                    weight=args.weight),
               read(document, original, args.operator, args.grid, mode="flat",
                    weight=args.weight, texture=args.texture)]
    results[0]["label"] = "replay: the solved boundary, back"
    results[1]["label"] = "reflat: a fresh flat torus at the original tau"
    if new != original:
        swapped = read(document, new, args.operator, args.grid, mode="flat",
                       weight=args.weight, texture=args.texture)
        swapped["label"] = "swap: fresh tori at the new tau"
        results.append(swapped)
    if args.permutations:
        for n, permutation in enumerate(translations(args.grid)):
            entry = read(document, new, args.operator, args.grid,
                         permutations=(permutation, None), mode="flat",
                         weight=args.weight, texture=args.texture)
            entry["label"] = "swap under translation %d of torus A" % n
            results.append(entry)

    for entry in results:
        if "refused" in entry:
            print("%-46s REFUSED %s" % (entry["label"], entry["refused"]))
            continue
        print("%-46s residual %.9e  own %.2e %.2e  tau_hat %s"
              % (entry["label"], entry["residual"],
                 entry["own_state_residuals"][0], entry["own_state_residuals"][1],
                 ", ".join("%.6f" % t.real + ("%+.6fi" % t.imag) for t in entry["tau_hat"])))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else
                            [complex(z).__repr__() for z in v] if k.endswith("hat") or k == "taus" else v)
                        for k, v in entry.items()} for entry in results],
                      handle, indent=2, default=repr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
