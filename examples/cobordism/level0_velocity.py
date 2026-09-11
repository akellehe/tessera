# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Does a level-0 cobordism carry the velocity? (#953, epic #938)

One layer, one question. The interaction node W carries two input fibers on
two vertex-disjoint tetrahedra of its own complex and is asked to carry the
velocity χ = B(ψ⊗φ) of the XY flip-flop as the frame transfer T_AB between
them. The stated procedure is followed in two phases:

1. the blocks carry their fibers (only the block residuals drive; the block
   regions may grow);
2. the block regions are PINNED (their geometry no longer moves) and the bulk
   alone is grown by gated cone-ins and relaxed by the analytic gradient
   against χ.

The leak of χ in T_AB is measured against the bulk's size and its
connectivity between the two frames (cells adjacent to each frame and to
both; the graph distance between the frames), with the transfer's Schmidt
spectrum and the metric conditioning of the read (σ_max/σ_min of M₀ᵁ and M₁ᵁ,
the certificate a 1.2e7 read lacked in #943's records). Every vertex-disjoint
attachment pair is ranked by the transfer's σ₂/σ₁ at the start, and the best
few are tried.

Run:  python examples/cobordism/level0_velocity.py --output ~/cobordism-runs/level0/run.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from tessera import cobordism as cob

MC = cob.MultiCobordism
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])
SEED_CELLS = [[0], [1], [2], [3]]


def _unit(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


def _payload(x):
    a = np.asarray(x, dtype=complex)
    if a.ndim == 1:
        return [[float(z.real), float(z.imag)] for z in a]
    return [[[float(z.real), float(z.imag)] for z in row] for row in a]


def flip_flop(psi, phi):
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return np.outer(D @ psi, D.T @ phi) + np.outer(D.T @ psi, D @ phi)


def fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = SEED_CELLS
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def frame_transfer_block(st, a, b):
    assembled = cob.PencilLayer.assemble([st])
    A = dense(cob.PencilLayer.pencil(assembled, 0).A)
    verts = [int(v[0]) for v in assembled.complex.kSimplexVertices(0)]
    idx = {v: i for i, v in enumerate(verts)}
    return A[np.ix_([idx[v] for v in a], [idx[v] for v in b])]


def metric_conditioning(st):
    assembled = cob.PencilLayer.assemble([st])
    out = {}
    for k in (0, 1):
        sv = np.linalg.svd(dense(assembled.op.Minv(k)), compute_uv=False)
        out[f"M{k}_condition"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return out


def connectivity(st, a, b):
    K = cob.ChainComplex.fromSpacetime(st)
    tets = [tuple(int(v) for v in t) for t in K.kSimplexVertices(3)]
    sa, sb = set(a), set(b)
    adjacent_a = sum(1 for t in tets if set(t) & sa)
    adjacent_b = sum(1 for t in tets if set(t) & sb)
    adjacent_both = sum(1 for t in tets if set(t) & sa and set(t) & sb)
    edges = [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]
    graph = {}
    for u, v in edges:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    # shortest path between the frames in the 1-skeleton
    dist = {v: 0 for v in a}
    queue = deque(a)
    best = None
    while queue:
        u = queue.popleft()
        if u in sb:
            best = dist[u]
            break
        for w in graph.get(u, ()):
            if w not in dist:
                dist[w] = dist[u] + 1
                queue.append(w)
    return {"cells_adjacent_to_a": adjacent_a, "cells_adjacent_to_b": adjacent_b,
            "cells_adjacent_to_both": adjacent_both, "frame_distance": best, "tetrahedra": len(tets),
            "vertices": int(st.getVertexList().size())}


def ranked_pairs(st):
    K = cob.ChainComplex.fromSpacetime(st)
    tets = [tuple(int(v) for v in t) for t in K.kSimplexVertices(3)]
    scored = []
    for a, b in itertools.combinations(tets, 2):
        if set(a) & set(b):
            continue
        sv = np.linalg.svd(frame_transfer_block(st, a, b), compute_uv=False)
        scored.append((float(sv[1] / sv[0]) if sv[0] > 0 else 0.0, a, b, sv))
    scored.sort(key=lambda t: -t[0])
    return scored


def projective_leak(T, chi):
    tt = np.vdot(T, T).real
    if tt == 0.0:
        return 1.0
    return float(1.0 - abs(np.vdot(T, chi)) ** 2 / (tt * np.linalg.norm(chi) ** 2))


def block_region(st, tet, shells):
    """The attached tetrahedron plus `shells` shells of top cells around it."""
    K = cob.ChainComplex.fromSpacetime(st)
    tets = [set(int(v) for v in t) for t in K.kSimplexVertices(3)]
    region = set(tet)
    for _ in range(shells):
        region |= set().union(*[t for t in tets if t & region]) if any(t & region for t in tets) else set()
    return region


def free_edge_count(node):
    return sum(1 for e in node.spacetime().getEdgeList().toVector()
               if not node.edge_is_pinned(int(e.getSource().getId()), int(e.getTarget().getId())))


def run_configuration(psi, phi, chi, seed, precone, pair, args):
    if args.carrier == "whole":
        return run_whole_carrier(psi, phi, chi, seed, precone, pair, args)
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=seed, precone=precone, einstein_hilbert=False)
    node.seed_inputs([0, 1])
    a, b = pair
    node.attach_input_fiber(0, fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, fiber(phi), [[v] for v in b])
    # Bounded block regions: the attached tetrahedron plus `--shells` shells of
    # cells, FIXED (emergent region growth would swallow the bulk and pinning
    # the blocks would then pin everything, which is what the first records did).
    regions = [block_region(node.spacetime(), a, args.shells), block_region(node.spacetime(), b, args.shells)]
    for i, region in enumerate(regions):
        node.set_input_block_region(i, region)
    node.use_fiber_residuals(True)
    record = {"seed": seed, "precone": precone, "pair": [list(a), list(b)],
              "start": {**connectivity(node.spacetime(), a, b),
                        "transfer_singular_values": [float(s) for s in np.linalg.svd(frame_transfer_block(node.spacetime(), a, b), compute_uv=False)]}}
    # Phase 1: the blocks carry their fibers.
    t0 = time.time()
    trace1 = [[node.fiber_residual_for_input_block(i) for i in range(2)]]
    for _ in range(args.phase1_rounds):
        if max(trace1[-1]) < args.block_tolerance:
            break
        node.run_stage1(max_steps=args.stage1_steps, n_candidate_moves=args.candidates)
        node.run_stage2(beta=1.0, max_iters=args.stage2_iters, tolerance=1e-15)
        trace1.append([node.fiber_residual_for_input_block(i) for i in range(2)])
    record["phase1"] = {"block_residual_trace": [[float(x) for x in r] for r in trace1],
                        "seconds": round(time.time() - t0, 1),
                        "block_regions": [sorted(int(v) for v in blk.vertices) for blk in node.inputs]}
    # Phase 2: pin the blocks, synthesize the bulk alone against chi.
    for i, blk in enumerate(node.inputs):
        node.declare_pinned_region(f"block_{i}", set(int(v) for v in blk.vertices))
    record["phase2_free_edges"] = free_edge_count(node)
    record["phase2_total_edges"] = int(node.spacetime().getEdgeList().size())
    node.set_two_body_target(chi, args.choi)
    t0 = time.time()
    trace2 = [node.two_body_residual()]
    for _ in range(args.phase2_rounds):
        if trace2[-1] < args.tolerance:
            break
        node.run_stage1(max_steps=args.stage1_steps, n_candidate_moves=args.candidates)
        node.run_stage2(beta=1.0, max_iters=args.stage2_iters, tolerance=1e-15)
        trace2.append(node.two_body_residual())
    read = node.read_two_body()
    T = np.asarray(read.transfer)
    record["phase2"] = {"leak_trace": [float(x) for x in trace2], "seconds": round(time.time() - t0, 1),
                        "leak": float(read.residual), "leak_recomputed": projective_leak(T, chi),
                        "schmidt": [float(s) for s in read.singular_values],
                        "block_residuals_after": [float(x) for x in read.input_fiber_residuals],
                        "reversal_residual": float(read.reversal_residual),
                        **connectivity(node.spacetime(), a, b), **metric_conditioning(node.spacetime()),
                        "transfer": _payload(T)}
    return record


def run_whole_carrier(psi, phi, chi, seed, precone, pair, args):
    """The whole complex carries the pair's one-particle content as ONE band
    restricting to ψ on frame A and φ on frame B (the emergent analog of the
    coupled boundary-state transfer that carried a one-particle operator at
    level 0), and the bulk — every edge outside the two frames — is
    synthesized against χ at the same time. The block residuals are the
    constant a single tetrahedron gives and do not compete."""
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=seed, precone=precone, einstein_hilbert=False)
    node.seed_inputs([0, 1])
    a, b = pair
    node.attach_input_fiber(0, fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, fiber(phi), [[v] for v in b])
    node.set_input_block_region(0, set(a))
    node.set_input_block_region(1, set(b))
    joint = cob.BoundaryFiber()
    joint.degree = 0
    joint.cells = [[v] for v in a] + [[v] for v in b]
    joint.images = np.concatenate([np.asarray(psi, dtype=complex), np.asarray(phi, dtype=complex)]).reshape(8, 1)
    node.set_whole_complex_fiber_target(joint)
    node.use_fiber_residuals(True)
    record = {"seed": seed, "precone": precone, "pair": [list(a), list(b)], "carrier": "whole",
              "start": {**connectivity(node.spacetime(), a, b),
                        "transfer_singular_values": [float(s) for s in np.linalg.svd(frame_transfer_block(node.spacetime(), a, b), compute_uv=False)],
                        "whole_fiber_residual": float(node.whole_complex_fiber_residual())}}
    # Phase 1: the whole complex carries the pair's one-particle content.
    t0 = time.time()
    trace1 = [node.whole_complex_fiber_residual()]
    for _ in range(args.phase1_rounds):
        if trace1[-1] < args.block_tolerance:
            break
        node.run_stage1(max_steps=args.stage1_steps, n_candidate_moves=args.candidates)
        node.run_stage2(beta=1.0, max_iters=args.stage2_iters, tolerance=1e-15)
        trace1.append(node.whole_complex_fiber_residual())
    record["phase1"] = {"whole_fiber_trace": [float(x) for x in trace1], "seconds": round(time.time() - t0, 1)}
    # Phase 2: the frames' own edges pinned; the bulk synthesized against chi
    # while the whole keeps carrying the pair (both terms in r_U).
    node.declare_pinned_region("frame_a", set(a))
    node.declare_pinned_region("frame_b", set(b))
    record["phase2_free_edges"] = free_edge_count(node)
    record["phase2_total_edges"] = int(node.spacetime().getEdgeList().size())
    node.set_two_body_target(chi, args.choi)
    t0 = time.time()
    trace2 = [node.two_body_residual()]
    carried = [node.whole_complex_fiber_residual()]
    for _ in range(args.phase2_rounds):
        if trace2[-1] < args.tolerance:
            break
        node.run_stage1(max_steps=args.stage1_steps, n_candidate_moves=args.candidates)
        node.run_stage2(beta=1.0, max_iters=args.stage2_iters, tolerance=1e-15)
        trace2.append(node.two_body_residual())
        carried.append(node.whole_complex_fiber_residual())
    read = node.read_two_body()
    T = np.asarray(read.transfer)
    record["phase2"] = {"leak_trace": [float(x) for x in trace2], "whole_fiber_trace": [float(x) for x in carried],
                        "seconds": round(time.time() - t0, 1),
                        "leak": float(read.residual), "leak_recomputed": projective_leak(T, chi),
                        "schmidt": [float(s) for s in read.singular_values],
                        "block_residuals_after": [float(x) for x in read.input_fiber_residuals],
                        "whole_fiber_residual_after": float(node.whole_complex_fiber_residual()),
                        "reversal_residual": float(read.reversal_residual),
                        **connectivity(node.spacetime(), a, b), **metric_conditioning(node.spacetime()),
                        "transfer": _payload(T)}
    return record


def run(args):
    rng = np.random.default_rng(args.seed)
    psi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi = _unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    chi = flip_flop(psi, phi)
    record = {"method": {"protocol": "phase 1 carrier fit; phase 2 frames pinned, bulk against chi", "carrier": args.carrier,
                         "reading": "choi_decomposed" if args.choi else "operator", "seed": args.seed,
                         "precones": args.precones, "pairs_per_bulk": args.pairs, "phase1_rounds": args.phase1_rounds,
                         "phase2_rounds": args.phase2_rounds, "stage2_iters": args.stage2_iters},
              "psi": _payload(psi), "phi": _payload(phi),
              "chi_singular_values": [float(s) for s in np.linalg.svd(chi, compute_uv=False)],
              "configurations": []}
    for precone in args.precones:
        probe = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
                   seed=args.seed, precone=precone, einstein_hilbert=False)
        ranked = ranked_pairs(probe.spacetime())
        for score, a, b, sv in ranked[:args.pairs]:
            cfg = run_configuration(psi, phi, chi, args.seed, precone, (a, b), args)
            cfg["start"]["sigma2_over_sigma1"] = score
            record["configurations"].append(cfg)
            phase1_last = cfg['phase1'].get('block_residual_trace', [[cfg['phase1'].get('whole_fiber_trace', [float('nan')])[-1]]])[-1]
            print(f"precone {precone} pair {a}{b} s2/s1 {score:.3f}: free edges {cfg['phase2_free_edges']}/{cfg['phase2_total_edges']} phase1 {['%.1e' % x for x in phase1_last]} "
                  f"-> phase2 leak {cfg['phase2']['leak']:.3f} schmidt {['%.3f' % s for s in cfg['phase2']['schmidt']]} "
                  f"adjacent-both {cfg['phase2']['cells_adjacent_to_both']} dist {cfg['phase2']['frame_distance']} "
                  f"cond M0 {cfg['phase2']['M0_condition']:.1e} M1 {cfg['phase2']['M1_condition']:.1e} ({cfg['phase2']['seconds']} s)", flush=True)
    leaks = [c["phase2"]["leak"] for c in record["configurations"]]
    record["checks"] = {"some_configuration_carries_the_velocity": bool(min(leaks) < args.tolerance),
                        "best_leak": float(min(leaks))}
    return record


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precones", type=int, nargs="+", default=[8, 14, 20, 30])
    p.add_argument("--pairs", type=int, default=2, help="attachment pairs tried per bulk, ranked by sigma2/sigma1")
    p.add_argument("--shells", type=int, default=1, help="shells of cells around each attached tetrahedron in its block region (blocks carrier)")
    p.add_argument("--carrier", choices=("whole", "blocks"), default="whole",
                   help="whole: one band of the whole complex carries psi on A and phi on B; blocks: each block's own sub-complex carries its fiber, then the blocks are pinned")
    p.add_argument("--phase1-rounds", type=int, default=6)
    p.add_argument("--phase2-rounds", type=int, default=8)
    p.add_argument("--stage1-steps", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--stage2-iters", type=int, default=200)
    p.add_argument("--block-tolerance", type=float, default=1e-8)
    p.add_argument("--tolerance", type=float, default=1e-8)
    p.add_argument("--operator", dest="choi", action="store_false")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    previous = cob.HodgeLaplacian.defaultMetricSource()
    cob.HodgeLaplacian.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        record = run(args)
    finally:
        cob.HodgeLaplacian.setDefaultMetricSource(previous)
    text = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps(record["checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
