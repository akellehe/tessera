# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The expansion of the self-energy in the screened interaction beyond
Sigma = i G W, as time-ordered skeleton diagrams evaluated in closed form.

The screened interaction of the random-phase approximation is the bare kernel
plus the exchange of the particle-hole modes (`screening.RandomPhase`),

    W_{pq,rs}(w) = (pq|rs) + sum_s g^s_pq g^s_rs 2 W_s / (w^2 - W_s^2) ,

so every diagram is a diagram of electrons coupled linearly to bosons of
energies W_s plus an instantaneous interaction, with the Hartree-Fock propagator
on every fermion line. The skeleton diagrams (no self-energy insertion on a
fermion line, no polarization insertion on an interaction line, both being
carried by the propagator and by W) are one at first order, one at second (the
two interaction lines crossed), and six at third: the four irreducible ways of
crossing three lines, and the two orientations of a fermion triangle that
absorbs three lines.

A diagram is evaluated as the sum over the orderings of its vertex times. In an
ordering a fermion line that runs forward in time is a particle (an empty
state, energy xi above the chemical potential), one that runs backward a hole (a
filled state, energy -xi, and a factor -1); a boson is present between its two
vertices; an instantaneous line puts its two vertices at one time. Between
consecutive times the intermediate state has the energy E of everything present,
and contributes 1 / (w - E) when the external line spans it forward, 1 / (-w - E)
when it spans it backward, and 1 / (-E) otherwise. A closed fermion loop carries
-1 and the factor 2 of its spin sum. Every denominator is w minus an excitation
energy, so nothing cancels between large terms, and the expression is analytic
in w: a complex w gives the derivative (`SkeletonSelfEnergy.evaluate`) and the
value on the imaginary axis, where the test suite holds every diagram to the
plain frequency integrals of its Feynman rules.
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

_POOL = None


def _pool():
    """One pool of worker processes for the life of the interpreter: the terms of
    a diagram are thousands of small independent contractions, whose cost is
    the interpreter's and does not thread."""
    global _POOL
    import __main__
    if not hasattr(__main__, "__file__"):
        return None                                              # workers re-import the main module; without one, run serially
    if _POOL is None:
        import multiprocessing
        # The workers start fresh and single-threaded: a threaded linear-algebra library in every one of them
        # would oversubscribe the machine.
        saved = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update({name: "1" for name in saved})
        try:
            _POOL = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1),
                                        mp_context=multiprocessing.get_context("spawn"))
            list(_POOL.map(abs, range(_POOL._max_workers)))      # start them while the environment is set
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
    return _POOL


def _task(arguments):
    engine, diagram, kinds, n, frequency, part, parts = arguments
    return engine._diagram(diagram, kinds, n, frequency, part, parts)


@dataclass(frozen=True)
class Diagram:
    """`edges` are the internal fermion lines (from vertex, to vertex); the
    external line enters at `entry` and leaves at `exit`; `chords` pair the
    vertices by interaction lines; `loops` counts closed fermion loops."""
    vertices: int
    edges: tuple
    entry: int
    exit: int
    chords: tuple
    loops: int = 0


def _line(chords):
    """One open fermion line through the vertices 0 .. V-1 with the given chords."""
    count = 2 * len(chords)
    return Diagram(count, tuple((j, j + 1) for j in range(count - 1)), 0, count - 1, tuple(chords))


def skeleton_diagrams(order):
    """The skeleton self-energy diagrams of `order` interaction lines."""
    if order == 1:
        return [_line([(0, 1)])]
    if order == 2:
        return [_line([(0, 2), (1, 3)])]
    if order == 3:
        crossed = [_line(chords) for chords in ([(0, 2), (1, 4), (3, 5)], [(0, 3), (1, 4), (2, 5)],
                                                [(0, 3), (1, 5), (2, 4)], [(0, 4), (1, 3), (2, 5)])]
        chords = ((0, 3), (1, 4), (2, 5))
        triangles = [Diagram(6, ((0, 1), (1, 2)) + loop, 0, 2, chords, loops=1)
                     for loop in (((3, 4), (4, 5), (5, 3)), ((3, 5), (5, 4), (4, 3)))]
        return crossed + triangles
    raise NotImplementedError(f"skeleton diagrams of order {order}")


class SkeletonSelfEnergy:
    """Sigma_nn(w) from the skeleton diagrams, for fermion levels `xi` (measured
    from a chemical potential inside the gap; the first `occupied` are filled),
    boson energies `bosons`, couplings `g[s, p, q]` and the instantaneous
    interaction `U[p, q, r, s] = (pq|rs)`. With `U = None` there are no
    instantaneous lines. At first order the instantaneous line is the exchange
    of the mean field and is left out, as it is in the Hartree-Fock levels."""

    def __init__(self, xi, occupied, bosons, g, U=None):
        self.xi, self.occupied = np.asarray(xi, dtype=float), int(occupied)
        self.bosons, self.g = np.asarray(bosons, dtype=float), np.asarray(g)
        self.U = None if U is None else np.asarray(U)
        self.filled, self.empty = np.arange(self.occupied), np.arange(self.occupied, len(self.xi))
        self._paths = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_paths"] = {}                                     # found again in the worker
        return state

    def evaluate(self, n, frequency, order, parallel=None):
        """The sum of the diagrams of `order` for the mode n at the (complex)
        frequency. The third order is spread over worker processes by default."""
        tasks = []
        for diagram in skeleton_diagrams(order):
            kinds = itertools.product(*[("dynamic", "static") if self.U is not None and order > 1 else ("dynamic",)
                                        for _ in diagram.chords])
            for kind in kinds:
                parts = 6 if order >= 3 and all(k == "dynamic" for k in kind) else 1          # even out the tasks
                tasks += [(self, diagram, kind, n, frequency, part, parts) for part in range(parts)]
        pool = _pool() if ((order >= 3) if parallel is None else parallel) else None
        return sum(pool.map(_task, tasks)) if pool is not None else sum(_task(task) for task in tasks)

    def value_and_derivative(self, n, frequency, order, step=1e-6):
        """Sigma and d Sigma / dw at a real frequency from one complex evaluation:
        the expression is analytic, so the derivative is Im Sigma(w + i h) / h."""
        shifted = self.evaluate(n, complex(frequency, step), order)
        return float(np.real(shifted)), float(np.imag(shifted)) / step

    # -- one diagram, one assignment of instantaneous and retarded lines

    def _diagram(self, diagram, kinds, n, frequency, part=0, parts=1):
        group = list(range(diagram.vertices))                    # instantaneous lines merge their two times
        for (u, v), kind in zip(diagram.chords, kinds):
            if kind == "static":
                group[v] = group[u]
        groups = sorted(set(group))
        dynamic = [c for c, kind in enumerate(kinds) if kind == "dynamic"]
        letters = "abcdefghijklmnopqrstuvwxyz"
        edge_letter = {e: letters[i] for i, e in enumerate(range(len(diagram.edges)))}
        chord_letter = {c: letters[len(diagram.edges) + i] for i, c in enumerate(dynamic)}
        incoming, outgoing = {diagram.entry: None}, {diagram.exit: None}
        for e, (u, v) in enumerate(diagram.edges):
            outgoing[u], incoming[v] = e, e
        total = 0.0
        for position, order in enumerate(itertools.permutations(range(len(groups)))):
            if position % parts != part:
                continue
            time = {vertex: order[groups.index(group[vertex])] for vertex in range(diagram.vertices)}
            if any(time[u] == time[v] for u, v in diagram.edges):
                continue                                         # an equal-time line is a mean-field insertion
            hole = [time[u] > time[v] for u, v in diagram.edges]
            subsets = [self.filled if h else self.empty for h in hole]
            if any(len(subset) == 0 for subset in subsets):
                continue
            # One factor per interval between consecutive times, on the lines present in it only.
            operands, subscripts = [], []
            for slice_ in range(len(groups) - 1):
                present = [("edge", e) for e, (u, v) in enumerate(diagram.edges)
                           if min(time[u], time[v]) <= slice_ < max(time[u], time[v])]
                present += [("chord", c) for c in dynamic
                            if min(time[diagram.chords[c][0]], time[diagram.chords[c][1]]) <= slice_
                            < max(time[diagram.chords[c][0]], time[diagram.chords[c][1]])]
                energy = np.zeros([1] * len(present))
                for axis, (kind, index) in enumerate(present):
                    values = ((-1.0 if hole[index] else 1.0) * self.xi[subsets[index]] if kind == "edge" else self.bosons)
                    view = [1] * len(present)
                    view[axis] = len(values)
                    energy = energy + values.reshape(view)
                if time[diagram.entry] <= slice_ < time[diagram.exit]:
                    factor = 1.0 / (frequency - energy)
                elif time[diagram.exit] <= slice_ < time[diagram.entry]:
                    factor = 1.0 / (-frequency - energy)
                else:
                    factor = 1.0 / (-energy)
                operands.append(factor)
                subscripts.append("".join(edge_letter[i] if kind == "edge" else chord_letter[i] for kind, i in present))
            # The couplings: g[s, out, in] at both ends of a retarded line, U[out, in, out', in'] on an instantaneous one.

            def fermion(vertex, which):
                e = (outgoing if which == "out" else incoming)[vertex]
                return (None, None) if e is None else (edge_letter[e], subsets[e])

            for c, (u, v) in enumerate(diagram.chords):
                if kinds[c] == "dynamic":
                    for vertex in (u, v):
                        (lo, so), (li, si) = fermion(vertex, "out"), fermion(vertex, "in")
                        tensor = self._pick(self.g, [None, so, si], n)
                        operands.append(tensor)
                        subscripts.append(chord_letter[c] + (lo or "") + (li or ""))
                else:
                    picks = [fermion(u, "out"), fermion(u, "in"), fermion(v, "out"), fermion(v, "in")]
                    operands.append(self._pick(self.U, [s for _, s in picks], n, leading=False))
                    subscripts.append("".join(letter or "" for letter, _ in picks))
            expression = ",".join(subscripts) + "->"
            key = (expression, tuple(operand.shape for operand in operands))
            if key not in self._paths:                           # the contraction order is found once per term
                self._paths[key] = np.einsum_path(expression, *operands, optimize="greedy")[0]
            value = np.einsum(expression, *operands, optimize=self._paths[key])
            sign = (-1.0) ** (sum(hole) + diagram.loops) * 2.0 ** diagram.loops
            total = total + sign * value
        return total

    @staticmethod
    def _pick(tensor, subsets, n, leading=True):
        """`tensor` with every fermion axis restricted to its subset, or fixed at
        the external mode n where the subset is None. `leading` marks a first
        axis that is kept whole (the boson axis of g)."""
        picked = tensor
        axis = 1 if leading else 0
        for subset in (subsets[1:] if leading else subsets):
            if subset is None:
                picked = np.take(picked, n, axis=axis)
            else:
                picked = np.take(picked, subset, axis=axis)
                axis += 1
        return picked
