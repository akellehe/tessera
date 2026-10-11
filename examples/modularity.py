"""Walk the Newman-Girvan modularity Q of an abstract undirected graph
in either direction by swap moves directed against a fixed modulo-M
partition.

At initialization, vertex ``v`` is assigned to community ``v % M``;
that labeling stays constant for the entire sweep.  Each iteration
proposes a swap constrained to two randomly-chosen communities
``(c1, c2)``:

    'up'   -> remove c1<->c2 edge + add within-c1 or within-c2 non-edge
              (raises within-module edge count, lowers between).
    'down' -> remove within-c1 or within-c2 edge + add c1<->c2 non-edge.

Within-edge count changes by exactly +1 ('up') or -1 ('down') per
swap, so the partition's Newman-Girvan Q tracks roughly monotonically
with iteration count -- sufficient for the threshold-driven D_S
sampling done here.

Whenever Q crosses the next ``target_dq`` increment we measure the
spectral dimension D_S via discrete diffusion, optionally with
bipartite-detection and negative-D_S retry safety nets driven by
graph thermalization.

Modularity follows Newman & Girvan (2004) with respect to the fixed
modulo partition: ``Q = sum_c [L_c/m - (D_c/2m)^2]``, where ``L_c`` is
the number of edges within community ``c``, ``D_c`` is the sum of
degrees in ``c``, and ``m = |E|``.

Run with ``--direction both`` to sweep up *and* down from the initial
graph and plot D_S vs Q over the full range.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse

import tessera

import tessera


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapMove:
    """A swap operation that removes one edge and adds another.

    Preserves ``|E|`` (the total edge count).

    Attributes
    ----------
    remove : tuple[int, int]
        The edge to remove, with smaller index first.
    add : tuple[int, int]
        The non-edge to add, with smaller index first.
    """
    remove: tuple[int, int]
    add: tuple[int, int]


@dataclass(frozen=True)
class Measurement:
    """A single recorded point on the (Q, D_S) trajectory.

    Attributes
    ----------
    Q : float
        Newman-Girvan modularity at the time of measurement, computed
        with respect to the fixed modulo-M partition.
    ds_small : float
        Spectral dimension averaged over short diffusion times
        (small sigma).  ``NaN`` if the graph has no valid sigma points.
    ds_large : float
        Spectral dimension averaged over long diffusion times
        (large sigma).  ``NaN`` in the same conditions.
    n_edges : int
        Number of undirected edges at the measurement.
    iter : int
        Sweep iteration index at which this point was recorded.
    direction : str
        Either ``'up'`` or ``'down'``; the sweep direction this point
        belongs to.
    """
    Q: float
    ds_small: float
    ds_large: float
    n_edges: int
    iter: int
    direction: str


@dataclass
class OptimizerConfig:
    """Hyperparameters for :class:`Optimizer`.

    Attributes
    ----------
    target_dq : float
        Q increment between recorded measurements.  Smaller -> denser
        sampling, longer runtime.  Default ``0.05``.
    max_iterations : int
        Hard cap on iterations per sweep direction.  Default ``400``.
    n_diffusion_walks : int
        Number of diffusion start nodes per spectral-dimension
        measurement.  Capped at ``Graph.n_nodes`` if larger.  Default
        ``80``.
    max_sigma : int
        Maximum diffusion time per measurement.  Cost is
        ``O(max_sigma * n_diffusion_walks * nnz)``.  Default ``200``.
    thermalization : int
        Number of degree-preserving random edge swaps per
        thermalization round (used both at startup and by the safety
        wrappers around measurement).  ``0`` disables the safety
        mechanisms.  Default ``0``.
    bipartite_check_frequency : float
        Probability in ``[0, 1]`` of running an
        :meth:`Graph.is_bipartite` check before each measurement; if
        bipartite, thermalize once.  ``0`` disables.  Default ``0.0``.
    negative_retry_max : int
        Maximum thermalize-and-retry attempts when a D_S measurement
        comes back negative (a centered-difference artifact).  Default
        ``10``.
    epsilon_q_max : float
        Absolute tolerance for the up-sweep's early-exit check.  When
        ``current_Q >= (1 - 1/M) - epsilon_q_max`` we declare the
        sweep effectively at the theoretical max for the partition
        and stop, avoiding long Type-2-only tails after Q has
        plateaued.  Set to ``0`` to disable.  Default ``0.01``.
    """
    target_dq: float = 0.05
    max_iterations: int = 400
    n_diffusion_walks: int = 80
    max_sigma: int = 200
    thermalization: int = 0
    bipartite_check_frequency: float = 0.0
    negative_retry_max: int = 10
    epsilon_q_max: float = 0.01


# ---------------------------------------------------------------------------
# ProgressBar: single-line, in-place, graphical progress display
# ---------------------------------------------------------------------------


class ProgressBar:
    """Single-line, in-place graphical progress display.

    Renders a graphical bar with an iteration counter plus arbitrary
    key=value postfix data; updates in place via carriage return and
    an ANSI clear-to-end-of-line escape.  Disabled (no-op) when the
    output stream is not a TTY (e.g. inside a pipe / log file), so
    output stays clean in those contexts.

    Use as a context manager so the trailing newline is emitted on
    exit::

        with ProgressBar(total=100, desc='up') as bar:
            for i in range(100):
                bar.update(current=i, Q=f'{q:.4f}')

    Attributes
    ----------
    BAR_FILLED, BAR_EMPTY : str
        Glyphs used for the filled / empty portions of the bar.
    """

    BAR_FILLED = '█'
    BAR_EMPTY = '░'
    _MIN_INTERVAL = 0.05  # seconds; cap renders at ~20 fps

    def __init__(self, total, desc='', bar_width=30, stream=None):
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.bar_width = max(4, int(bar_width))
        self.stream = stream or sys.stderr
        self.current = 0
        self.postfix: dict[str, str] = {}
        self._enabled = bool(getattr(self.stream, 'isatty', lambda: False)())
        self._last_render = 0.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self, current=None, force=False, **postfix):
        """Refresh the bar with new state.

        Parameters
        ----------
        current : int, optional
            Iteration index.  If omitted, the previous value is kept.
        force : bool, optional
            If True, render immediately, bypassing the throttle.
        **postfix
            key=value pairs displayed after the bar.  Insertion order
            is preserved on first add; existing keys are updated in
            place.
        """
        if current is not None:
            self.current = int(current)
        if postfix:
            self.postfix.update({k: str(v) for k, v in postfix.items()})
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_render < self._MIN_INTERVAL):
            return
        self._render()
        self._last_render = now

    def write_above(self, message):
        """Print ``message`` on its own line above the bar without
        permanently disrupting the bar.

        Falls back to a plain ``print`` on non-TTY streams.
        """
        if not self._enabled:
            print(message, file=self.stream)
            return
        self.stream.write('\r\x1b[2K' + str(message) + '\n')
        self.stream.flush()
        self._render()

    def close(self):
        """Render the final state and move the cursor to a fresh line.

        Idempotent.  Called automatically when the bar is used as a
        context manager.
        """
        if not self._enabled:
            return
        self._render()
        self.stream.write('\n')
        self.stream.flush()
        self._enabled = False

    def _render(self):
        bar = self._make_bar()
        post = '  '.join(f"{k}={v}" for k, v in self.postfix.items())
        remaining = max(0, self.total - self.current)
        line = (f"{self.desc} [{bar}] {self.current}/{self.total} "
                f"({remaining} left)  {post}")
        self.stream.write('\r\x1b[2K' + line)
        self.stream.flush()

    def _make_bar(self):
        if self.total == 0:
            return self.BAR_EMPTY * self.bar_width
        filled = min(self.bar_width,
                     int(self.bar_width * self.current / self.total))
        return (self.BAR_FILLED * filled
                + self.BAR_EMPTY * (self.bar_width - filled))


# ---------------------------------------------------------------------------
# Graph: immutable wrapper around a scipy sparse adjacency, with all the
# operations we need (modularity, spectral dimension, swap, thermalize, ...).
# ---------------------------------------------------------------------------


class Graph:
    """Immutable undirected graph backed by a scipy sparse adjacency.

    Construction is via ``Graph(adjacency)`` or one of the factory
    classmethods (:meth:`mst`, :meth:`barabasi_albert`).  All
    transformations -- :meth:`with_swap`, :meth:`thermalize` -- return
    a new ``Graph`` rather than mutating in place, so any ``Graph``
    instance is safe to share between callers.

    Parameters
    ----------
    adjacency : scipy.sparse.spmatrix
        Symmetric, binary (0/1) adjacency.  Self-loops are not
        expected and are not enforced against.
    """

    # --- construction --------------------------------------------------

    def __init__(self, adjacency):
        self._A = adjacency.tocsc()

    @classmethod
    def mst(cls, n_nodes, rng):
        """Build the minimum spanning tree of a complete graph on
        ``n_nodes`` with iid uniform edge weights.

        Generates a symmetric weight matrix W ~ Uniform[0, 1) for all
        off-diagonal entries, computes the MST, and binarizes.  Result
        has exactly ``n_nodes - 1`` edges, is connected, and is always
        bipartite (every tree is).

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        rng : numpy.random.Generator
            Random source for the edge weights.

        Returns
        -------
        Graph
        """
        if n_nodes <= 1:
            return cls(sparse.csc_matrix((n_nodes, n_nodes)))
        W = rng.random((n_nodes, n_nodes))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        mst = sparse.csgraph.minimum_spanning_tree(sparse.csr_matrix(W))
        A = mst.maximum(mst.T)
        A.data[:] = 1.0
        return cls(A)

    @classmethod
    def barabasi_albert(cls, n_nodes, m, rng):
        """Build a Barabasi-Albert preferential-attachment graph.

        Wraps ``networkx.barabasi_albert_graph(n_nodes, m)``.  Each new
        node attaches to ``m`` existing nodes with probability
        proportional to current degree; asymptotic degree distribution
        ``P(k) ~ k^{-3}`` (scale-free).  ``m == 1`` is a tree (always
        bipartite); ``m >= 2`` generally has odd cycles.

        Parameters
        ----------
        n_nodes : int
            Total node count.  If ``n_nodes <= m``, falls back to a
            complete graph on those nodes.
        m : int
            Edges added per new node.  Must be ``>= 1``.
        rng : numpy.random.Generator
            Used to draw a 32-bit seed for networkx (so the graph is
            reproducible from this script's seed).

        Returns
        -------
        Graph
        """
        if n_nodes <= 0:
            return cls(sparse.csc_matrix((0, 0)))
        if n_nodes <= m:
            G = nx.complete_graph(n_nodes)
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        A = nx.to_scipy_sparse_array(G, format='csc', dtype=float)
        A.data[:] = 1.0
        return cls(A)

    # --- repr / accessors ---------------------------------------------

    def __repr__(self):
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges})"

    @property
    def adjacency(self):
        """The symmetric binary adjacency as a CSC matrix."""
        return self._A

    @property
    def n_nodes(self):
        """Number of nodes."""
        return self._A.shape[0]

    @property
    def n_edges(self):
        """Number of undirected edges."""
        return int(self._A.nnz / 2)

    @property
    def degree(self):
        """Degree sequence as a 1-D ``numpy.ndarray`` of int.

        ``self.degree[i]`` is the number of edges incident to node ``i``.
        """
        return np.asarray(self._A.sum(axis=0)).ravel().astype(int)

    def has_edge(self, i, j):
        """Return True iff ``(i, j)`` is an edge."""
        return self._A[i, j] != 0

    def edges(self):
        """Return the edge list as ``(i, j)`` tuples with ``i < j``.

        Returns
        -------
        list of tuple[int, int]
        """
        coo = sparse.triu(self._A, k=1).tocoo()
        return [(int(i), int(j))
                for i, j in zip(coo.row.tolist(), coo.col.tolist())]

    def _sparse_graph(self):
        """Build the bound C++ ``tessera.SparseGraph`` for this topology.

        The abstract graph is binary and undirected, so its edge list maps
        directly onto a ``SparseGraph`` (which collapses duplicates and adds
        both directions).  Structural queries (bipartiteness, modularity)
        delegate here rather than re-implementing the C++ formulas.
        """
        rows, cols = [], []
        for i, j in self.edges():
            rows.append(i)
            cols.append(j)
        return tessera.SparseGraph.from_coo(rows, cols, self.n_nodes)

    # --- structural queries -------------------------------------------

    def is_bipartite(self):
        """Return True iff the graph is bipartite.

        A graph is bipartite iff its node set admits a 2-coloring with
        no monochromatic edge -- equivalently, contains no odd cycle.
        Empty / zero-edge graphs are trivially bipartite.

        Delegates to ``SparseGraph.is_bipartite`` (BFS 2-coloring in C++).

        Returns
        -------
        bool
        """
        if self.n_nodes == 0 or self._A.nnz == 0:
            return True
        return self._sparse_graph().is_bipartite()

    # --- spectral dimension -------------------------------------------

    def spectral_dimension(self, n_diffusion_walks, max_sigma, rng,
                           tail_fraction=0.2, n_times=40, t_min=0.5):
        """Estimate the spectral dimension at small and large diffusion
        times via the continuous-time heat kernel.

        ``D_S(t) = -2 * d log K(t) / d log t`` where
        ``K(t) = (e^{-t L_rw})[start, start]`` averaged over random
        starts.  We sample ``t`` log-uniformly in ``[t_min, max_sigma]``
        (so log-log slope extraction has uniform precision across the
        range), take centered finite differences on
        ``(log t, log K)``, and return the means over the first
        ``tail_fraction`` (small ``t`` -- local geometry) and last
        ``tail_fraction`` (large ``t`` -- global geometry) of the
        resulting D_S array.

        Continuous time avoids the bipartite parity zeros that the
        discrete-step variant exhibits on trees / chains, so the
        centered-difference estimator is robust on every graph in the
        catalog (no thermalize-and-retry safety net needed for the D_S
        extraction itself).

        Parameters
        ----------
        n_diffusion_walks : int
            Number of starts (capped at ``self.n_nodes``).
        max_sigma : int or float
            Upper bound of the log-spaced time grid.  Argument name
            kept for backward compatibility with the discrete-walk
            interface; in continuous-time semantics, ``σ -> t``.
        rng : numpy.random.Generator
            Random source for choosing starts.
        tail_fraction : float, optional
            Fraction of grid points at each end to average into
            D_S(small) and D_S(large).  Must lie in ``(0, 0.5]``.
            Default ``0.2``.
        n_times : int, optional
            Number of log-spaced ``t`` samples.  Default ``40``.
        t_min : float, optional
            Lower bound of the log-spaced time grid.  Default ``0.5``.
            Below ``t ~ 1`` the walker is in a transient (non-power-
            law) regime; this default puts the small-``t`` tail just
            into the diffusion regime for typical graphs.

        Returns
        -------
        D_S_small : float
            Mean D_S over the first ``tail_fraction`` of grid points.
            ``NaN`` if the graph is empty or has too few valid points.
        D_S_large : float
            Mean D_S over the last ``tail_fraction`` of grid points.
            ``NaN`` in the same conditions.
        """
        if self.n_nodes == 0:
            return float('nan'), float('nan')
        # Delegate to the C++ SparseGraph heat-kernel estimator: it runs
        # the same Krylov-Lanczos diffusion + centered-difference D_S
        # extraction this method used to do in SciPy.  L_sym (SparseGraph)
        # and L_rw (used here previously) share the heat-kernel diagonal,
        # so the (small, large) pair is numerically identical.  ``rng``
        # seeds the start-vertex subsample for reproducibility.
        coo = self._A.tocoo()
        sg = tessera.SparseGraph.from_coo(
            coo.row.astype(np.uint32).tolist(),
            coo.col.astype(np.uint32).tolist(),
            int(self.n_nodes))
        seed = int(rng.integers(0, 2**63 - 1))
        return sg.spectral_dimension(
            n_walks=int(n_diffusion_walks), max_sigma=float(max_sigma),
            seed=seed, tail_fraction=float(tail_fraction),
            n_times=int(n_times), t_min=float(t_min))

    # --- modularity ---------------------------------------------------

    def modularity(self, labels):
        """Newman-Girvan modularity Q for a given partition.

        ``Q = sum_c [L_c/m - (D_c/2m)^2]`` where ``L_c`` is the number
        of edges within community ``c``, ``D_c`` is the sum of degrees
        of nodes in ``c``, and ``m = |E|``.

        Parameters
        ----------
        labels : sequence of int, length ``self.n_nodes``
            Community label per node.  Distinct values are treated as
            distinct communities; labels need not be dense or start
            at 0.

        Returns
        -------
        float
            Modularity.  Range ``[-0.5, 1)``: 0 = no community
            structure, ~1 = strong community structure, negative =
            anti-community structure.

        Raises
        ------
        ValueError
            If ``len(labels) != self.n_nodes``.
        """
        n = self.n_nodes
        if len(labels) != n:
            raise ValueError(
                f"labels has length {len(labels)} but graph has {n} nodes")
        if n == 0:
            return 0.0
        # Newman-Girvan Q in C++ (SparseGraph.modularity); edgeless graphs
        # return 0.0 there, matching the previous behaviour.
        return self._sparse_graph().modularity([int(c) for c in labels])

    # --- transformations ----------------------------------------------

    def with_swap(self, move):
        """Return a new graph with one edge swapped.

        Parameters
        ----------
        move : SwapMove
            The edge to remove and the non-edge to add.

        Returns
        -------
        Graph
            New graph; ``self`` is not modified.
        """
        A = self._A.tolil()
        ri, rj = move.remove
        ai, aj = move.add
        A[ri, rj] = 0.0
        A[rj, ri] = 0.0
        A[ai, aj] = 1.0
        A[aj, ai] = 1.0
        return Graph(A.tocsc())

    def thermalize(self, n_swaps, rng, max_attempts_per_swap=200):
        """Return a new graph after ``n_swaps`` random edge swaps.

        Each swap (i) picks a random edge to remove, retrying until
        both endpoints have degree ``> 1`` (so the removal will not
        isolate either), then (ii) picks a uniformly random non-edge
        to add.  ``|E|`` is preserved by construction; minimum degree
        stays ``>= 1``.

        Parameters
        ----------
        n_swaps : int
            Number of swaps; ``<= 0`` returns an unchanged copy.
        rng : numpy.random.Generator
            Random source for picking edges and non-edges.
        max_attempts_per_swap : int, optional
            Cap on retries per pick (both for finding a removable edge
            and for finding a valid non-edge).  If exceeded for any
            swap, thermalization stops early.  Default ``200``.

        Returns
        -------
        Graph
            New graph; ``self`` is not modified.

        Notes
        -----
        This is a "double-edge swap" preserving the edge count but
        NOT the full degree sequence.  For a degree-sequence-preserving
        rewiring, a swap would need to be ``(a, b), (c, d) -> (a, c),
        (b, d)``.
        """
        if n_swaps <= 0 or self.n_nodes == 0:
            return Graph(self._A.copy())
        A = self._A.tolil(copy=True)
        n = A.shape[0]
        deg = self.degree.copy()
        edges = self.edges()
        edge_set = set(edges)

        for _ in range(n_swaps):
            ri = rj = remove_idx = None
            for _ in range(max_attempts_per_swap):
                idx = int(rng.integers(len(edges)))
                i, j = edges[idx]
                if deg[i] > 1 and deg[j] > 1:
                    ri, rj, remove_idx = i, j, idx
                    break
            if ri is None:
                break

            ai = aj = None
            for _ in range(max_attempts_per_swap):
                ij = rng.integers(0, n, size=2)
                i, j = int(ij[0]), int(ij[1])
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if (i, j) in edge_set or (i, j) == (ri, rj):
                    continue
                ai, aj = i, j
                break
            if ai is None:
                break

            A[ri, rj] = 0.0
            A[rj, ri] = 0.0
            A[ai, aj] = 1.0
            A[aj, ai] = 1.0
            deg[ri] -= 1
            deg[rj] -= 1
            deg[ai] += 1
            deg[aj] += 1
            edge_set.discard((ri, rj))
            edge_set.add((ai, aj))
            edges[remove_idx] = (ai, aj)

        return Graph(A.tocsc())


# ---------------------------------------------------------------------------
# Optimizer: walks Q in a given direction by swap moves directed against
# a fixed modulo-M partition.
# ---------------------------------------------------------------------------


class Optimizer:
    """Module-targeted swap-move modularity sweep.

    Walks Newman-Girvan modularity Q in a chosen direction by swap
    moves constrained against a *fixed* partition (``labels``, passed
    at construction).  Two move types are available on the up-sweep:

      Type 1 (Q-changing): pick (c1, c2) at random; ``'up'`` removes
      a c1<->c2 edge and adds a within-c1 or within-c2 non-edge,
      ``'down'`` does the reverse.  Removed edges are constrained to
      have both endpoints at degree > 1 so the move never isolates
      a node.

      Type 2 (Q-preserving, fallback for ``'up'``): inside a randomly-
      chosen module, move a within-module edge from the highest-
      degree node to the lowest-degree node.  Modularity is exactly
      preserved but degree is rebalanced, unblocking Type-1 moves
      that were stuck on degree-1 endpoints.

    Type 2 is used only when Type 1 has no eligible move on the
    current iteration, so every Q-changing move that's available is
    still taken first.  Records D_S at every ``target_dq`` crossing,
    with bipartite-detection and negative-D_S retry safety nets
    driven by graph thermalization.

    Parameters
    ----------
    config : OptimizerConfig
        Hyperparameters.
    rng : numpy.random.Generator
        Random source for moves, walks, and thermalization.
    labels : sequence of int
        Per-node community labels; held constant for the entire run.
        Length must match the graph passed to :meth:`sweep`.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> g = Graph.barabasi_albert(60, m=2, rng=rng)
    >>> M = 4
    >>> labels = [v % M for v in range(g.n_nodes)]
    >>> opt = Optimizer(OptimizerConfig(target_dq=0.02), rng, labels)
    >>> measurements = opt.sweep(g, direction='up')
    """

    def __init__(self, config, rng, labels):
        self.config = config
        self.rng = rng
        self.labels = list(labels)

    def sweep(self, initial, direction):
        """Sweep Q from ``initial`` in the given direction.

        Parameters
        ----------
        initial : Graph
            Starting graph.  Not mutated.
        direction : {'up', 'down'}
            Direction in which Q should walk.

        Returns
        -------
        list of Measurement
            One per recorded measurement.  The first entry is always
            the initial-graph measurement at iter 0.

        Raises
        ------
        ValueError
            If ``direction`` is not ``'up'`` or ``'down'``, or if
            ``len(self.labels) != initial.n_nodes``.
        """
        if direction not in ('up', 'down'):
            raise ValueError(
                f"direction must be 'up' or 'down', got {direction!r}")
        if len(self.labels) != initial.n_nodes:
            raise ValueError(
                f"labels has length {len(self.labels)} but graph has "
                f"{initial.n_nodes} nodes")
        sign = 1.0 if direction == 'up' else -1.0
        cfg = self.config
        desc = '↑ up  ' if direction == 'up' else '↓ down'

        # Theoretical max for the fixed partition, computed once.
        # Q_max = 1 - 1/M for M balanced modules with all edges within;
        # we use it as the up-sweep early-exit reference.
        n_modules = len(set(self.labels))
        q_max_target = 1.0 - 1.0 / n_modules if n_modules > 0 else 0.0

        with ProgressBar(total=cfg.max_iterations, desc=desc) as bar:
            graph, m = self._measure(initial, iter_=0,
                                     direction=direction, bar=bar)
            measurements = [m]
            initial_Q = m.Q
            current_Q = m.Q
            self._update_bar(bar, current=0, m=m, n_meas=1,
                             initial_Q=initial_Q, force=True)
            next_threshold = current_Q + sign * cfg.target_dq

            exit_reason = "reached max iterations"
            last_iter = 0
            type_counts = {'type1': 0, 'type2': 0}
            for it in range(1, cfg.max_iterations + 1):
                last_iter = it
                move, kind = self._propose_swap(graph, direction)
                if move is None:
                    exit_reason = "no eligible swap (local extremum)"
                    break
                type_counts[kind] += 1

                graph = graph.with_swap(move)
                current_Q = graph.modularity(self.labels)
                bar.update(current=it,
                           Q=f"{current_Q:.4f}",
                           dQ=f"{current_Q - initial_Q:+.3f}",
                           E=graph.n_edges,
                           t1=type_counts['type1'],
                           t2=type_counts['type2'])

                crossed = (current_Q >= next_threshold) if direction == 'up' \
                    else (current_Q <= next_threshold)
                if crossed:
                    graph, m = self._measure(graph, iter_=it,
                                             direction=direction, bar=bar)
                    measurements.append(m)
                    current_Q = m.Q
                    self._update_bar(bar, current=it, m=m,
                                     n_meas=len(measurements),
                                     initial_Q=initial_Q, force=True,
                                     t1=type_counts['type1'],
                                     t2=type_counts['type2'])
                    next_threshold = current_Q + sign * cfg.target_dq

                if (direction == 'up'
                        and cfg.epsilon_q_max > 0
                        and current_Q >= q_max_target - cfg.epsilon_q_max):
                    exit_reason = (
                        f"within ε={cfg.epsilon_q_max:.4f} of theoretical "
                        f"max (1-1/M={q_max_target:.4f})"
                    )
                    break

            # Final render so the bar reflects the actual stop state
            # regardless of the throttle.
            bar.update(current=last_iter, force=True,
                       t1=type_counts['type1'], t2=type_counts['type2'])

        logger.info(
            "Sweep finished (%s): %s after %d iterations "
            "(%d type-1, %d type-2); %d measurements, "
            "Q: %.4f -> %.4f (Δ%+.4f)",
            direction, exit_reason, last_iter,
            type_counts['type1'], type_counts['type2'],
            len(measurements),
            measurements[0].Q, measurements[-1].Q,
            measurements[-1].Q - measurements[0].Q,
        )
        return measurements

    # --- internals -----------------------------------------------------

    @staticmethod
    def _update_bar(bar, current=None, m=None, n_meas=None,
                    initial_Q=None, force=False, **extra):
        """Refresh ``bar`` with measurement-derived state.

        Convenience helper that builds the postfix dict from the
        latest ``Measurement`` and any extra key=value pairs.  Keeps
        the postfix order consistent across calls.
        """
        kwargs = {}
        if m is not None:
            kwargs['Q'] = f"{m.Q:.4f}"
            if initial_Q is not None:
                kwargs['dQ'] = f"{m.Q - initial_Q:+.3f}"
            kwargs['E'] = m.n_edges
        if n_meas is not None:
            kwargs['meas'] = n_meas
        if m is not None:
            kwargs['D_S'] = f"({m.ds_small:.3f}, {m.ds_large:.3f})"
        kwargs.update(extra)
        bar.update(current=current, force=force, **kwargs)

    def _propose_swap(self, graph, direction):
        """Top-level swap proposal.

        Tries the Q-changing module-pair swap (Type 1) first; on
        ``'up'`` falls back to the Q-preserving intra-module degree
        rebalancing swap (Type 2) when Type 1 has no eligible move.
        Type 2 doesn't advance Q on its own, but it shifts degree
        from high-degree to low-degree nodes inside a module so
        previously stuck Type 1 moves (blocked by degree-1
        endpoints) can succeed on subsequent iterations.

        Parameters
        ----------
        graph : Graph
        direction : {'up', 'down'}

        Returns
        -------
        move : SwapMove or None
            The proposed swap, or ``None`` if both move types fail.
        kind : str or None
            ``'type1'`` if Type 1 produced ``move``, ``'type2'`` if
            Type 2 did, or ``None`` if ``move`` is ``None``.  Caller
            uses this to track per-type fire counts.
        """
        move = self._propose_type1(graph, direction)
        if move is not None:
            return move, 'type1'
        if direction == 'up':
            move = self._propose_type2(graph)
            if move is not None:
                return move, 'type2'
        return None, None

    def _propose_type1(self, graph, direction,
                       max_pair_retries=50, max_add_retries=30):
        """Module-pair-directed swap (Type 1).

        Picks two distinct communities (c1, c2) uniformly at random
        from ``self.labels``, then proposes a swap constrained to
        that pair.  By construction such a swap pushes the within-
        edge count in the desired direction (``'up'`` removes a
        between-edge and adds a within; ``'down'`` does the reverse).

        The removed edge must have both endpoints at degree ``> 1``
        so the move can never isolate a node; this is the
        constraint the user observed plateauing the up-sweep.

        Parameters
        ----------
        graph : Graph
        direction : {'up', 'down'}
        max_pair_retries : int, optional
            Max draws of (c1, c2) before giving up.  Default ``50``.
        max_add_retries : int, optional
            Max attempts to find a valid non-edge to add for a given
            (c1, c2).  Default ``30``.

        Returns
        -------
        SwapMove or None
            ``None`` if no valid swap was found within the retry
            budget (e.g. a single-community partition, no eligible
            edges, or all candidate removes blocked by the
            degree-1 protection).
        """
        edges = graph.edges()
        if not edges:
            return None

        labels = self.labels
        unique_comms = sorted(set(labels))
        if len(unique_comms) < 2:
            return None

        nodes_by_comm: dict[int, list[int]] = {c: [] for c in unique_comms}
        for i, c in enumerate(labels):
            nodes_by_comm[c].append(i)

        deg = graph.degree
        edges_by_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, j in edges:
            if deg[i] <= 1 or deg[j] <= 1:
                continue  # degree-1 protection: skip un-removable edges
            ci, cj = labels[i], labels[j]
            key = (ci, cj) if ci <= cj else (cj, ci)
            edges_by_pair.setdefault(key, []).append((i, j))

        A = graph.adjacency

        for _ in range(max_pair_retries):
            pair = self.rng.choice(len(unique_comms), size=2, replace=False)
            c1, c2 = unique_comms[int(pair[0])], unique_comms[int(pair[1])]
            pair_key = (c1, c2) if c1 <= c2 else (c2, c1)

            if direction == 'up':
                remove_pool = edges_by_pair.get(pair_key, [])
            else:
                remove_pool = (edges_by_pair.get((c1, c1), [])
                               + edges_by_pair.get((c2, c2), []))
            if not remove_pool:
                continue
            ri, rj = remove_pool[int(self.rng.integers(len(remove_pool)))]

            add = None
            for _ in range(max_add_retries):
                if direction == 'up':
                    target_c = c1 if self.rng.random() < 0.5 else c2
                    pool = nodes_by_comm[target_c]
                    if len(pool) < 2:
                        continue
                    pick = self.rng.choice(len(pool), size=2, replace=False)
                    i, j = pool[int(pick[0])], pool[int(pick[1])]
                else:
                    p1, p2 = nodes_by_comm[c1], nodes_by_comm[c2]
                    if not p1 or not p2:
                        continue
                    i = p1[int(self.rng.integers(len(p1)))]
                    j = p2[int(self.rng.integers(len(p2)))]
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if A[i, j] != 0:
                    continue
                if (i, j) == (ri, rj):
                    continue
                add = (i, j)
                break
            if add is None:
                continue

            return SwapMove(remove=(ri, rj), add=add)

        return None

    def _propose_type2(self, graph,
                       max_module_retries=10, max_edge_retries=20):
        """Intra-module degree-rebalancing swap (Type 2).

        Within a randomly-chosen module ``c``: picks the highest-
        degree node ``u`` and the lowest-degree node ``v``, finds a
        within-module edge ``(u, w)``, and proposes the swap
        ``remove (u, w), add (v, w)``.  Both endpoints stay in
        ``c``, so ``L_c`` (within-edges) and ``D_c`` (degree sum) are
        unchanged -- modularity is exactly preserved.  Net effect on
        degrees: ``u`` loses one, ``v`` gains one, ``w`` is unchanged.

        Used as a fallback when :meth:`_propose_type1` plateaus
        (typically because remaining between-module edges all have a
        degree-1 endpoint).  Rebalancing degree from ``u`` to ``v``
        unblocks future Type-1 moves incident on previously-low-
        degree nodes.

        Parameters
        ----------
        graph : Graph
        max_module_retries : int, optional
            Max draws of a module before giving up.  Default ``10``.
        max_edge_retries : int, optional
            Max attempts to find a valid ``(u, w)`` edge for the
            chosen module.  Default ``20``.

        Returns
        -------
        SwapMove or None
            ``None`` if no valid rebalancing move exists in any of
            the sampled modules within the retry budget.
        """
        labels = self.labels
        deg = graph.degree
        A = graph.adjacency

        nodes_by_comm: dict[int, list[int]] = {}
        for i, c in enumerate(labels):
            nodes_by_comm.setdefault(c, []).append(i)
        unique_comms = list(nodes_by_comm)
        if not unique_comms:
            return None

        for _ in range(max_module_retries):
            c = unique_comms[int(self.rng.integers(len(unique_comms)))]
            nodes_c = nodes_by_comm[c]
            if len(nodes_c) < 2:
                continue

            deg_c = [int(deg[i]) for i in nodes_c]
            max_d = max(deg_c)
            min_d = min(deg_c)
            if max_d <= 1 or max_d == min_d:
                # Either un-removable (would isolate u) or no degree
                # gradient to rebalance.
                continue

            u_candidates = [nodes_c[k] for k, d in enumerate(deg_c)
                            if d == max_d]
            v_candidates = [nodes_c[k] for k, d in enumerate(deg_c)
                            if d == min_d]
            u = u_candidates[int(self.rng.integers(len(u_candidates)))]
            v = v_candidates[int(self.rng.integers(len(v_candidates)))]
            if u == v:
                continue

            u_within = [w for w in nodes_c if w != u and A[u, w] != 0]
            if not u_within:
                continue

            order = self.rng.permutation(len(u_within))
            for k in order[:max_edge_retries]:
                w = u_within[int(k)]
                if w == v:
                    continue
                if A[v, w] != 0:
                    continue
                ri, rj = (u, w) if u < w else (w, u)
                ai, aj = (v, w) if v < w else (w, v)
                return SwapMove(remove=(ri, rj), add=(ai, aj))

        return None

    def _measure(self, graph, iter_, direction, bar=None):
        """Measure D_S with bipartite-detection and negative-D_S retry
        safety wrappers.

        Parameters
        ----------
        graph : Graph
        iter_ : int
            Sweep iteration index, stored on the resulting Measurement.
        direction : str
            ``'up'`` or ``'down'``, stored on the resulting Measurement.
        bar : ProgressBar, optional
            If provided, safety-mechanism log lines are written above
            the bar via :meth:`ProgressBar.write_above` rather than
            via the standard logger.  Keeps the live progress display
            uncluttered while still surfacing retries.

        Returns
        -------
        graph_after : Graph
            Possibly-thermalized graph.  The caller should adopt it.
        measurement : Measurement
            With Q recomputed against ``self.labels`` on
            ``graph_after``.
        """
        cfg = self.config

        def _emit(msg):
            if bar is not None:
                bar.write_above(msg)
            else:
                logger.info(msg)

        if (cfg.thermalization > 0
                and cfg.bipartite_check_frequency > 0
                and self.rng.random() < cfg.bipartite_check_frequency
                and graph.is_bipartite()):
            _emit("    [bipartite -> thermalize]")
            graph = graph.thermalize(cfg.thermalization, self.rng)

        ds_s = ds_l = float('nan')
        for retry in range(cfg.negative_retry_max + 1):
            ds_s, ds_l = graph.spectral_dimension(
                cfg.n_diffusion_walks, cfg.max_sigma, self.rng)
            if np.isnan(ds_s) or np.isnan(ds_l):
                break
            if ds_s >= 0 and ds_l >= 0:
                break
            if cfg.thermalization == 0 or retry == cfg.negative_retry_max:
                break
            _emit(f"    [negative D_S (s={ds_s:.3f}, l={ds_l:.3f}) -> "
                  f"thermalize, retry {retry + 1}]")
            graph = graph.thermalize(cfg.thermalization, self.rng)

        Q = graph.modularity(self.labels)
        return graph, Measurement(Q=Q, ds_small=ds_s, ds_large=ds_l,
                                  n_edges=graph.n_edges, iter=iter_,
                                  direction=direction)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_initial(args, rng):
    """Construct the initial graph from CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; uses ``init``, ``n_nodes``, ``ba_m``.
    rng : numpy.random.Generator

    Returns
    -------
    graph : Graph
    label : str
        Human-readable description of the initial graph.
    """
    if args.init == 'mst':
        return Graph.mst(args.n_nodes, rng), f"MST on {args.n_nodes} nodes"
    if args.init == 'ba':
        return (Graph.barabasi_albert(args.n_nodes, args.ba_m, rng),
                f"BA(m={args.ba_m}) on {args.n_nodes} nodes")
    raise ValueError("You must either init with --init ba or --init mst")


def _make_parser():
    parser = argparse.ArgumentParser(
        description="Module-targeted modularity sweep with "
                    "spectral-dimension measurement at each Q increment.")

    g_init = parser.add_argument_group("initial graph")
    g_init.add_argument('--init', choices=['mst', 'ba'],
                        default='ba',
                        help='Initial topology: Barabasi-Albert or '
                             'Minimum Spanning Tree of a random complete '
                             'graph.')
    g_init.add_argument('--n-nodes', type=int, default=80,
                        help='Number of nodes in the initial graph.')
    g_init.add_argument('--ba-m', type=int, default=2,
                        help='BA preferential-attachment parameter m '
                             '(>= 1; m=1 is a tree.  Choose m > 1 if you '
                             'want actual results).')
    g_init.add_argument('--target-n-modules', type=int, default=4,
                        help='Number of modules to target.  Vertex v is '
                             'assigned to community v %% M at '
                             'initialization, and that labeling drives '
                             'all swap proposals throughout the sweep.')

    g_sweep = parser.add_argument_group("sweep")
    g_sweep.add_argument('--direction', choices=['up', 'down', 'both'],
                         default='both',
                         help='Walk Q upward, downward, or both from '
                              'the initial graph.')
    g_sweep.add_argument('--target-dq', type=float, default=0.05,
                         help='Record measurements every dQ '
                              '(target change in modularity).')
    g_sweep.add_argument('--max-iterations', type=int, default=1000,
                         help='Hard cap on iterations in each direction.')
    g_sweep.add_argument('--epsilon-q-max', type=float, default=0.01,
                         help='Up-sweep early-exit tolerance: stop when '
                              'Q is within this absolute distance of '
                              'the theoretical max 1-1/M.  0 disables.')
    g_sweep.add_argument('--sweep-modules', action='store_true',
                         help='Run separate sweeps for M = 2, 3, ..., '
                              '--target-n-modules and overlay all of '
                              'them on the output plot.')

    g_meas = parser.add_argument_group("D_S measurement")
    g_meas.add_argument('--n-diffusion-walks', type=int, default=80,
                        help='Diffusion start nodes per measurement.')
    g_meas.add_argument('--max-sigma', type=int, default=200,
                        help='Maximum diffusion time per measurement.')

    g_safe = parser.add_argument_group("safety")
    g_safe.add_argument('--thermalization', type=int, default=0,
                        help='Number of degree-preserving random edge '
                             'swaps after initial construction (also '
                             'used by the bipartite/negative-D_S '
                             'safety mechanisms).  0 disables safety.')
    g_safe.add_argument('--bipartite-check-frequency', type=float,
                        default=0.0,
                        help='Probability in [0, 1] of checking for '
                             'bipartiteness before each measurement; '
                             'if bipartite, re-thermalize.')
    g_safe.add_argument('--negative-retry-max', type=int, default=10,
                        help='Max thermalize-and-retry attempts when '
                             'D_S comes back negative.')

    g_misc = parser.add_argument_group("misc")
    g_misc.add_argument('--seed', type=int, default=0,
                        help='RNG seed for reproducibility.')
    g_misc.add_argument('--save', type=str,
                        default='./spectral_vs_modularity.png',
                        help='Path to save the output plot.')
    return parser


def _plot(measurements, label, direction, save_path, n_modules):
    """Render the (Q, D_S) and (Q, |E|) panels and save to disk.

    Parameters
    ----------
    measurements : Iterable[Measurement]
        All recorded measurements across both sweep directions.
    label : str
        Initial-graph description for the plot title.
    direction : str
        ``'up'``, ``'down'``, or ``'both'``; used in the plot title.
    save_path : str
        Output path for the PNG figure.
    n_modules : int
        The fixed module count M used by the sweep, displayed in the
        plot title and x-axis label.
    """
    measurements = list(measurements)
    if not measurements:
        return
    Q = np.array([m.Q for m in measurements])
    Dl = np.array([m.ds_large for m in measurements])
    Ds = np.array([m.ds_small for m in measurements])
    Es = np.array([m.n_edges for m in measurements])
    dirs = np.array([m.direction for m in measurements])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    direction_styles = [
        ('up', 'C0', 'upward sweep'),
        ('down', 'C3', 'downward sweep'),
    ]
    for dval, color, dlabel in direction_styles:
        mask = dirs == dval
        if not mask.any():
            continue
        order = np.argsort(Q[mask])
        Q_d = Q[mask][order]
        Dl_d = Dl[mask][order]
        Ds_d = Ds[mask][order]
        Es_d = Es[mask][order]
        ax1.plot(Q_d, Dl_d, color=color, marker='o', linestyle='-',
                 linewidth=1.5, markersize=6,
                 label=f'{dlabel}: $D_S$ (large $\\sigma$)')
        ax1.plot(Q_d, Ds_d, color=color, marker='s', linestyle='--',
                 linewidth=1.0, markersize=5, alpha=0.6,
                 label=f'{dlabel}: $D_S$ (small $\\sigma$)')
        ax2.plot(Q_d, Es_d, color=color, marker='.', linestyle='-',
                 linewidth=1.0, markersize=6, label=dlabel)
    ax1.set_xlabel(f'Newman-Girvan $Q$ ({n_modules}-modulo partition)',
                   fontsize=13)
    ax1.set_ylabel(r'Spectral dimension $D_S$', fontsize=13)
    ax1.set_title(f'Spectral dimension vs modularity ({label}, '
                  f'M={n_modules}, {direction}-directed swap)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax2.set_xlabel(r'Modularity $Q$', fontsize=13)
    ax2.set_ylabel('Number of edges', fontsize=13)
    ax2.set_title('Edge count along trajectory', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def _run_one_M(initial, M, direction, config, rng):
    """Run sweep(s) for a single M value and return the combined
    measurement list (up + down deduped if both)."""
    labels = [v % M for v in range(initial.n_nodes)]
    optimizer = Optimizer(config, rng, labels=labels)
    measurements: list[Measurement] = []
    if direction in ('up', 'both'):
        measurements.extend(optimizer.sweep(initial, direction='up'))
    if direction in ('down', 'both'):
        down = optimizer.sweep(initial, direction='down')
        measurements.extend(down[1:] if direction == 'both' else down)
    return measurements


def _plot_module_sweep(measurements_by_m, label, direction, save_path):
    """Render a multi-M overlay plot.

    Two panels: ``D_S(large sigma)`` and ``D_S(small sigma)`` against
    ``Q``.  One line per M, color-coded along the viridis colormap so
    larger M is brighter.  Up- and down-sweep measurements for a
    given M are combined and sorted by Q, since both extend a single
    Q-vs-D_S curve from the same initial graph.

    Parameters
    ----------
    measurements_by_m : dict[int, list[Measurement]]
        Per-M measurement lists.
    label : str
        Initial-graph description for the plot title.
    direction : str
        ``'up'``, ``'down'``, or ``'both'``; for the title only.
    save_path : str
        Output PNG path.
    """
    Ms = sorted(m for m, meas in measurements_by_m.items() if meas)
    if not Ms:
        return

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=Ms[0], vmax=Ms[-1] if len(Ms) > 1 else Ms[0] + 1)

    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    for M in Ms:
        meas = measurements_by_m[M]
        Q = np.array([m.Q for m in meas])
        Dl = np.array([m.ds_large for m in meas])
        Ds = np.array([m.ds_small for m in meas])
        order = np.argsort(Q)
        color = cmap(norm(M))
        ax_l.plot(Q[order], Dl[order], color=color, marker='o',
                  linestyle='-', linewidth=1.3, markersize=4,
                  label=f'M={M}')
        ax_s.plot(Q[order], Ds[order], color=color, marker='s',
                  linestyle='-', linewidth=1.3, markersize=4,
                  label=f'M={M}')

    for ax, ds_label in [(ax_l, r'$D_S$ (large $\sigma$)'),
                          (ax_s, r'$D_S$ (small $\sigma$)')]:
        ax.set_xlabel(r'Newman-Girvan $Q$ (M-modulo partition)',
                      fontsize=13)
        ax.set_ylabel(ds_label, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2 if len(Ms) > 6 else 1)

    fig.suptitle(f'Spectral dimension vs modularity ({label}, '
                 f'M={Ms[0]}..{Ms[-1]}, {direction}-directed swap)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def main():
    args = _make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.target_n_modules < 1:
        raise ValueError(
            f"--target-n-modules must be >= 1, got {args.target_n_modules}")
    if args.sweep_modules and args.target_n_modules < 2:
        raise ValueError(
            "--sweep-modules requires --target-n-modules >= 2 "
            f"(got {args.target_n_modules})")

    rng = np.random.default_rng(args.seed)
    initial, label = _build_initial(args, rng)
    logger.info("Initial: %s, %d edges", label, initial.n_edges)
    if args.thermalization > 0:
        initial = initial.thermalize(args.thermalization, rng)
        logger.info("Thermalized: %d swaps (%d edges, bipartite=%s)",
                    args.thermalization, initial.n_edges,
                    initial.is_bipartite())

    config = OptimizerConfig(
        target_dq=args.target_dq,
        max_iterations=args.max_iterations,
        n_diffusion_walks=args.n_diffusion_walks,
        max_sigma=args.max_sigma,
        thermalization=args.thermalization,
        bipartite_check_frequency=args.bipartite_check_frequency,
        negative_retry_max=args.negative_retry_max,
        epsilon_q_max=args.epsilon_q_max,
    )

    t0 = time.time()
    if args.sweep_modules:
        Ms = list(range(2, args.target_n_modules + 1))
        logger.info("Module sweep (%s): M=2..%d, dq=%.4f, max_iter=%d",
                    args.direction, args.target_n_modules,
                    args.target_dq, args.max_iterations)
        measurements_by_m: dict[int, list[Measurement]] = {}
        for M in Ms:
            logger.info("--- M=%d ---", M)
            measurements_by_m[M] = _run_one_M(
                initial, M, args.direction, config, rng)
        elapsed = time.time() - t0
        n_total = sum(len(v) for v in measurements_by_m.values())
        logger.info("Done in %.1fs across %d M values, "
                    "%d measurements total", elapsed, len(Ms), n_total)
        _plot_module_sweep(measurements_by_m, label, args.direction,
                           args.save)
    else:
        M = args.target_n_modules
        logger.info("Module-targeted sweep (%s): M=%d, dq=%.4f, "
                    "max_iter=%d", args.direction, M, args.target_dq,
                    args.max_iterations)
        measurements = _run_one_M(initial, M, args.direction, config, rng)
        elapsed = time.time() - t0
        logger.info("Done in %.1fs, %d measurements",
                    elapsed, len(measurements))
        _plot(measurements, label, args.direction, args.save, M)
    logger.info("Saved %s", args.save)


if __name__ == '__main__':
    main()
