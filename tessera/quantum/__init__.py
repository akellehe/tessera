"""tessera.quantum -- Schwinger model on a Jordan-Wigner spin chain via DMRG.

Ground-state DMRG for the 1+1D Kogut-Susskind Schwinger model with U(1)
total-charge conservation, contiguous-cut Schmidt spectra and the
majorization poset on those spectra, a q-qbar quench + 2-site TDVP
pipeline that produces real-time-evolution snapshots, and an end-to-end
causal-order comparison between the majorization order, the Lieb-
Robinson cone, and the (regular-chain) causet order.
The C++ backend is ITensor v3 vendored under ``third_party/itensor``;
this Python layer is a thin result viewer: Python and C++ cross as
rarely as possible. No MPS or
MPO objects cross the language barrier — only scalar configs in and
scalar / list diagnostics out.

API style
---------

Every Python-visible operation is a method on a coarse-grained class —
there are no free functions in ``tessera.quantum``. The four user-facing
classes are:

* :class:`SchwingerModel`  — DMRG ground-state pipeline.
* :class:`SchwingerQuench` — q-qbar quench + TDVP + causal-order
                              comparison pipeline.
* :class:`Majorization`    — static utility: predicate-driven poset
                              construction and pairwise order-agreement
                              statistics.
* :class:`Causet`          — static utility: tessera.Spacetime → causet
                              adapters.

Plus the data classes (:class:`QuantumConfig`, :class:`GroundStateResult`,
:class:`SchmidtSpectra`, :class:`TDVPConfig`, :class:`TDVPSnapshot`, …)
and the :class:`MajorizationPredicate` hierarchy
(:class:`StandardMajorization`, :class:`LogConcaveMajorization`,
:class:`PeakRadialMajorization`) that callers configure workflows with.

Availability
------------
The quantum subsystem is an optional ITensor-backed C++ component. Like
CUDA support, it is auto-detected at build time — compiled in whenever the
ITensor submodule is present and a BLAS/LAPACK backend is available. To
enable it from scratch in one step::

    TESSERA_QUANTUM=1 pip install -e .

``TESSERA_QUANTUM=1`` checks out the ITensor submodule for you; ``=0``
forces the subsystem off.

This module always imports cleanly, regardless of how tessera was built.
Call :func:`is_available` to test whether the subsystem is present;
accessing a workflow class in a build without it raises
:class:`ImportError` with rebuild instructions::

    import tessera.quantum as q
    if q.is_available():
        result = q.SchwingerModel(q.QuantumConfig()).solve()

Hamiltonian
-----------------------------------------------------

After Jordan-Wigner mapping and Gauss's-law elimination, the staggered
Schwinger Hamiltonian on N sites with open boundary conditions is::

    H = H_hop + H_m + H_E

    H_hop = (1/(4a)) Σ_{n=1..N-1} (X_n X_{n+1} + Y_n Y_{n+1})
    H_m   = (m/2)    Σ_{n=1..N}   (-1)^n σ^z_n
    H_E   = (g²a/2)  Σ_{n=1..N-1} L_n²

    L_n   = L0 + Σ_{k=1..n} [(1 - σ^z_k)/2 - (1 - (-1)^k)/2]

Quickstart
----------

Compute the ground state at the spec parameters::

    >>> from tessera.quantum import QuantumConfig, SchwingerModel
    >>> cfg = QuantumConfig()
    >>> cfg.N = 20
    >>> cfg.a = 1.0; cfg.g = 1.0
    >>> cfg.m = 0.0
    >>> cfg.L0 = 0.0
    >>> cfg.maxBondDim = 100
    >>> cfg.nSweeps = 12
    >>> result = SchwingerModel(cfg).solve()
    >>> result.energy < 0
    True

Schmidt spectra and majorization poset
--------------------------------------

For each contiguous interval A = [i, j] on the spin chain, the Schmidt
spectrum λ_A is the list of eigenvalues of ρ_A = Tr_{Ā}|ψ⟩⟨ψ|, sorted
non-increasingly. Majorization (μ ≼ λ iff "λ is at least as
concentrated as μ") defines a partial order on the cuts; the Hasse
diagram is the transitive reduction of the strict-majorization graph.

Construct a poset directly from a list of spectra::

    >>> from tessera.quantum import Majorization, StandardMajorization
    >>> p = Majorization.posetOf([[1/3]*3, [0.5, 0.5], [1.0]])
    >>> p.getNodeCount, sorted(p.covers)
    (3, [(1, 0), (2, 1)])

Or run the full DMRG → Schmidt → poset pipeline in one call::

    >>> r = SchwingerModel(cfg).solveWithMajorization()
    >>> r.spectra.N
    20

The cut family is contiguous intervals 1 ≤ i ≤ j ≤ N excluding the
trivial full-chain bipartition [1, N] | ∅; this is N(N+1)/2 - 1 cuts
total.

q-qbar quench and TDVP real-time evolution
------------------------------------------

The :meth:`SchwingerQuench.evolve` method runs the full DMRG → quench →
TDVP pipeline. The quench operator is

    U_qqbar(i0, d)  =  σ⁻_{i0} · σ⁺_{i0 + d}

which on the heavy-quark Néel vacuum ``|↑↓↑↓ … ⟩`` creates a +1 flux tube
on the d links between sites i0 and i0+d (Buyens 2014 string state).
Parity constraint: i0 odd + d odd. Example::

    >>> from tessera.quantum import TDVPConfig, SchwingerQuench
    >>> cfg = TDVPConfig()
    >>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0
    >>> cfg.i0 = 5; cfg.d = 5
    >>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshotEvery = 5
    >>> r = SchwingerQuench(cfg).evolve()
    >>> r.snapshots[0].lProfile[:3]
    [-1.0, -0.0, -0.0]

Causal-order comparison
-----------------------

:meth:`SchwingerQuench.compareCausalOrders` ties the ground-state,
Schmidt, and TDVP pipelines together:
DMRG ground state → q-qbar quench → TDVP loop → build three partial
orders on (cut, time) labels → compare. The orders are:

  ≼_maj: strict-majorization on Schmidt spectra (across time)
  ≼_LR:  Lieb-Robinson cone, dist(A, B) ≤ vLr · (t_B - t_A)
  ≼_cs:  causet — time-only on regular chain

Each comparison reports Kendall-τ, discordant-pair fraction, and Hasse-
graph edit distance. Example::

    >>> cfg = TDVPConfig()
    >>> cfg.N = 10; cfg.m = 0.5; cfg.g = 1.0
    >>> cfg.i0 = 3; cfg.d = 3
    >>> cfg.dt = 0.1; cfg.T = 1.0; cfg.snapshotEvery = 1
    >>> r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
    >>> r.lrVsCs.kendallTau
    1.0

The ≼_LR ⊂ ≼_cs invariant gives the strongest sanity check: τ = 1.0
exactly because every ≼_LR pair is also a ≼_cs pair in the same
direction.

References
----------

* Bañuls, Cichy, Cirac, Jansen, *JHEP* **11**, 158 (2013), arXiv:1305.3765
  -- primary reference for the Hamiltonian convention and benchmarks.
* Schwinger, *Phys. Rev.* **128**, 2425 (1962) -- original gauge theory.
* Coleman, *Ann. Phys.* **101**, 239 (1976) -- massive Schwinger model.
* Kogut, Susskind, *Phys. Rev. D* **11**, 395 (1975) -- staggered
  fermion lattice formulation.
* ITensor library: https://itensor.org -- the MPS/MPO backend.

"""

# ─── C++ export names ──────────────────────────────────────────────────────
# Every name this module re-exports from the C++ ``quantum`` submodule.
# Declared once: bound below when the subsystem is available, and used to
# raise a build-aware error when it is not.
_EXPORTS = (
    # Data classes (configs, results, labels, posets)
    "QuantumConfig", "GroundStateResult", "Interval", "SchmidtSpectra",
    "Poset", "GroundStateMajorizationResult", "TDVPConfig", "TDVPSnapshot",
    "QuenchResult", "InteractionConfig", "InitialChargeMode", "LabelSpacetime",
    "CausalOrders", "OrderAgreement", "CausalComparisonReport", "CausetChain",
    # MajorizationPredicate hierarchy
    "MajorizationPredicate", "StandardMajorization", "LogConcaveMajorization",
    "PeakRadialMajorization",
    # Coarse-grained workflow classes
    "SchwingerModel", "SchwingerQuench", "InteractionSimulation",
    "Majorization", "Causet", "MutualInformation", "ChoiJamiolkowski",
    # Exterior-algebra / graded-tensor primitives: occupation
    # bitsets with the prefix-popcount sign rule, the CAR operator layer,
    # the graded chain/tensor differential, the Fock direct-sum functor and
    # dGamma, and the edge-mode registry with its deterministic compilation
    # order and relabeling parity.
    "OccupationBitset", "ExteriorAlgebra", "GradedTensorComplex",
    "FockDirectSum", "EdgeModeRecord", "EdgeModeRegistry",
    # Lazy graded Fock oracle and boundary carrier: the
    # expression-DAG engine over the #766 primitives — lazy graded tensor
    # products with crossing-only expansion, sector direct sums, Slater
    # wedges with the exact projector covariance, bit-level dGamma, the
    # vacuum embedding + inductive compatibility read, exact certification
    # / stated truncation, and content-hashed DAG checkpoints.
    "LazyNodeKind", "LazySectorKind", "LazyFockState", "LazyScalarRead",
    "LazySlaterReference", "LazyCovarianceRead", "LazyCompatibilityRead",
    "LazyFockEngine",

    # Quasi-free covariance layer: the number-conserving
    # covariance state Gamma_ij = <a_j+ a_i> with exact Wick contraction of
    # every polynomial certificate, exact one-particle propagation, the
    # certificates-blind mean-field loop, cached Wick reads, and
    # checkpoint serialization of Gamma.
    "CovarianceState", "WickCertificateRead", "MeanFieldStepRead",
    # KI + QuantumSimplex (Van Raamsdonk-metric simplex factory)
    "QuantumSimplex", "QuantumSimplexPosition", "QuantumVertex",
    "createQuantumVertex",
    "KoashiImotoResult", "KoashiImotoBlock", "KoashiImotoTolerances",
    "koashiImotoDecompose", "partialTraceA", "partialTraceB",
    "mutualInformation",
    # Free functions — compareOrders: pairwise agreement statistics between
    # two Posets on a shared label set (see docs/source/causal_sets.md).
    "compareOrders",
)

_UNAVAILABLE_MESSAGE = (
    "tessera.quantum is unavailable: this build of tessera does not include "
    "the quantum subsystem (Schwinger model / DMRG / ITensor).\n\n"
    "Enable it with a single command — TESSERA_QUANTUM=1 checks out the "
    "ITensor submodule and builds the subsystem:\n\n"
    "    TESSERA_QUANTUM=1 pip install -e .\n\n"
    "TESSERA_QUANTUM=0 forces it off. See docs/source/quantum-plan.md."
)

try:
    # tessera._tessera is a single C extension (.so), not a Python package, so
    # the submodule is exposed as an attribute rather than a separate
    # importable module. Pybind11's def_submodule installs it on the parent
    # module's __dict__ at import time; we just look it up. In a build without
    # the quantum subsystem the attribute is simply absent (AttributeError).
    from tessera import _tessera
    _quantum = _tessera.quantum
    _AVAILABLE = True
    _IMPORT_ERROR = None
except (ImportError, AttributeError) as exc:
    _quantum = None
    _AVAILABLE = False
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """Return ``True`` if this tessera build includes the quantum subsystem.

    Safe to call and branch on regardless of how tessera was built — the
    same role :func:`torch.cuda.is_available` plays for CUDA. When this
    returns ``False``, accessing any quantum export raises
    :class:`ImportError` with rebuild instructions.
    """
    return _AVAILABLE


if _AVAILABLE:
    # Bind every C++ export onto this module's namespace.
    for _name in _EXPORTS:
        globals()[_name] = getattr(_quantum, _name)
    del _name
else:
    # PEP 562 module-level __getattr__: consulted only for names not already
    # bound. Turn access to a known quantum export into an actionable error
    # rather than a bare NameError, while leaving genuine typos as
    # AttributeError.
    def __getattr__(name):
        if name in _EXPORTS:
            raise ImportError(_UNAVAILABLE_MESSAGE) from _IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The theta quantization register is Python and NumPy only: a marking
# lattice quantized at level k, on which a direct sum of tori is a tensor
# product and an integer monodromy acts by its Weil matrix. It needs nothing
# from the C++ backend, so it is bound regardless of `_AVAILABLE`.
from .theta_register import ThetaRegister, WeilFit  # noqa: E402
from .surface_periods import SurfacePeriods  # noqa: E402
from .symmetric_genus_two import SymmetricGenusTwo  # noqa: E402

__all__ = [*_EXPORTS, "is_available", "ThetaRegister", "WeilFit", "SurfacePeriods", "SymmetricGenusTwo"]
