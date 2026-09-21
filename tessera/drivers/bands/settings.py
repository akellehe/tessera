# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The approximations a run makes for the sake of its cost, in one place, each
adjustable from the command line (`python -m tessera.drivers.bands.gaas
ab-initio --help`). Nothing here changes what is computed in the limit; every
entry trades cost against a truncation, and a run records the values it used.

A request for an order that is not implemented is refused by name. It is never
replaced by a lower one.
"""
from dataclasses import asdict, dataclass, fields

# The orders of each expansion that exist in the code today.
IMPLEMENTED = {"self_energy_order": (1, 2, 3, 4, 5), "zero_momentum_order": (1, 2, 3, 4, 5)}
# The same on a momentum set of more than one momentum.
IMPLEMENTED_ON_A_SET = {"self_energy_order": (1, 2, 3), "zero_momentum_order": (1, 2, 3, 4, 5)}


@dataclass(frozen=True)
class Approximations:
    """`self_energy_order`: the number of terms kept in the expansion of the
    self-energy in the screened interaction W (1 is Sigma = i G W; 1 to 5). The
    terms beyond the first are the skeleton diagrams of `diagrams`: one at
    second order, six at third, 49 at fourth, 542 at fifth. An order k puts as
    many as 2 k - 1 fermion lines and k lines of the screened interaction in
    one energy denominator, so its cost grows as half of `vertex_bands` to the
    power 2 k - 1 times `vertex_poles` to the power k, and the diagrams are
    evaluated on the `vertex_bands` modes nearest the gap and the
    `vertex_poles` modes of the screened interaction that couple to them most
    strongly. `vertex_memory` is what the diagrams may hold at once, in GiB,
    over all worker processes; it changes the time of a run and none of its
    numbers (`diagrams.SkeletonSelfEnergy`).

    `zero_momentum_order`: how the self-energy integrand is averaged over the
    momentum transfers that sampling the zone centre leaves out. 1 is the closed
    form at vanishing momentum. k > 1 is the midpoint grid of k momentum
    transfers per axis (`momentum_nodes`): the Hartree-Fock pencil is solved at
    every node, the singular part of the integrand is averaged analytically,
    and the regular remainder by the grid (`RandomPhase.set_momentum_terms`);
    1 to 5.

    `refinement_terms`: the number of terms of the refinement series that
    `GridCoulombKernel.zero_momentum_constant` removes (1 to 5).

    `lattice_images`: the number of periodic images per axis in the lattice sums
    of the ionic potential, the projectors' short-range parts and the atomic
    density (odd, at least 1).

    `frequency_nodes`: the number of terms of the Chebyshev series that carries
    the screened interaction along the imaginary axis in
    `KineticBasisScreening` (at least 5).

    `momenta`: the momentum set on which the covariance is sampled, a uniform
    grid of `momenta` per axis of the cell through its zone centre
    (`momentum_set.uniform_set`); 1 is the zone centre alone. On a set the
    offsets of `zero_momentum_order` surround every transfer of the set
    (`momentum_set.set_nodes`), and the diagrams beyond the first order run
    over the states of the set nearest the gap (`SetScreening.set_vertex`).

    `exchange_history`: the number of earlier exchange updates of Hartree-Fock
    whose filled sections join the span in which the accelerator of the update
    minimizes the energy (`acceleration`; 0 is the span of the computed bands
    alone). It changes the number of updates and the cost of each, by about
    (bands + history x filled)^2 Poisson solves, and leaves the converged state
    where it is."""
    self_energy_order: int = 3
    zero_momentum_order: int = 3
    refinement_terms: int = 5
    lattice_images: int = 5
    frequency_nodes: int = 64
    vertex_bands: int = 12
    vertex_poles: int = 12
    momenta: int = 1
    exchange_history: int = 5
    vertex_memory: float = 8.0

    def __post_init__(self):
        for name in ("self_energy_order", "zero_momentum_order", "refinement_terms"):
            if not 1 <= getattr(self, name) <= 5:
                raise ValueError(f"{name} is between 1 and 5")
        if self.lattice_images < 1 or self.lattice_images % 2 == 0:
            raise ValueError("lattice_images is odd and at least 1")
        if self.frequency_nodes < 5:
            raise ValueError("frequency_nodes is at least 5")
        if self.vertex_bands < 2 or self.vertex_poles < 1:
            raise ValueError("vertex_bands is at least 2 and vertex_poles at least 1")
        if self.momenta < 1:
            raise ValueError("momenta is at least 1")
        if self.exchange_history < 0:
            raise ValueError("exchange_history is at least 0")
        if not self.vertex_memory > 0.0:
            raise ValueError("vertex_memory is positive")

    def require_implemented(self):
        """Refuse, by name, an order that does not exist yet."""
        for name, available in (IMPLEMENTED if self.momenta == 1 else IMPLEMENTED_ON_A_SET).items():
            if getattr(self, name) not in available:
                raise NotImplementedError(
                    f"{name} = {getattr(self, name)} is not implemented"
                    f"{'' if self.momenta == 1 else ' on a momentum set'} (available: "
                    f"{', '.join(str(order) for order in available)}); it is not replaced by a lower order")

    @property
    def momentum_nodes(self):
        """The momentum transfers of `zero_momentum_order` k > 1 as (reciprocal
        coordinates, weight): the midpoint grid of k per axis, a transfer and
        its opposite counted once (the integrand is even under time reversal).
        The node at zero transfer, which an odd k has, is taken as the mean of
        three small transfers along the axes, because the remainder is bounded
        there and its limit is what enters."""
        k = self.zero_momentum_order
        if k == 1:
            return []
        axis = [(i + 0.5) / k - 0.5 for i in range(k)]
        weights = {}
        for kappa in ((a, b, c) for a in axis for b in axis for c in axis):
            kappa = tuple(round(v, 12) + 0.0 for v in kappa)
            if max(abs(v) for v in kappa) < 1e-9:
                for direction in ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)):
                    weights[direction] = weights.get(direction, 0.0) + 1.0 / (3 * k ** 3)
                continue
            key = max(kappa, tuple(-v + 0.0 for v in kappa))
            weights[key] = weights.get(key, 0.0) + 1.0 / k ** 3
        return sorted(weights.items())

    @property
    def refinements(self):
        return (2, 3, 4, 6, 8)[5 - self.refinement_terms:]

    @property
    def images(self):
        half = self.lattice_images // 2
        return tuple(range(-half, half + 1))

    def record(self):
        return asdict(self)

    @staticmethod
    def add_arguments(parser):
        defaults = Approximations()
        group = parser.add_argument_group("approximations made for the sake of cost")
        group.add_argument("--self-energy-order", type=int, default=defaults.self_energy_order,
                           help="terms of the expansion of the self-energy in the screened interaction, 1 to 5 "
                                f"(default {defaults.self_energy_order}; implemented: "
                                f"{IMPLEMENTED['self_energy_order']})")
        group.add_argument("--zero-momentum-order", type=int, default=defaults.zero_momentum_order,
                           help="terms of the expansion in the momentum transfer over the unsampled part of momentum "
                                f"space, 1 to 5 (default {defaults.zero_momentum_order}; implemented: "
                                f"{IMPLEMENTED['zero_momentum_order']})")
        group.add_argument("--refinement-terms", type=int, default=defaults.refinement_terms,
                           help="terms of the refinement series of the zero-momentum constant, 1 to 5")
        group.add_argument("--lattice-images", type=int, default=defaults.lattice_images,
                           help="periodic images per axis in the lattice sums (odd)")
        group.add_argument("--frequency-nodes", type=int, default=defaults.frequency_nodes,
                           help="terms of the Chebyshev series along the imaginary frequency axis")
        group.add_argument("--vertex-bands", type=int, default=defaults.vertex_bands,
                           help="modes nearest the gap on the internal lines of the diagrams beyond the first order")
        group.add_argument("--vertex-poles", type=int, default=defaults.vertex_poles,
                           help="modes of the screened interaction kept in the diagrams beyond the first order")
        group.add_argument("--momenta", type=int, default=defaults.momenta,
                           help="momenta per axis of the set on which the covariance is sampled (1 is the zone centre; "
                                f"on a set the orders implemented are {IMPLEMENTED_ON_A_SET})")
        group.add_argument("--exchange-history", type=int, default=defaults.exchange_history,
                           help="earlier exchange updates whose filled sections join the span of the Hartree-Fock "
                                "accelerator (changes the cost of the loop, not the converged state)")
        group.add_argument("--vertex-memory", type=float, default=defaults.vertex_memory,
                           help="GiB the diagrams beyond the first order may hold at once; changes the time, not the numbers")

    @classmethod
    def from_arguments(cls, args):
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})
