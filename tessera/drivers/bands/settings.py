# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The approximations a run makes for the sake of its cost, in one place, each
adjustable from the command line (`python -m tessera.drivers.bands.gaas
ab-initio --help`). Nothing here changes what is computed in the limit; every
entry trades cost against a truncation, and a run records the values it used.

A request for an order that is not implemented is refused by name. It is never
replaced by a lower one.
"""
from dataclasses import asdict, dataclass

# The orders of each expansion that exist in the code today.
IMPLEMENTED = {"self_energy_order": (1,), "zero_momentum_order": (1,)}


@dataclass(frozen=True)
class Approximations:
    """`self_energy_order`: the number of terms kept in the expansion of the
    self-energy in the screened interaction W (1 is Sigma = i G W; 1 to 5).

    `zero_momentum_order`: the number of terms kept in the expansion, in the
    momentum transfer about every sampled momentum, of the self-energy integrand
    over the part of momentum space the sampling leaves out (1 is the value at
    vanishing momentum times the zero-momentum constant; 1 to 5).

    `refinement_terms`: the number of terms of the refinement series that
    `GridCoulombKernel.zero_momentum_constant` removes (1 to 5).

    `lattice_images`: the number of periodic images per axis in the lattice sums
    of the ionic potential, the projectors' short-range parts and the atomic
    density (odd, at least 1).

    `frequency_nodes`: the number of terms of the Chebyshev series that carries
    the screened interaction along the imaginary axis in
    `KineticBasisScreening` (at least 5)."""
    self_energy_order: int = 3
    zero_momentum_order: int = 3
    refinement_terms: int = 5
    lattice_images: int = 5
    frequency_nodes: int = 64

    def __post_init__(self):
        for name in ("self_energy_order", "zero_momentum_order", "refinement_terms"):
            if not 1 <= getattr(self, name) <= 5:
                raise ValueError(f"{name} is between 1 and 5")
        if self.lattice_images < 1 or self.lattice_images % 2 == 0:
            raise ValueError("lattice_images is odd and at least 1")
        if self.frequency_nodes < 5:
            raise ValueError("frequency_nodes is at least 5")

    def require_implemented(self):
        """Refuse, by name, an order that does not exist yet."""
        for name, available in IMPLEMENTED.items():
            if getattr(self, name) not in available:
                raise NotImplementedError(
                    f"{name} = {getattr(self, name)} is not implemented (available: "
                    f"{', '.join(str(order) for order in available)}); it is not replaced by a lower order")

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

    @classmethod
    def from_arguments(cls, args):
        return cls(args.self_energy_order, args.zero_momentum_order, args.refinement_terms, args.lattice_images,
                   args.frequency_nodes)
