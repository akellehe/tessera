# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Experimentally determined values for the real-material fixtures of the band
drivers. A calculation of a real material is held to these; a plane-wave
calculation of the same model Hamiltonian is a consistency check of that model
and does not arbitrate a discrepancy.

References: Vurgaftman, Meyer & Ram-Mohan, Journal of Applied Physics 89, 5815
(2001), Table I, for the band parameters and the lattice constant; Strauch &
Dorner, Journal of Physics: Condensed Matter 2, 1457 (1990), for the phonons.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Semiconductor:
    """Energies in electron volts, lengths in angstrom, frequencies in terahertz.
    Gaps are measured from the top of the valence band at zero temperature."""
    name: str
    lattice_constant: float              # at 300 K
    gap_gamma: float
    gap_x: float
    gap_l: float
    spin_orbit_splitting: float
    varshni_alpha: float                 # eV / K, of the gap at the zone centre
    varshni_beta: float                  # K
    optical_phonon_transverse: float     # at the zone centre, 12 K
    optical_phonon_longitudinal: float

    def direct_gap(self, temperature=0.0):
        """E_g(T) = E_g(0) - alpha T^2 / (T + beta), the Varshni form."""
        return self.gap_gamma - self.varshni_alpha * temperature ** 2 / (temperature + self.varshni_beta)


GALLIUM_ARSENIDE = Semiconductor(
    name="GaAs", lattice_constant=5.65325, gap_gamma=1.519, gap_x=1.981, gap_l=1.815,
    spin_orbit_splitting=0.341, varshni_alpha=0.5405e-3, varshni_beta=204.0,
    optical_phonon_transverse=8.02, optical_phonon_longitudinal=8.55)
