# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.bands -- band structures from the covariant Whitney pencil.

A crystal is a periodic cell meshed by `PeriodicKuhnGrid`. Its one-particle
problem is the degree-zero covariant Whitney pencil: the stiffness and mass
matrices of piecewise-linear finite elements, dressed by the flat U(1)
connection of a crystal momentum, with a scalar potential entering as its
weighted mass matrix. The modules work up in stages, each against a reference
with a known answer:

* `crystal`      -- the cell, its pencil at a crystal momentum, the sparse solve,
                    Richardson extrapolation in the mesh spacing, and the
                    certificates every read carries.
* `potentials`   -- model potentials, and the Cohen-Bergstresser empirical
                    pseudopotential of a zinc-blende crystal with the plane-wave
                    diagonalization it is compared against.
* `spin`         -- two sheets as the two spin components, and the spin-orbit
                    block between them.
* `fiber`        -- a static potential as the phase of the timelike edges of the
                    history complex.
* `coulomb`      -- the finite-element Coulomb kernel, densities, and
                    Hartree-Fock on the covariance.
* `screening`    -- the independent-particle polarizability, the screened
                    interaction, and the one-shot quasiparticle correction.
* `response`     -- the derivative of a band energy with respect to the squared
                    edge lengths.
* `gaas`         -- the gallium arsenide runs.

Lengths are in angstrom and energies in electron volts, with
hbar^2 / 2 m_e = 3.80998 eV angstrom^2 and e^2 = 14.3996 eV angstrom.
"""

HBAR2_OVER_2M = 3.80998212   # eV angstrom^2
E2 = 14.3996455              # eV angstrom: e^2 / (4 pi epsilon_0)
RYDBERG = 13.605693123       # eV
BOHR = 0.529177211           # angstrom
