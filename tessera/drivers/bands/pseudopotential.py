# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Norm-conserving pseudopotentials in the Unified Pseudopotential Format, and
the screened pseudo-atom as the test that a pseudopotential is resolved by a
mesh.

This module and `abinitio` work in Rydberg atomic units: lengths in bohr,
energies in rydberg, the kinetic operator is minus the Laplacian
(`kinetic_scale = 1`) and the Coulomb energy of two unit charges at distance r
is 2 / r, so the Coulomb kernel has the strength 8 pi.

A norm-conserving pseudopotential is a local potential `V_loc(r)`, which tends
to -2 Z / r, plus a separable nonlocal part

    V_nl = sum_{ij, lm} |beta_i Y_lm> D_ij <beta_j Y_lm| ,

with radial projector functions beta_i of angular momentum l_i and coefficients
D_ij that couple projectors of the same angular momentum. The file stores
r beta(r) and 4 pi r^2 rho(r) on a radial mesh.

References: Hamann, Schlueter & Chiang, Physical Review Letters 43, 1494 (1979),
for norm conservation; Kleinman & Bylander, Physical Review Letters 48, 1425
(1982), for the separable form.
"""
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass
class Pseudopotential:
    """One species: radial mesh `r` (bohr), local potential `local` (Ry), the
    projectors as (l, r beta(r)) pairs, their coefficients `D` (Ry), the atomic
    valence density `density` as 4 pi r^2 rho, and the valence charge."""
    element: str
    valence: float
    r: np.ndarray
    local: np.ndarray
    projectors: list
    D: np.ndarray
    density: np.ndarray

    @classmethod
    def from_upf(cls, path):
        root = ElementTree.parse(path).getroot()
        header = root.find("PP_HEADER").attrib
        if header.get("pseudo_type", "NC").strip() not in ("NC", "SL"):
            raise ValueError("only norm-conserving pseudopotentials are supported")
        numbers = lambda node: np.array(node.text.split(), dtype=float)
        nonlocal_part = root.find("PP_NONLOCAL")
        projectors = [(int(node.attrib["angular_momentum"]), numbers(node))
                      for node in nonlocal_part if node.tag.startswith("PP_BETA")]
        count = len(projectors)
        D = numbers(nonlocal_part.find("PP_DIJ")).reshape(count, count) if count else np.zeros((0, 0))
        for i, j in zip(*np.nonzero(D)):
            if projectors[i][0] != projectors[j][0]:
                raise ValueError("D couples projectors of different angular momentum")
        return cls(element=header["element"].strip(), valence=float(header["z_valence"]),
                   r=numbers(root.find("PP_MESH/PP_R")), local=numbers(root.find("PP_LOCAL")),
                   projectors=projectors, D=D, density=numbers(root.find("PP_RHOATOM")))

    @classmethod
    def from_hgh(cls, element, density=None):
        """The published relativistic pseudopotential of `element` with its
        spin-orbit coefficients, as closed forms (`spinorbit.hgh`)."""
        from tessera.drivers.bands.spinorbit import hgh
        return hgh(element, density)

    # -- radial functions at arbitrary radii

    def local_at(self, radius):
        """V_loc at `radius`, continued by its Coulomb tail beyond the mesh."""
        radius = np.asarray(radius, dtype=float)
        inside = np.interp(radius, self.r, self.local)
        return np.where(radius > self.r[-1], -2.0 * self.valence / np.maximum(radius, 1e-30), inside)

    def short_range_at(self, radius, width):
        """V_loc + 2 Z erf(r / sqrt(2) width) / r: what is left of the local
        potential once the potential of a Gaussian charge -Z of that width is
        taken out. It decays like the Gaussian."""
        from scipy.special import erf
        radius = np.asarray(radius, dtype=float)
        safe = np.maximum(radius, 1e-12)
        smooth = np.where(radius > 1e-8, 2.0 * self.valence * erf(safe / (np.sqrt(2.0) * width)) / safe,
                          2.0 * self.valence * np.sqrt(2.0 / np.pi) / width)
        return self.local_at(radius) + smooth

    def projector_at(self, index, radius):
        """beta_index(r) (the file stores r beta)."""
        radius = np.asarray(radius, dtype=float)
        l, r_beta = self.projectors[index]
        value = np.interp(radius, self.r, r_beta, right=0.0) / np.maximum(radius, 1e-12)
        if l == 0:
            # beta is finite at the origin; take it from the first mesh points.
            origin = np.polyfit(self.r[1:6], r_beta[1:6] / self.r[1:6], 2)[-1]
            value = np.where(radius < self.r[1], origin, value)
        else:
            value = np.where(radius < 1e-10, 0.0, value)
        return value

    def density_at(self, radius):
        """The atomic valence density rho(r)."""
        radius = np.asarray(radius, dtype=float)
        shell = np.interp(radius, self.r, self.density, right=0.0)
        origin = np.polyfit(self.r[2:8], self.density[2:8] / (4.0 * np.pi * self.r[2:8] ** 2), 2)[-1]
        return np.where(radius < self.r[2], origin, shell / (4.0 * np.pi * np.maximum(radius, 1e-12) ** 2))


# ---------------------------------------------------------------- the screened, confined pseudo-atom

def confinement(radius, strength):
    """strength * r^4 / 16: keeps the orbitals of the test atom inside a cell."""
    return strength * np.asarray(radius, dtype=float) ** 4 / 16.0


def screened_potential(pseudo, radius, strength):
    """V_loc + V_H[rho_atom] + confinement at `radius`: the Hartree potential of
    the neutral pseudo-atom frozen at its atomic density."""
    grid = np.arange(0.005, 40.0, 0.005)
    shell = np.interp(grid, pseudo.r, pseudo.density, right=0.0)
    step = grid[1] - grid[0]
    inside = np.cumsum(shell) * step
    outside = np.cumsum((shell / grid)[::-1])[::-1] * step
    hartree = 2.0 * (inside / grid + outside - shell / grid * step)
    total = pseudo.local_at(grid) + hartree
    return np.interp(radius, grid, total) + confinement(radius, strength)


def radial_levels(pseudo, l, count, strength, radius=16.0, step=0.01):
    """The lowest `count` levels of angular momentum `l` of the screened,
    confined pseudo-atom, in rydberg, by finite differences on u(r) = r psi(r).
    The separable part acts on u through <beta| psi> = int (r beta) u dr."""
    r = np.arange(step, radius, step)
    n = len(r)
    H = np.diag(2.0 / step ** 2 + screened_potential(pseudo, r, strength) + l * (l + 1) / r ** 2)
    H -= np.diag(np.ones(n - 1), 1) / step ** 2 + np.diag(np.ones(n - 1), -1) / step ** 2
    channel = [i for i, (li, _) in enumerate(pseudo.projectors) if li == l]
    if channel:
        B = np.array([np.interp(r, pseudo.r, pseudo.projectors[i][1], right=0.0) for i in channel]).T * np.sqrt(step)
        H += B @ pseudo.D[np.ix_(channel, channel)] @ B.T
    return scipy.linalg.eigvalsh(H, subset_by_index=[0, count - 1])


# ---------------------------------------------------------------- real spherical harmonics

def real_harmonics(l, vectors):
    """The 2 l + 1 real spherical harmonics of order l <= 2 on the directions of
    `vectors` (rows), orthonormal on the sphere; zero vectors give the l = 0
    value and zero otherwise."""
    norm = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(norm, 1e-300)[:, None]
    x, y, z = unit.T
    present = norm > 1e-12
    if l == 0:
        return [np.full(len(norm), 1.0 / np.sqrt(4.0 * np.pi))]
    if l == 1:
        return [np.where(present, np.sqrt(3.0 / (4.0 * np.pi)) * c, 0.0) for c in (x, y, z)]
    if l == 2:
        a, b = np.sqrt(15.0 / (4.0 * np.pi)), np.sqrt(5.0 / (16.0 * np.pi))
        functions = (a * x * y, a * y * z, a * z * x, b * (3.0 * z * z - 1.0), 0.5 * a * (x * x - y * y))
        return [np.where(present, f, 0.0) for f in functions]
    raise NotImplementedError("real harmonics are implemented up to l = 2")
