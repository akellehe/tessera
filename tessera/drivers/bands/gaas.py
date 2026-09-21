# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Gallium arsenide on the conventional cubic cell with the empirical
pseudopotential: the machinery test on the real crystal.

The conventional cell folds the three zone-boundary points X onto the zone
centre, so its spectrum at the zone centre is the union of the primitive bands
at the centre and at the three X points. The two families are told apart by how
an eigenvector transforms under the face-centred translations a (0, 1, 1) / 2,
a (1, 0, 1) / 2 and a (1, 1, 0) / 2, which are symmetries of the crystal and of
the mesh: a state from the zone centre is invariant under all three, and a
state from an X point changes sign under two of them. The direct gap is read
between the states of the first family: the valence triplet (irreducible
representation Gamma_15) and the conduction singlet (Gamma_1) above it.

The mesh is not cubic. Every cube is cut along the same body diagonal, which
leaves the threefold axis along (1, 1, 1) and removes the fourfold axes, so a
cubic triplet splits into a doublet and a singlet at the order of the mesh
error. The triplet is therefore read as its mean, in which the splitting
cancels to first order, and the splitting is reported.

Every energy is extrapolated in the mesh spacing from three meshes, and the
reference is a plane-wave diagonalization with the same form factors, so no
external number enters.
"""
import argparse
import json
import time

import numpy as np

from tessera.drivers.bands.crystal import CrystalCell, richardson
from tessera.drivers.bands.potentials import ZincBlendeEPM


def translation_characters(cell, read, shifts):
    """<z | M T z> per eigenvector and per grid translation in `shifts`."""
    mass = cell.mass.dressed(read.kappa)
    n = np.array(cell.divisions)
    n2, n3 = cell.divisions[1], cell.divisions[2]
    characters = []
    for shift in shifts:
        moved = (cell.index + np.asarray(shift)) % n
        permutation = (moved[:, 0] * n2 + moved[:, 1]) * n3 + moved[:, 2]
        translated = read.vectors[permutation, :]
        characters.append(np.einsum("ij,ij->j", read.vectors.conj(), mass @ translated))
    return np.array(characters)


def zone_centre_read(epm, divisions, count=24, tolerance=1e-10):
    """The lowest `count` bands of the conventional cell at the zone centre,
    each marked as coming from the primitive zone centre or from an X point."""
    if divisions % 2:
        raise ValueError("the face-centred translations need an even number of divisions")
    cell = CrystalCell(epm.conventional_lattice(), divisions, kinetic_scale=epm.kinetic_scale)
    started = time.time()
    read = cell.solve((0.0, 0.0, 0.0), count, epm.potential(cell), tolerance=tolerance)
    half = divisions // 2
    characters = translation_characters(cell, read, [(0, half, half), (half, 0, half), (half, half, 0)])
    from_centre = characters.real.sum(axis=0) > 1.0
    return {"divisions": divisions, "spacing": cell.spacing, "vertices": cell.size,
            "energies": read.energies, "from_centre": from_centre,
            "character_defect": float(np.abs(np.abs(characters.real.sum(axis=0) - 1.0) - 2.0).max()),
            "residual": read.residual, "conditioning": read.conditioning,
            "certified": read.certified(), "seconds": time.time() - started}


def gap_states(read):
    """The valence singlet, the valence triplet and the conduction singlet of
    the primitive zone centre, from a `zone_centre_read`."""
    centre = read["energies"][read["from_centre"]]
    if len(centre) < 5:
        raise ValueError("fewer than five zone-centre states were computed; raise count")
    triplet = centre[1:4]
    return {"valence_bottom": centre[0], "valence_top_mean": triplet.mean(),
            "valence_top_max": triplet.max(), "triplet_splitting": float(np.ptp(triplet)),
            "conduction_bottom": centre[4]}


def direct_gap(epm, divisions, count=24, tolerance=1e-10, log=print):
    """The direct gap on three meshes, extrapolated in the mesh spacing, with
    the plane-wave reference."""
    reads = []
    for n in divisions:
        reads.append(zone_centre_read(epm, n, count, tolerance))
        states = gap_states(reads[-1])
        log(f"N={n:3d} h={reads[-1]['spacing']:.3f} A vertices={reads[-1]['vertices']:7d} "
            f"gap={states['conduction_bottom'] - states['valence_top_mean']:8.4f} eV "
            f"triplet splitting={states['triplet_splitting']:.4f} eV residual={reads[-1]['residual']:.1e} "
            f"certified={reads[-1]['certified']} ({reads[-1]['seconds']:.0f} s)")
    spacings = [r["spacing"] for r in reads]
    states = [gap_states(r) for r in reads]
    extrapolated = {key: float(richardson(spacings, [s[key] for s in states])[0])
                    for key in ("valence_bottom", "valence_top_mean", "conduction_bottom")}
    reference = epm.plane_wave_bands((0.0, 0.0, 0.0), 8)
    return {"reads": reads, "states": states, "extrapolated": extrapolated,
            "gap": extrapolated["conduction_bottom"] - extrapolated["valence_top_mean"],
            "valence_width": extrapolated["valence_top_mean"] - extrapolated["valence_bottom"],
            "reference_gap": float(reference[4] - reference[3]),
            "reference_valence_width": float(reference[3] - reference[0]),
            "certified": all(r["certified"] for r in reads)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The direct gap of GaAs from the Cohen-Bergstresser "
                                     "empirical pseudopotential on three meshes of the conventional cell.")
    parser.add_argument("--divisions", type=int, nargs=3, default=(16, 24, 32))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--out", default=None, help="write the reads as JSON")
    args = parser.parse_args(argv)
    result = direct_gap(ZincBlendeEPM.gallium_arsenide(), args.divisions, args.count)
    print(f"extrapolated gap {result['gap']:.4f} eV, plane waves {result['reference_gap']:.4f} eV, "
          f"difference {1e3 * (result['gap'] - result['reference_gap']):+.1f} meV; "
          f"valence width {result['valence_width']:.4f} eV against {result['reference_valence_width']:.4f} eV; "
          f"certified={result['certified']}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    return result


if __name__ == "__main__":
    main()
