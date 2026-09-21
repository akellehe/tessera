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
    mass = cell.pencil(read.kappa)[1]
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


def _cluster_means(levels, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(np.mean(levels[start:start + size]))
        start += size
    return np.array(out)


def folded_levels(epm, divisions, tolerance=1e-10, log=print):
    """The valence and lowest conduction levels at the three high-symmetry
    points, extrapolated from three meshes, against plane waves.

    The zone centre of the conventional cell carries the primitive zone centre
    and the three X points; its corner momentum (1/2, 1/2, 1/2) carries the four
    L points. A level of multiplicity g in the primitive cell appears as a
    cluster of 3 g states (X) or 4 g states (L), which the mesh splits at the
    order of its error, so each cluster is read as its mean. Energies are
    returned relative to the top of the valence band."""
    unit = 2.0 * np.pi / epm.a
    centre_reads = [zone_centre_read(epm, n, 24, tolerance) for n in divisions]
    spacings = [r["spacing"] for r in centre_reads]
    centre = [_cluster_means(r["energies"][r["from_centre"]], (1, 3, 1)) for r in centre_reads]
    boundary = [_cluster_means(r["energies"][~r["from_centre"]], (3, 3, 6, 3, 3)) for r in centre_reads]
    corner, certified = [], all(r["certified"] for r in centre_reads)
    for n in divisions:
        cell = CrystalCell(epm.conventional_lattice(), n, kinetic_scale=epm.kinetic_scale)
        read = cell.solve((0.5, 0.5, 0.5), 20, epm.potential(cell), tolerance=tolerance)
        certified = certified and read.certified()
        corner.append(_cluster_means(read.energies, (4, 4, 8, 4)))
        log(f"N={n:3d} corner read residual={read.residual:.1e} certified={read.certified()}")
    extrapolate = lambda values: richardson(spacings, values)[0]
    top = extrapolate(centre)[1]
    reference_centre = epm.plane_wave_bands((0.0, 0.0, 0.0), 5)
    reference_top = reference_centre[3]
    reference_x = epm.plane_wave_bands((unit, 0.0, 0.0), 6)
    reference_l = epm.plane_wave_bands((0.5 * unit,) * 3, 5)
    return {"certified": certified,
            "centre": extrapolate(centre) - top,
            "centre_reference": reference_centre[[0, 3, 4]] - reference_top,
            "boundary": extrapolate(boundary) - top,
            "boundary_reference": reference_x[[0, 1, 2, 4, 5]] - reference_top,
            "corner": extrapolate(corner) - top,
            "corner_reference": reference_l[[0, 1, 2, 4]] - reference_top}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The direct gap of GaAs from the Cohen-Bergstresser "
                                     "empirical pseudopotential on three meshes of the conventional cell.")
    parser.add_argument("--divisions", type=int, nargs=3, default=(16, 24, 32))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--out", default=None, help="write the reads as JSON")
    parser.add_argument("--points", action="store_true",
                        help="the levels at the zone centre, X and L instead of the gap alone")
    args = parser.parse_args(argv)
    if args.points:
        table = folded_levels(ZincBlendeEPM.gallium_arsenide(), args.divisions)
        for name in ("centre", "boundary", "corner"):
            print(name, np.round(table[name], 4), "plane waves", np.round(table[name + "_reference"], 4),
                  flush=True)
        print("certified", table["certified"])
        return table
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


# ---------------------------------------------------------------- ab initio

def ab_initio_levels(cation_upf, anion_upf, divisions, a=5.64, cutoff=25.0, bands=24, log=print,
                     mean_field="hartree_fock"):
    """Gallium arsenide with norm-conserving pseudopotentials read from
    `cation_upf` and `anion_upf`, in the mean field `mean_field`:
    "hartree_fock", the mean field of the Coulomb interaction (the Wick
    contraction of the quartic into its direct and exchange parts), or
    "hartree", the direct part alone, which tests the ionic potentials and the
    Hartree kernel against plane waves at a fraction of the cost. Either way the calculation runs self-consistently
    on three meshes of the conventional cell at its zone centre, extrapolated,
    against plane waves with the same pseudopotentials at the equivalent
    momenta (the primitive zone centre and the three X points).

    Levels are returned in electron volts from the top of the valence band, for
    the states of the primitive zone centre (the valence singlet, the valence
    triplet, the conduction singlet) and of the X points (clusters of three
    times the primitive multiplicities 1, 1, 2, 1, 1)."""
    from tessera.drivers.bands import BOHR, RYDBERG
    from tessera.drivers.bands.abinitio import Crystal, MeshCrystal, PlaneWaveCrystal
    from tessera.drivers.bands.pseudopotential import Pseudopotential
    cation, anion = Pseudopotential.from_upf(cation_upf), Pseudopotential.from_upf(anion_upf)
    lattice_constant = a / BOHR

    primitive = Crystal.zinc_blende(lattice_constant, cation, anion, conventional=False)
    unit = 2.0 * np.pi / lattice_constant
    momenta = [np.zeros(3)] + [unit * np.eye(3)[axis] for axis in range(3)]
    if mean_field not in ("hartree_fock", "hartree"):
        raise ValueError("mean_field is 'hartree_fock' or 'hartree'")
    plane_waves = PlaneWaveCrystal(primitive, momenta, [1, 1, 1, 1], cutoff)
    reference = (plane_waves.run_hartree_fock(8, lattice_constant) if mean_field == "hartree_fock"
                 else plane_waves.run(8))
    centre_reference = reference["levels"][0] * RYDBERG
    boundary_reference = reference["levels"][1] * RYDBERG
    top = centre_reference[3]
    log(f"plane waves ({mean_field}): converged={reference['converged']}, gap {centre_reference[4] - top:.4f} eV")

    conventional = Crystal.zinc_blende(lattice_constant, cation, anion, conventional=True)
    spacings, centre, boundary, certified, runs = [], [], [], True, []
    for n in divisions:
        started = time.time()
        mesh = MeshCrystal(conventional, n)
        run = mesh.run_hartree_fock(bands, log=log) if mean_field == "hartree_fock" else mesh.run(bands)
        levels = run["levels"] * RYDBERG
        half = n // 2
        read = type("Read", (), {"kappa": (0.0, 0.0, 0.0), "vectors": run["vectors"].astype(complex)})
        characters = translation_characters(mesh.cell, read, [(0, half, half), (half, 0, half), (half, half, 0)])
        from_centre = characters.real.sum(axis=0) > 1.0
        centre.append(_cluster_means(levels[from_centre], (1, 3, 1)))
        boundary.append(_cluster_means(levels[~from_centre], (3, 3, 6, 3, 3)))
        spacings.append(mesh.cell.spacing)
        certified = certified and run["certified"]
        runs.append({"divisions": n, "levels": levels, "from_centre": from_centre, "iterations": len(run["history"]),
                     "residual": run["residual"], "certified": run["certified"], "seconds": time.time() - started})
        log(f"N={n:3d} gap={centre[-1][2] - centre[-1][1]:.4f} eV iterations={len(run['history'])} "
            f"residual={run['residual']:.1e} certified={run['certified']} ({time.time() - started:.0f} s)")
    extrapolated_centre = richardson(spacings, centre)[0]
    extrapolated_boundary = richardson(spacings, boundary)[0]
    mesh_top = extrapolated_centre[1]
    return {"certified": certified, "runs": runs,
            "centre": extrapolated_centre - mesh_top, "centre_reference": centre_reference[[0, 3, 4]] - top,
            "boundary": extrapolated_boundary - mesh_top,
            "boundary_reference": boundary_reference[[0, 1, 2, 4, 5]] - top,
            "gap": extrapolated_centre[2] - mesh_top, "reference_gap": centre_reference[4] - top}
