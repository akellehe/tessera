# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 6a of the band-structure drivers (#1159): gallium arsenide with the
Cohen-Bergstresser empirical pseudopotential on the conventional cell, against
a plane-wave diagonalization with the same form factors."""
import numpy as np
import pytest

from tessera.drivers.bands import gaas
from tessera.drivers.bands.crystal import richardson
from tessera.drivers.bands.potentials import ZincBlendeEPM


@pytest.fixture(scope="module")
def epm():
    return ZincBlendeEPM.gallium_arsenide()


def test_states_are_told_apart_by_the_face_centred_translations(epm):
    """Of the 24 lowest states of the conventional cell at the zone centre, six
    come from the primitive zone centre (the valence singlet and triplet, the
    conduction singlet and the next one) and eighteen from the three X points;
    the translation characters that tell them apart are exact."""
    read = gaas.zone_centre_read(epm, 12)
    assert read["certified"] and read["character_defect"] < 1e-10
    assert read["from_centre"].sum() == 6 and len(read["energies"]) == 24
    states = gaas.gap_states(read)
    assert states["valence_bottom"] < states["valence_top_mean"] < states["conduction_bottom"]
    # The mesh is trigonal, not cubic: the valence triplet is a doublet and a singlet.
    triplet = read["energies"][read["from_centre"]][1:4]
    assert triplet[1] - triplet[0] < 1e-8 and states["triplet_splitting"] > 0.1
    with pytest.raises(ValueError, match="even"):
        gaas.zone_centre_read(epm, 9)


def test_the_valence_width_extrapolates_to_the_plane_wave_value(epm):
    """On meshes coarse enough for a test the gap itself is not yet in the
    asymptotic regime (the conduction states are the hardest to resolve), but
    the valence band width already extrapolates to the plane-wave value."""
    reads = [gaas.zone_centre_read(epm, n) for n in (12, 16, 20)]
    spacings = [r["spacing"] for r in reads]
    widths = [gaas.gap_states(r)["valence_top_mean"] - gaas.gap_states(r)["valence_bottom"] for r in reads]
    reference = epm.plane_wave_bands((0, 0, 0), 4)
    assert all(r["certified"] for r in reads)
    assert abs(widths[-1] - (reference[3] - reference[0])) > 0.5
    assert richardson(spacings, widths)[0] == pytest.approx(reference[3] - reference[0], abs=0.2)


@pytest.mark.slow
def test_the_direct_gap_matches_plane_waves_to_twenty_millielectronvolts(epm):
    """The production run: 24, 32 and 40 divisions (up to 64,000 vertices),
    about a quarter of an hour. Measured: 1.4188 eV against 1.4186 eV."""
    result = gaas.direct_gap(epm, (24, 32, 40), log=lambda line: None)
    assert result["certified"]
    assert result["gap"] == pytest.approx(result["reference_gap"], abs=0.020)
    assert result["valence_width"] == pytest.approx(result["reference_valence_width"], abs=0.020)
    gaps = [s["conduction_bottom"] - s["valence_top_mean"] for s in result["states"]]
    assert gaps[0] < gaps[1] < gaps[2] < result["gap"]          # the asymptotic regime: monotone from below
