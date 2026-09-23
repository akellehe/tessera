# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The anti-cluster certificate (#1219): the whitepaper proposes that an
anti-cluster is an effective void, a region whose enclosing surface is a
certified coexact near-cycle of L_2, whose interior spectrum is nearly empty,
and whose enclosing coorientation is inward; falsifier 9 fires when a certified
anti-lineage carries no such void, or when a void's lineage does not pair with a
cluster's. The degree-zero band is the weight-aware second proposer of cluster
supports, beside Newman-Girvan modularity on the combinatorial one-skeleton."""
import itertools

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs
from tests.observables.test_effective_components_python import (_operator, bridged_rings,
                                                                kuhn_block)

# The scale every void on the Kuhn block is read at: far below the first
# nonzero level of every degree, so the enclosed band is the near-kernel alone.
VOID_SCALE = 1e-6
# The squared length of a weak link: a bottleneck of low conductance, which the
# covariant operator sees and the combinatorial one-skeleton does not.
WEAK = 1.0e6
# The scale the arcs of `weighted_cycle` are read at; see its docstring.
ARC_SCALE = 1.0e-4


def cavity_corners(n=3):
    """The eight vertex ids of the cube `kuhn_block(n, hollow=True)` removes."""
    side = n + 1
    origin = n // 2
    return sorted(p[0] + side * (p[1] + side * p[2])
                  for p in itertools.product((origin, origin + 1), repeat=3))


def weighted_cycle(length=12, weak=(4, 11), bridge=WEAK):
    """A cycle on `length` vertices whose edges (j, j+1 mod length) are of unit
    squared length except at the positions named by `weak`, which carry
    `bridge`. Every vertex has degree two, so the combinatorial one-skeleton is
    a uniform cycle with no community structure in it; the covariant operator
    sees two effective components, the two arcs the weak links separate.

    The arcs are read at `ARC_SCALE`. A long edge is a weak link, but in the
    Whitney metric its two endpoints also carry its length as mass, so beside
    the constant and the arc-splitting mode (at about 1e-5 here) the degree-zero
    spectrum holds one heavy-endpoint mode per arc (at about 1e-3) before the
    arcs' own levels begin (from about 0.3). The scale sits between the split
    and the heavy-endpoint modes, where the band is the two arcs and nothing
    else."""
    cells = [[j, (j + 1) % length] for j in range(length)]
    K = cob.ChainComplex.fromTopCells(cells)
    edges = [tuple(e) for e in K.kSimplexVertices(1)]
    weak_edges = {tuple(sorted(((j + 1) % length, j))) for j in weak}
    s = [bridge if e in weak_edges else 1.0 for e in edges]
    return K, edges, _operator(K, s)


def blind_modularity(K):
    """Newman-Girvan modularity on the combinatorial one-skeleton of `K`: every
    edge of unit weight, which is the metric-blind proposer the whitepaper
    names."""
    edges = K.kSimplexVertices(1)
    src = [int(e[0]) for e in edges]
    tgt = [int(e[1]) for e in edges]
    cells = [int(v[0]) for v in K.kSimplexVertices(0)]
    return obs.PersistentModularity.fromWeightedEdges(src, tgt, [1.0] * len(edges), cells)


class TestTheAntiClusterCertificate:
    """A region whose enclosing surface is a certified coexact near-cycle of
    L_2, whose interior spectrum is nearly empty, and whose enclosing
    coorientation is inward."""

    def test_a_cavity_is_a_certified_anti_cluster(self):
        """The eight corners of the removed cube enclose the cavity: they hold
        no cell of the complex, their twelve faces are the cavity wall, that
        wall is closed and lies in the certified coexact part of the degree-two
        band, and no mode of the operator lives inside."""
        K, cov = kuhn_block(3, hollow=True)
        certificate = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE)
        assert certificate.region == cavity_corners()
        # The cavity's own cells are not in the complex, so the region holds
        # none; its enclosing surface is the part of the complex's own boundary
        # that its faces carry.
        assert list(certificate.interiorCells) == []
        assert len(certificate.surface) == 12
        assert np.count_nonzero(np.abs(np.array(certificate.enclosingSurface)) > 1e-12) == 12
        # The first clause: a certified coexact near-cycle of L_2.
        assert certificate.band.degree == 2 and certificate.band.certified
        assert certificate.voids == 1
        assert certificate.cycleResidual < 1e-10
        assert certificate.voidContent > certificate.minimumVoidContent
        assert certificate.minimumVoidContent == 0.1
        # The second clause: the interior spectrum is nearly empty. The eight
        # corners carry a Dirichlet spectrum, and none of it is in the window.
        assert certificate.interiorDegree == 0
        assert len(certificate.interiorSpectrum) == 8
        assert certificate.interiorRank == 0 and certificate.interiorEmpty
        assert certificate.interiorFloor > VOID_SCALE
        # The third clause: the enclosing coorientation is inward.
        assert certificate.coorientation == obs.EnclosingCoorientation.Inward
        assert certificate.coorientationSource == obs.CoorientationSource.InteriorSpectrum
        assert certificate.certified and certificate.reason == ""

    def test_the_filled_block_has_no_anti_cluster(self):
        """The same region on the filled block holds the six cells of the cube.
        Its enclosing surface is their boundary, which is exact and reaches no
        coexact near-cycle, so the certificate fails and says so."""
        K, cov = kuhn_block(3)
        certificate = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE)
        assert len(certificate.interiorCells) == 6
        assert len(certificate.surface) == 12
        assert certificate.cycleResidual < 1e-10
        assert certificate.voids == 0 and certificate.voidContent == pytest.approx(0.0, abs=1e-12)
        assert not certificate.certified
        assert "coexact near-cycle of L_2" in certificate.reason
        # Every other clause still holds: the failure is the void alone.
        assert certificate.interiorEmpty
        assert certificate.coorientation == obs.EnclosingCoorientation.Inward

    def test_a_region_that_holds_a_mode_is_a_blob(self):
        """A bubble and a blob are told apart by the interior spectrum and by
        the coorientation that follows it: read at a scale that takes in the
        region's own lowest Dirichlet level, the cavity's corners hold a mode
        of the operator and their surface is cooriented outward."""
        _, cov = kuhn_block(3, hollow=True)
        bubble = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE)
        options = obs.AntiClusterOptions()
        options.minimumGap = 1.0
        blob = obs.EffectiveTopology.antiCluster(cov, cavity_corners(),
                                                 1.01 * bubble.interiorFloor, options)
        assert blob.interiorRank >= 1 and not blob.interiorEmpty
        assert blob.coorientation == obs.EnclosingCoorientation.Outward
        assert not blob.certified
        assert "interior spectrum is not nearly empty" in blob.reason
        assert "enclosing coorientation is not inward" in blob.reason

    def test_a_supplied_reference_decides_the_coorientation(self):
        """With no reference the enclosing surface carries no direction of its
        own, because the band fixes a complex near-cycle only up to a nonzero
        complex scale. A caller who has established one independently supplies
        it as a chain, and the certificate compares the two directly."""
        _, cov = kuhn_block(3, hollow=True)
        outward = np.array(
            obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE).enclosingSurface)
        options = obs.AntiClusterOptions()
        options.coorientationReference = outward
        along = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        assert along.coorientationSource == obs.CoorientationSource.Reference
        assert along.coorientationOverlap == pytest.approx(1.0, rel=1e-9)
        assert along.coorientation == obs.EnclosingCoorientation.Outward
        assert not along.certified
        options.coorientationReference = -outward
        against = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        assert against.coorientationOverlap == pytest.approx(-1.0, rel=1e-9)
        assert against.coorientation == obs.EnclosingCoorientation.Inward
        assert against.certified
        # A reference orthogonal to the enclosing surface reads no direction.
        # It sits on a face the surface does not reach, so the two are exactly
        # orthogonal rather than nearly so.
        elsewhere = np.zeros_like(outward)
        elsewhere[int(np.argmin(np.abs(outward)))] = 1.0
        assert np.abs(np.vdot(elsewhere, outward)) == 0.0
        options.coorientationReference = elsewhere
        orthogonal = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        assert orthogonal.coorientation == obs.EnclosingCoorientation.Undeclared
        assert not orthogonal.certified

    def test_the_interior_degree_is_declared(self):
        """The interior spectrum is read at a declared degree. Degree zero, the
        default, is the degree whose band carries the effective components; the
        cavity's corners hold cells of degree one and two as well, and neither
        of those spectra reaches the window either."""
        _, cov = kuhn_block(3, hollow=True)
        options = obs.AntiClusterOptions()
        for degree in (1, 2):
            options.interiorDegree = degree
            certificate = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
            assert certificate.interiorDegree == degree
            assert len(certificate.interiorSpectrum) > 0
            assert certificate.interiorRank == 0 and certificate.certified

    def test_refusals(self):
        _, cov = kuhn_block(3, hollow=True)
        with pytest.raises(ValueError, match="dimension three"):
            obs.EffectiveTopology.antiCluster(weighted_cycle()[2], [0, 1], VOID_SCALE)
        with pytest.raises(ValueError, match="which the complex does not have"):
            obs.EffectiveTopology.antiCluster(cov, [10 ** 6], VOID_SCALE)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            obs.EffectiveTopology.antiCluster(cov, cavity_corners(), 0.0)
        options = obs.AntiClusterOptions()
        options.coorientationReference = np.ones(3, dtype=complex)
        with pytest.raises(ValueError, match="coorientation reference"):
            obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        options = obs.AntiClusterOptions()
        options.interiorDegree = 4
        with pytest.raises(ValueError, match="outside"):
            obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)


class TestFalsifierNine:
    """Falsifier 9: a certified anti-lineage, N_Q = -1, whose support carries no
    certified coexact near-kernel -- no effective void -- or an effective void
    whose lineage does not pair-create with a cluster's; either returns the
    anti-cluster to a purely kinematic definition.

    The lineage number itself is not in this tree (it is the work of #1197), so
    each lineage here is declared by the cooriented cut it carries: the chain
    that coorients the enclosing surface of its support. An anti-lineage
    coorients it into the region and a cluster's lineage out of it."""

    def _cut(self, cov, support, sign):
        """The cooriented cut a lineage of the declared sign carries over its
        support: the region's enclosing surface, cooriented out of the region
        for a cluster (+1) and into it for an anti-cluster (-1)."""
        read = obs.EffectiveTopology.antiCluster(cov, support, VOID_SCALE)
        return sign * np.array(read.enclosingSurface)

    def test_a_certified_anti_lineage_carries_a_certified_void(self):
        """The falsifier does not fire on the cavity: the anti-lineage's support
        carries an effective void, the void is certified, and its enclosing
        coorientation agrees with the anti-lineage's cut."""
        _, cov = kuhn_block(3, hollow=True)
        options = obs.AntiClusterOptions()
        options.coorientationReference = self._cut(cov, cavity_corners(), -1)
        certificate = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        assert certificate.voids == 1 and certificate.voidContent > certificate.minimumVoidContent
        assert certificate.coorientationSource == obs.CoorientationSource.Reference
        assert certificate.coorientation == obs.EnclosingCoorientation.Inward
        assert certificate.certified

    def test_an_anti_lineage_on_a_filled_support_fires_the_falsifier(self):
        """The first clause of the falsifier: a certified anti-lineage whose
        support carries no certified coexact near-kernel. The same support on
        the filled block carries no void, so the anti-cluster has nothing but
        its kinematics left."""
        _, cov = kuhn_block(3)
        options = obs.AntiClusterOptions()
        options.coorientationReference = self._cut(cov, cavity_corners(), -1)
        certificate = obs.EffectiveTopology.antiCluster(cov, cavity_corners(), VOID_SCALE, options)
        assert certificate.coorientation == obs.EnclosingCoorientation.Inward
        assert certificate.voids == 0 and certificate.voidContent < certificate.minimumVoidContent
        assert not certificate.certified
        assert "coexact near-cycle of L_2" in certificate.reason

    def test_the_void_pairs_with_a_cluster_lineage(self):
        """The second clause: an effective void whose lineage does not
        pair-create with a cluster's. The cavity's anti-lineage and the lineage
        of a block of cells beside it carry opposite cuts over one and the same
        wall, so the two lineages sum to zero; only the anti-lineage's side
        carries the void, and the cluster's side holds the material."""
        _, cov = kuhn_block(3, hollow=True)
        corners = cavity_corners()
        anti = obs.AntiClusterOptions()
        anti.coorientationReference = self._cut(cov, corners, -1)
        cluster = obs.AntiClusterOptions()
        cluster.coorientationReference = self._cut(cov, corners, +1)
        bubble = obs.EffectiveTopology.antiCluster(cov, corners, VOID_SCALE, anti)
        blob = obs.EffectiveTopology.antiCluster(cov, corners, VOID_SCALE, cluster)
        # One wall, two lineages, opposite coorientations summing to zero.
        assert np.allclose(np.array(anti.coorientationReference)
                           + np.array(cluster.coorientationReference), 0.0)
        assert bubble.coorientationOverlap == pytest.approx(-blob.coorientationOverlap, rel=1e-9)
        assert bubble.coorientation == obs.EnclosingCoorientation.Inward
        assert blob.coorientation == obs.EnclosingCoorientation.Outward
        # The anti-lineage's side is the void; the cluster's side is not.
        assert bubble.certified and not blob.certified
        assert "enclosing coorientation is not inward" in blob.reason


class TestTheBandProposesSupports:
    """Modularity runs on the combinatorial one-skeleton and does not see the
    complex Hodge weights; the degree-zero band of the covariant operator is
    nothing but those weights. Both propose, and neither may veto."""

    def test_the_band_proposes_supports_modularity_does_not(self):
        """On a uniform cycle whose two weak links separate an arc of five
        vertices from an arc of seven, the combinatorial one-skeleton is the
        same at every edge and modularity cannot see the separation. The
        degree-zero band proposes both arcs, and each carries the certified
        band that accepts it."""
        K, _, cov = weighted_cycle()
        band = obs.EffectiveTopology.components(cov, ARC_SCALE)
        assert band.band.rank == 2 and band.certified
        assert sorted(sorted(c.support) for c in band.components) == [[0, 1, 2, 3, 4],
                                                                      [5, 6, 7, 8, 9, 10, 11]]
        settings = obs.PersistentModularityConfig()
        settings.resolutions = [1.0]
        communities = blind_modularity(K).discover(1.0, settings).components
        proposals = obs.ParticleClusters.proposeSupports(communities, band)
        offered = [list(p.support) for p in proposals]
        assert [list(c.support) for c in communities] == offered[:len(communities)]
        band_only = [p for p in proposals if p.band and not p.modularity]
        assert [list(p.support) for p in band_only] == [[0, 1, 2, 3, 4],
                                                        [5, 6, 7, 8, 9, 10, 11]]
        for proposal in band_only:
            assert proposal.modularityIndex == obs.ClusterSupportProposal.NO_PROPOSER
            assert proposal.bandIndex != obs.ClusterSupportProposal.NO_PROPOSER
            # The band's own certificate accepts the support modularity never
            # proposed: the separation and the gap of the degree-zero band.
            assert band.components[proposal.bandIndex].support == proposal.support
            assert band.band.certified and band.band.gap >= band.band.minimumGap
            assert 0.0 < proposal.crossProposerOverlap < 1.0

    def test_a_support_both_proposers_offer_is_one_proposal(self):
        """Two hexagonal rings joined by one long edge are two communities of
        the combinatorial one-skeleton and two effective components of the
        operator. Where the proposers agree the proposal is one, carrying both,
        and its overlap with the other proposer's supports is exactly one."""
        K, _, cov = bridged_rings(2)
        band = obs.EffectiveTopology.components(cov, 0.01)
        settings = obs.PersistentModularityConfig()
        communities = blind_modularity(K).discover(1.0, settings).components
        rings = [list(range(6)), list(range(6, 12))]
        assert sorted(list(c.support) for c in communities) == rings
        assert sorted(list(c.support) for c in band.components) == rings
        proposals = obs.ParticleClusters.proposeSupports(communities, band)
        assert len(proposals) == 2
        for index, proposal in enumerate(proposals):
            assert proposal.modularity and proposal.band
            assert proposal.modularityIndex == index
            assert list(band.components[proposal.bandIndex].support) == list(proposal.support)
            assert proposal.crossProposerOverlap == pytest.approx(1.0)
