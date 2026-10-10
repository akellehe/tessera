// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: effective Betti numbers,
// topology and signatures. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_observables_effective_topology(py::module_ &m) {
  py::class_<EffectiveBettiNumber> effectiveBettiNumber(m, "EffectiveBettiNumber",
      "The effective Betti number of one degree of a declared operator at a scale: the number of "
      "eigenvalues with |lambda| <= epsilon, the moduli that bracket the window, their gap, and how "
      "the count was obtained. rank is -1 when unmeasured and reason says why. A count is certified "
      "when it is converged (its residual met the tolerance) and separated (gap >= minimumGap); a "
      "window that cuts through the spectrum returns its count uncertified.");
  py::enum_<EffectiveBettiNumber::Method>(effectiveBettiNumber, "Method")
      .value("DenseSpectrum", EffectiveBettiNumber::Method::DenseSpectrum)
      .value("SparsePencil", EffectiveBettiNumber::Method::SparsePencil)
      .value("Unmeasured", EffectiveBettiNumber::Method::Unmeasured);
  effectiveBettiNumber
      .def_readonly("degree", &EffectiveBettiNumber::degree)
      .def_readonly("rank", &EffectiveBettiNumber::rank)
      .def_readonly("lastInside", &EffectiveBettiNumber::lastInside)
      .def_readonly("firstOutside", &EffectiveBettiNumber::firstOutside)
      .def_readonly("gap", &EffectiveBettiNumber::gap)
      .def_readonly("minimumGap", &EffectiveBettiNumber::minimumGap)
      .def_readonly("method", &EffectiveBettiNumber::method)
      .def_readonly("converged", &EffectiveBettiNumber::converged)
      .def_readonly("separated", &EffectiveBettiNumber::separated)
      .def_readonly("certified", &EffectiveBettiNumber::certified)
      .def_readonly("reason", &EffectiveBettiNumber::reason);

  py::class_<EffectiveHodgeSplit>(m, "EffectiveHodgeSplit",
      R"doc(The exact/coexact split of one degree's effective band. The exact part is d_k^# of the
degree k-1 band (at k = 1 the bridge flows of the effective components); the coexact part
is the band's null space under the boundary, the near-cycles, which include the harmonic
chains. The effective holes (k = 1) and the effective voids (k = 2 in three dimensions)
are the coexact part. Frames are orthonormal bases of chains.)doc")
      .def_readonly("band", &EffectiveHodgeSplit::band)
      .def_readonly("exact", &EffectiveHodgeSplit::exact)
      .def_readonly("coexact", &EffectiveHodgeSplit::coexact)
      .def_readonly("exactFrame", &EffectiveHodgeSplit::exactFrame)
      .def_readonly("coexactFrame", &EffectiveHodgeSplit::coexactFrame)
      .def_readonly("boundarySingularValues", &EffectiveHodgeSplit::boundarySingularValues)
      .def_readonly("rankTolerance", &EffectiveHodgeSplit::rankTolerance)
      .def_readonly("splitGap", &EffectiveHodgeSplit::splitGap)
      .def_readonly("closure", &EffectiveHodgeSplit::closure)
      .def_readonly("frameResidual", &EffectiveHodgeSplit::frameResidual)
      .def_readonly("certified", &EffectiveHodgeSplit::certified)
      .def_readonly("reason", &EffectiveHodgeSplit::reason);

  py::class_<EffectiveComponentSupport>(m, "EffectiveComponentSupport",
      "One effective component: its pivot vertex, the vertex ids assigned to it, and its committor "
      "on every vertex in the canonical vertex order (one at its pivot, zero at the other pivots).")
      .def_readonly("pivot", &EffectiveComponentSupport::pivot)
      .def_readonly("support", &EffectiveComponentSupport::support)
      .def_readonly("committor", &EffectiveComponentSupport::committor);

  py::class_<EffectiveComponentPartition>(m, "EffectiveComponentPartition",
      "The effective components read from the degree-zero band: the band's count and gap (which "
      "certifies each support's isolation), one support per direction of the band, the "
      "conditioning of the pivot rows, and the partition-of-unity defect of the committors.")
      .def_readonly("band", &EffectiveComponentPartition::band)
      .def_readonly("components", &EffectiveComponentPartition::components)
      .def_readonly("pivotConditioning", &EffectiveComponentPartition::pivotConditioning)
      .def_readonly("partitionDefect", &EffectiveComponentPartition::partitionDefect)
      .def_readonly("certified", &EffectiveComponentPartition::certified)
      .def_readonly("reason", &EffectiveComponentPartition::reason);

  py::enum_<EnclosingCoorientation>(m, "EnclosingCoorientation",
      "Which way the enclosing surface of a region is cooriented: Inward from the material into the "
      "region (a bubble, what an anti-cluster requires), Outward from the region into the material (a "
      "blob), or Undeclared when a supplied reference is orthogonal to the enclosing surface.")
      .value("Inward", EnclosingCoorientation::Inward)
      .value("Outward", EnclosingCoorientation::Outward)
      .value("Undeclared", EnclosingCoorientation::Undeclared);

  py::enum_<CoorientationSource>(m, "CoorientationSource",
      "Where a coorientation was read from: Reference, a chain the caller supplied carrying a "
      "coorientation established independently, such as the cooriented cut of a lineage; or "
      "InteriorSpectrum, which reads the coorientation as inward exactly when the enclosed side is the "
      "one the operator's band does not reach, because with no reference the enclosing surface carries "
      "no direction of its own (the band fixes a complex near-cycle only up to a nonzero complex "
      "scale).")
      .value("Reference", CoorientationSource::Reference)
      .value("InteriorSpectrum", CoorientationSource::InteriorSpectrum);

  py::class_<AntiClusterOptions>(m, "AntiClusterOptions",
      "What the anti-cluster certificate is allowed to assume beyond the operator, the region and the "
      "scale: the band's tolerance and minimum gap, the share of the enclosing surface required in the "
      "certified coexact part of the degree-two band, the degree whose restriction to the region is "
      "the interior spectrum, and an optional coorientation reference chain with the modulus below "
      "which it reads no direction.")
      .def(py::init<>())
      .def_readwrite("tolerance", &AntiClusterOptions::tolerance)
      .def_readwrite("minimumGap", &AntiClusterOptions::minimumGap)
      .def_readwrite("minimumVoidContent", &AntiClusterOptions::minimumVoidContent)
      .def_readwrite("interiorDegree", &AntiClusterOptions::interiorDegree)
      .def_readwrite("coorientationReference", &AntiClusterOptions::coorientationReference)
      .def_readwrite("coorientationTolerance", &AntiClusterOptions::coorientationTolerance);

  py::class_<AntiClusterCertificate>(m, "AntiClusterCertificate",
      R"doc(The whitepaper's proposal that the support of a hole, an antiquark's occupation
deficit relative to the matched reference (WP v18 Sections 5 and 10), is an effective void,
evaluated on one declared region: a region whose enclosing surface is a certified coexact
near-cycle of L_2, whose interior spectrum is nearly empty, and whose enclosing
coorientation is inward. The region's cells and its enclosing surface come from the
declared vertices; voidContent is the share of the normalized enclosing surface that lies
in the certified coexact part of the degree-two band; interiorSpectrum is the operator
restricted to the region, and interiorRank counts how much of it lies inside the window. A
region declared around a cavity has no cells of its own, since the cavity's cells are not
in the complex.)doc")
      .def_readonly("region", &AntiClusterCertificate::region)
      .def_readonly("interiorCells", &AntiClusterCertificate::interiorCells)
      .def_readonly("surface", &AntiClusterCertificate::surface)
      .def_readonly("enclosingSurface", &AntiClusterCertificate::enclosingSurface)
      .def_readonly("band", &AntiClusterCertificate::band)
      .def_readonly("voids", &AntiClusterCertificate::voids)
      .def_readonly("voidContent", &AntiClusterCertificate::voidContent)
      .def_readonly("minimumVoidContent", &AntiClusterCertificate::minimumVoidContent)
      .def_readonly("cycleResidual", &AntiClusterCertificate::cycleResidual)
      .def_readonly("interiorDegree", &AntiClusterCertificate::interiorDegree)
      .def_readonly("interiorSpectrum", &AntiClusterCertificate::interiorSpectrum)
      .def_readonly("interiorRank", &AntiClusterCertificate::interiorRank)
      .def_readonly("interiorFloor", &AntiClusterCertificate::interiorFloor)
      .def_readonly("interiorEmpty", &AntiClusterCertificate::interiorEmpty)
      .def_readonly("coorientation", &AntiClusterCertificate::coorientation)
      .def_readonly("coorientationSource", &AntiClusterCertificate::coorientationSource)
      .def_readonly("coorientationOverlap", &AntiClusterCertificate::coorientationOverlap)
      .def_readonly("certified", &AntiClusterCertificate::certified)
      .def_readonly("reason", &AntiClusterCertificate::reason);

  py::class_<EffectiveTopology> effectiveTopology(m, "EffectiveTopology",
      R"doc(What a declared operator sees at a scale, as opposed to what the complex is. The actual
topology of a complex is its incidence (ChainComplex.bettiNumbers, built by the
spacetime Topology classes); the effective topology of an operator on it is
beta_k^eff(epsilon) = rank P_[0, epsilon](h_k(s, U)), which depends on the squared
lengths and the connection, certified by the gap between the enclosed band and the rest
of the spectrum. A bottleneck gives two effective components on one incidence
component; a torus with a connection of nontrivial holonomy keeps incidence Betti
numbers (1, 3, 3, 1) while every effective Betti number is zero. This class never
consults the incidence ranks.)doc");
  effectiveTopology.attr("DEFAULT_MINIMUM_GAP") = kDefaultMinimumGap;
  effectiveTopology
      .def_static("read", &EffectiveTopology::read, py::arg("operator"), py::arg("epsilon"),
           py::arg("tolerance") = 1e-10, py::arg("minimum_gap") = kDefaultMinimumGap,
           "Read every degree at scale epsilon: the dense Schur form below the crossover, the sparse "
           "pencil for degree zero above it, and unmeasured otherwise. A degree is certified when its "
           "residual meets the tolerance and its gap reaches minimum_gap.")
      .def_static("split", &EffectiveTopology::split, py::arg("operator"), py::arg("degree"),
           py::arg("epsilon"), py::arg("tolerance") = 1e-10, py::arg("minimum_gap") = kDefaultMinimumGap,
           "The exact/coexact split of the degree's band at scale epsilon.")
      .def_static("voids", &EffectiveTopology::voids, py::arg("operator"), py::arg("epsilon"),
           py::arg("tolerance") = 1e-10, py::arg("minimum_gap") = kDefaultMinimumGap,
           "The effective voids of a three-dimensional complex: the coexact part of the degree-two "
           "band (split(operator, 2, epsilon)), whose coexact count is the number of voids.")
      .def_static("components", &EffectiveTopology::components, py::arg("operator"), py::arg("epsilon"),
           py::arg("tolerance") = 1e-10, py::arg("minimum_gap") = kDefaultMinimumGap,
           "The effective components at scale epsilon: the supports of the degree-zero band, from the "
           "committors that the column-pivoted QR of the band recovers.")
      .def_static("antiCluster", &EffectiveTopology::antiCluster, py::arg("operator"),
           py::arg("region_vertices"), py::arg("epsilon"), py::arg("options") = AntiClusterOptions{},
           "The anti-cluster certificate on a region declared by its vertices: the whitepaper's "
           "proposal that the support of a hole, an antiquark's occupation deficit relative to the "
           "matched reference, is an effective void, evaluated clause by clause. The region "
           "supplies its cells and its enclosing surface, the surface's share of the certified coexact "
           "part of the degree-two band says whether it is a near-cycle of L_2, the operator restricted "
           "to the region is the interior spectrum, and the coorientation comes from a supplied "
           "reference chain or from that spectrum.")
      .def("epsilon", &EffectiveTopology::epsilon)
      .def("dimension", &EffectiveTopology::dimension)
      .def("degrees", &EffectiveTopology::degrees)
      .def("betti", &EffectiveTopology::betti,
           "(beta_0^eff, ..., beta_d^eff), with -1 for an unmeasured degree.")
      .def("certified", &EffectiveTopology::certified);

  py::class_<EffectivePlateau>(m, "EffectivePlateau",
      "A run of consecutive reads over which one degree keeps one certified count: the degree, the "
      "count, the indices and scales of the run's ends, and the smallest gap over it.")
      .def_readonly("degree", &EffectivePlateau::degree)
      .def_readonly("rank", &EffectivePlateau::rank)
      .def_readonly("first", &EffectivePlateau::first)
      .def_readonly("last", &EffectivePlateau::last)
      .def_readonly("firstEpsilon", &EffectivePlateau::firstEpsilon)
      .def_readonly("lastEpsilon", &EffectivePlateau::lastEpsilon)
      .def_readonly("gap", &EffectivePlateau::gap)
      .def("length", &EffectivePlateau::length);

  py::class_<EffectivePersistence>(m, "EffectivePersistence",
      R"doc(An effective count that persists across a stated range of scales: a sequence of effective
reads (one operator swept over epsilon, or a refinement or relaxation sequence read at one
scale) and, per degree, the runs of consecutive certified reads that share one count. A
degree persists when the whole sequence is one such run.)doc")
      .def(py::init<std::vector<EffectiveTopology>>(), py::arg("reads"))
      .def_static("sweep", &EffectivePersistence::sweep, py::arg("operator"), py::arg("epsilons"),
           py::arg("tolerance") = 1e-10, py::arg("minimum_gap") = kDefaultMinimumGap,
           "Read the operator at every scale of the sweep, in the order given.")
      .def("reads", &EffectivePersistence::reads)
      .def("dimension", &EffectivePersistence::dimension)
      .def("plateaus", &EffectivePersistence::plateaus, py::arg("degree"))
      .def("persistentRank", &EffectivePersistence::persistentRank, py::arg("degree"),
           "The count that persists at the degree across every read, -1 when none does.")
      .def("betti", &EffectivePersistence::betti)
      .def("certified", &EffectivePersistence::certified);

  py::class_<EffectiveSignatureCertificate>(m, "EffectiveSignatureCertificate",
      "The verdict of an effective signature on an effective read or a persistence range: required "
      "and measured Betti numbers (-1 where nothing is required, measured or persistent), whether "
      "they match, whether the reads were certified at every scale covered, the scales, and the "
      "smallest gap among the required degrees.")
      .def_readonly("signature", &EffectiveSignatureCertificate::signature)
      .def_readonly("epsilon", &EffectiveSignatureCertificate::epsilon)
      .def_readonly("scales", &EffectiveSignatureCertificate::scales)
      .def_readonly("expected", &EffectiveSignatureCertificate::expected)
      .def_readonly("measured", &EffectiveSignatureCertificate::measured)
      .def_readonly("matches", &EffectiveSignatureCertificate::matches)
      .def_readonly("certified", &EffectiveSignatureCertificate::certified)
      .def_readonly("gap", &EffectiveSignatureCertificate::gap)
      .def("holds", &EffectiveSignatureCertificate::holds);

  py::class_<EffectiveSignature>(m, "EffectiveSignature",
      "A named pattern of effective Betti numbers, recognized in an operator rather than built "
      "into a complex: the effective counterpart of a spacetime Topology. The claim is one of "
      "rank at a scale, not of homeomorphism type and not of incidence.")
      .def("name", &EffectiveSignature::name)
      .def("betti", &EffectiveSignature::betti)
      .def("certify",
           py::overload_cast<const EffectiveTopology &>(&EffectiveSignature::certify, py::const_),
           py::arg("topology"))
      .def("certify",
           py::overload_cast<const EffectivePersistence &>(&EffectiveSignature::certify, py::const_),
           py::arg("persistence"))
      .def("certify",
           py::overload_cast<const chainhodge::CovariantChainHodge &, double, double, double>(
               &EffectiveSignature::certify, py::const_),
           py::arg("operator"), py::arg("epsilon"), py::arg("tolerance") = 1e-10,
           py::arg("minimum_gap") = kDefaultMinimumGap);
  py::class_<EffectiveTorus, EffectiveSignature>(m, "EffectiveTorus",
      "The effective d-torus: beta_k^eff = binomial(d, k).")
      .def(py::init<int>(), py::arg("dimension"));
  py::class_<EffectiveSphere, EffectiveSignature>(m, "EffectiveSphere",
      "The effective d-sphere: beta_0^eff = beta_d^eff = 1 and zero between.")
      .def(py::init<int>(), py::arg("dimension"));
  py::class_<EffectiveComponents, EffectiveSignature>(m, "EffectiveComponents",
      "n effective components: beta_0^eff = n, with no requirement on the higher degrees.")
      .def(py::init<int, int>(), py::arg("count"), py::arg("dimension"));
}
