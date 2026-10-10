// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: dual volume signs and sheet
// isomorphisms. One of the translation units that Bindings.cpp registers in
// order (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_observables_dual_volume_sheet(py::module_ &m) {
  // ========================================
  // DualVolumeSigns
  // ========================================
  py::class_<DualVolumeSigns::DimensionReport>(m, "DualVolumeDimensionReport",
      "Per-dimension counts from the diagonal discrete exterior calculus (DEC) Hodge star sign audit.")
      .def_readonly("dimension", &DualVolumeSigns::DimensionReport::dimension)
      .def_readonly("n_simplices", &DualVolumeSigns::DimensionReport::nSimplices)
      .def_readonly("n_negative_dual_volume",
                    &DualVolumeSigns::DimensionReport::nNegativeDualVolume)
      .def_readonly("n_degenerate_volume",
                    &DualVolumeSigns::DimensionReport::nDegenerateVolume)
      .def_readonly("n_circumcenter_outside",
                    &DualVolumeSigns::DimensionReport::nCircumcenterOutside)
      .def_readonly("n_negative_circumradius",
                    &DualVolumeSigns::DimensionReport::nNegativeCircumradius)
      .def_readonly("n_negative_star",
                    &DualVolumeSigns::DimensionReport::nNegativeStar)
      .def_readonly("n_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nAllSpacelike)
      .def_readonly("n_negative_star_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nNegativeStarAllSpacelike)
      .def_readonly("n_mixed_signature",
                    &DualVolumeSigns::DimensionReport::nMixedSignature)
      .def_readonly(
          "n_negative_star_mixed_signature",
          &DualVolumeSigns::DimensionReport::nNegativeStarMixedSignature)
      .def_readonly("min_star_ratio",
                    &DualVolumeSigns::DimensionReport::minStarRatio)
      .def_readonly("max_star_ratio",
                    &DualVolumeSigns::DimensionReport::maxStarRatio)
      .def_readonly("mean_star_ratio",
                    &DualVolumeSigns::DimensionReport::meanStarRatio);

  py::class_<DualVolumeSigns::Report>(m, "DualVolumeReport",
      "The full diagonal DEC Hodge star sign audit, one entry per simplex "
      "dimension.")
      .def_readonly("dimensions", &DualVolumeSigns::Report::dimensions)
      .def_readonly("n_simplices", &DualVolumeSigns::Report::nSimplices)
      .def_readonly("n_negative_star", &DualVolumeSigns::Report::nNegativeStar);

  py::class_<DualVolumeSigns, std::shared_ptr<DualVolumeSigns> >(
      m, "DualVolumeSigns",
      R"doc(Read-only audit of the sign of the diagonal discrete exterior
calculus (DEC) Hodge star.

The star assigns each k-simplex the scalar ratio |*sigma| / |sigma|: the
signed circumcentric dual cell content over the simplex's own signed content.
A gauge term discretised with DEC carries its whole metric dependence in that
ratio, so a negative entry costs positive-definiteness of the Hodge Laplacian
and breaks the sign structure a self-dual / anti-self-dual split of a
2-cochain relies on.

Two causes of a negative ratio are separated. A circumcenter outside its
simplex (a negative barycentric coordinate) is the Riemannian
well-centeredness violation and indicates badly shaped cells. A timelike
circumcenter displacement (negative signed circumradius squared) is reachable
only in Lorentzian signature and is expected rather than defective. Counts
are therefore broken out by all-spacelike versus mixed-signature cells.

Changes no geometry and enforces nothing.)doc")
      .def(py::init<double>(), py::arg("tolerance") = 1e-12)
      .def("analyze", &DualVolumeSigns::analyze, py::arg("spacetime"),
           "The full per-dimension audit.")
      .def("compute", &DualVolumeSigns::compute, py::arg("spacetime"),
           "Fraction of audited, non-degenerate simplices whose star ratio is "
           "negative. Zero means the diagonal star is positive everywhere.");

  // ==========================================================================
  // SheetedColor (#1195): colour as sheet multiplicity.  The k-sheeted
  // support, the exterior algebra of the sheet space, the attachment matrix
  // of the connecting simplices, and the transported determinant-wedge
  // singlet amplitude.  Pure reads over caller-supplied data; nothing enters
  // the emergence objective.
  // ==========================================================================
  py::class_<SheetIsomorphismRead>(m, "SheetIsomorphismRead",
      R"doc(The certificate that k supports really are k sheets of one base
complex: the measured disagreement between corresponding cells of different
sheets, kept separate for the geometry (the stored complex squared lengths)
and for the declared connection, because the two fail for different physical
reasons.)doc")
      .def(py::init<>())
      .def_readonly("sheet_count", &SheetIsomorphismRead::sheetCount)
      .def_readonly("base_cell_count", &SheetIsomorphismRead::baseCellCount)
      .def_readonly("squared_length_residual",
                    &SheetIsomorphismRead::squaredLengthResidual,
                    "max over cells and sheet pairs of the squared-length "
                    "disagreement.")
      .def_readonly("connection_residual",
                    &SheetIsomorphismRead::connectionResidual,
                    "max over cells and sheet pairs of the connection-value "
                    "disagreement.")
      .def_readonly("isomorphic", &SheetIsomorphismRead::isomorphic)
      .def_readonly("certificate", &SheetIsomorphismRead::certificate,
                    "AlgebraicallyExact / Static, grading the larger of the "
                    "two residuals against the declared tolerance.");

  py::class_<SheetSector>(m, "SheetSector",
      R"doc(One occupation sector of the exterior algebra of the sheet space:
its occupation number N, its dimension binomial(k, N), its fermion parity
(-1)^N, and the complex representation it carries, named for three sheets as
the whitepaper names it.)doc")
      .def(py::init<>())
      .def_readonly("occupation", &SheetSector::occupation)
      .def_readonly("dimension", &SheetSector::dimension)
      .def_readonly("fermion_parity", &SheetSector::fermionParity)
      .def_readonly("representation", &SheetSector::representation);

  py::class_<SheetedSupport>(m, "SheetedSupport",
      R"doc(A cluster support carried as k isomorphic copies -- sheets -- of
one base complex.  A sheet is more cells and nothing else: each edge still
carries its complex squared length, its connection value and one two-level
mode.  The class certifies that the copies agree, lifts a base operator h to
the free sheeted operator h (x) I_k, lifts a base Riesz band to the
colour-spin fibre E-bar (x) C^k, and builds the two commuting actions
I (x) g (sheet relabeling) and D (x) I_k (a base symmetry).  A sheeted mode
is stored at the flat index base * k + sheet.)doc")
      .def(py::init<std::size_t, std::size_t>(), py::arg("sheetCount"),
           py::arg("baseCellCount"))
      .def_property_readonly("sheet_count", &SheetedSupport::sheetCount)
      .def_property_readonly("base_cell_count",
                             &SheetedSupport::baseCellCount)
      .def_property_readonly("cell_count", &SheetedSupport::cellCount)
      .def("modeIndex", &SheetedSupport::modeIndex, py::arg("baseCell"),
           py::arg("sheet"),
           "The flat index base * k + sheet of one sheeted cell.")
      .def("certifyIsomorphism", &SheetedSupport::certifyIsomorphism,
           py::arg("sheetSquaredLengths"), py::arg("sheetConnections"),
           py::arg("tolerance") = 1e-12,
           "Certify that the per-sheet squared lengths and connection values "
           "agree across the sheets.  Empty connection vectors declare a "
           "support with no connection values to compare.")
      .def("freeOperator", &SheetedSupport::freeOperator,
           py::arg("baseOperator"),
           "The free sheeted operator h (x) I_k.")
      .def("liftBand", &SheetedSupport::liftBand, py::arg("baseBand"),
           "The colour-spin fibre E-bar (x) C^k of a base Riesz band.")
      .def("sheetFrameOperator", &SheetedSupport::sheetFrameOperator,
           py::arg("g"), "The sheet relabeling I (x) g.")
      .def("baseSymmetryOperator", &SheetedSupport::baseSymmetryOperator,
           py::arg("baseAction"), "A base symmetry action D (x) I_k.")
      .def("sheetCommutatorResidual",
           &SheetedSupport::sheetCommutatorResidual, py::arg("baseOperator"),
           py::arg("g"),
           "||[A (x) I_k, I (x) g]||_max -- zero up to rounding for every "
           "base operator and every frame.");

  py::class_<SheetFock>(m, "SheetFock",
      R"doc(The exterior algebra Lambda^bullet E of the sheet space E = C^k of
ONE base mode.  For three sheets the sectors are the scalar vacuum, the
fundamental E, the determinant-twisted dual det E (x) E-dual and the
determinant line, with fermion parities even, odd, even, odd.  The sector
projectors and canonical anticommutation-relation matrices are delegated to
tessera.quantum.ExteriorAlgebra, and at three sheets they agree with
ColorFiber's exactly.)doc")
      .def(py::init<std::size_t>(), py::arg("sheetCount"))
      .def_property_readonly("sheet_count", &SheetFock::sheetCount)
      .def_property_readonly("dimension", &SheetFock::dimension)
      .def("sectors", &SheetFock::sectors,
           "The k + 1 occupation sectors in ascending occupation order.")
      .def("sectorProjector", &SheetFock::sectorProjector,
           py::arg("occupation"))
      .def("exteriorCreation", &SheetFock::exteriorCreation, py::arg("sheet"))
      .def("contraction", &SheetFock::contraction, py::arg("sheet"))
      .def("sheetBilinear", &SheetFock::sheetBilinear, py::arg("i"),
           py::arg("j"),
           "The gl(E) bilinear E^i_j = epsilon_i iota^j.")
      .def("commutatorResidual", &SheetFock::commutatorResidual,
           "The worst deviation over the whole gl(k, C) commutator table.")
      .def("sectorAgreementResidual", &SheetFock::sectorAgreementResidual,
           "The worst deviation from ColorFiber's sector projectors "
           "(three sheets only).");

  py::class_<ConnectingSimplex>(m, "ConnectingSimplex",
      R"doc(One connecting simplex between two sheeted supports: the sheet of
A and the sheet of B it is attached to, and the weight it contributes to that
entry of the attachment matrix.  The default gluing rule attaches sheet to
sheet, making the attachment matrix diagonal and a lone interaction
colour-abelian; a cross-sheet attachment is the nonabelian part of the colour
transport.)doc")
      .def(py::init([](std::size_t sheetA, std::size_t sheetB,
                       std::complex<double> weight) {
             return ConnectingSimplex{sheetA, sheetB, weight};
           }),
           py::arg("sheetA"), py::arg("sheetB"),
           py::arg("weight") = std::complex<double>(1.0, 0.0))
      .def_readwrite("sheet_a", &ConnectingSimplex::sheetA)
      .def_readwrite("sheet_b", &ConnectingSimplex::sheetB)
      .def_readwrite("weight", &ConnectingSimplex::weight);

  py::class_<AttachmentRead>(m, "AttachmentRead",
      R"doc(The attachment matrix S_AB and the data that decide whether it is
usable as a colour transport: its determinant (canonical, never normalized or
cube-rooted), its conditioning, its smallest singular value -- the distance to
the rank-dropping configuration the determinant winding encircles -- and
whether every connecting simplex joined like-indexed sheets.)doc")
      .def(py::init<>())
      .def_readonly("matrix", &AttachmentRead::matrix)
      .def_readonly("determinant", &AttachmentRead::determinant)
      .def_readonly("conditioning", &AttachmentRead::conditioning)
      .def_readonly("min_singular_value", &AttachmentRead::minSingularValue)
      .def_readonly("sheet_diagonal", &AttachmentRead::sheetDiagonal)
      .def_readonly("simplex_count", &AttachmentRead::simplexCount)
      .def_readonly("certificate", &AttachmentRead::certificate);

  py::class_<HolonomyInvariants>(m, "HolonomyInvariants",
      R"doc(The conjugacy-invariant data of a closed colour holonomy: the
ordered product around the sequence, its power traces tr H^j, the
characteristic polynomial they determine through the Newton identities (in
ascending powers of lambda) and its determinant.)doc")
      .def(py::init<>())
      .def_readonly("holonomy", &HolonomyInvariants::holonomy)
      .def_readonly("power_traces", &HolonomyInvariants::powerTraces)
      .def_readonly("characteristic_polynomial",
                    &HolonomyInvariants::characteristicPolynomial)
      .def_readonly("determinant", &HolonomyInvariants::determinant)
      .def_readonly("link_count", &HolonomyInvariants::linkCount);

  py::class_<SheetAttachment>(m, "SheetAttachment",
      R"doc(The colour transport of the sheet convention: the attachment
matrix S_AB reconstructed from the connecting simplices, its frame law
S_AB -> g_A^-1 S_AB g_B, the factorized coupling block C-bar_AB (x) S_AB, the
composition of transport along a declared path and the closed holonomy.  The
retained datum is the GL(k, C) element with its determinant line: no polar
factor is taken and no cube root of the determinant is chosen.)doc")
      .def_static("attachmentMatrix", &SheetAttachment::attachmentMatrix,
                  py::arg("sheetCount"), py::arg("simplices"),
                  py::arg("fullRankTolerance") = 1e-12)
      .def_static("frameChanged", &SheetAttachment::frameChanged,
                  py::arg("attachment"), py::arg("frameA"), py::arg("frameB"),
                  "S_AB -> g_A^-1 S_AB g_B.")
      .def_static("couplingBlock", &SheetAttachment::couplingBlock,
                  py::arg("baseCoupling"), py::arg("attachment"),
                  "C_AB = C-bar_AB (x) S_AB.")
      .def_static("compose", &SheetAttachment::compose, py::arg("path"),
                  py::arg("sheetCount") = std::size_t{0},
                  "The path transport, first factor applied first.")
      .def_static("holonomy", &SheetAttachment::holonomy, py::arg("links"));

  py::class_<ColorSingletRead>(m, "ColorSingletRead",
      R"doc(The common-frame colour amplitude and the data around it.  Nothing
is normalized: the physical singlet condition is a nonzero, refinement-stable,
covariantly trivial determinant wedge, so the reported content is the complex
amplitude and whether it vanishes, not a modulus square driven to one.)doc")
      .def(py::init<>())
      .def_readonly("amplitude", &ColorSingletRead::amplitude,
                    "S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C).")
      .def_readonly("transported_wedge", &ColorSingletRead::transportedWedge,
                    "det of the transported representatives, before the "
                    "determinant-line trivialization is applied.")
      .def_readonly("transported_columns",
                    &ColorSingletRead::transportedColumns)
      .def_readonly("magnitude", &ColorSingletRead::magnitude)
      .def_readonly("nonvanishing", &ColorSingletRead::nonvanishing)
      .def_readonly("min_singular_value",
                    &ColorSingletRead::minSingularValue)
      .def_readonly("certificate", &ColorSingletRead::certificate);

  py::class_<ColorSinglet>(m, "ColorSinglet",
      R"doc(The singlet readout of the sheet convention: the three colour
representatives are transported to a common base cluster p and wedged, and the
dual determinant trivialization Omega_p is applied.  Invariant under all local
frame changes when Omega_p transforms dually, which frameCovarianceResidual
measures rather than asserts.)doc")
      .def_static("amplitude", &ColorSinglet::amplitude,
                  py::arg("trivialization"), py::arg("transports"),
                  py::arg("colors"), py::arg("tolerance") = 1e-12)
      .def_static("frameCovarianceResidual",
                  &ColorSinglet::frameCovarianceResidual,
                  py::arg("trivialization"), py::arg("transports"),
                  py::arg("colors"), py::arg("baseFrame"), py::arg("frames"),
                  "|S_ABC(frame-changed) - S_ABC(original)|.");
}
