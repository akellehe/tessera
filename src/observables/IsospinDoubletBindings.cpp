// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the isospin-doublet detector (observables/IsospinDoublet.h),
// registered into `tessera.observables` by src/bindings.cpp. The types it
// shares with other observables (QuarkConditionRead, SpectralFiber,
// GeneralLinearTransportRead, LineageNumberRead) are bound in
// src/observables/Bindings.cpp.

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "observables/IsospinDoublet.h"

namespace py = pybind11;
using namespace tessera::observables;

void register_isospin_doublet(py::module_ m) {
  py::class_<IsospinDoubletConfig>(m, "IsospinDoubletConfig",
      "Thresholds of the isospin-doublet detector. Each selects which reads "
      "are certified, never which value is reported.")
      .def(py::init<>())
      .def_readwrite("grouping_tolerance", &IsospinDoubletConfig::groupingTolerance)
      .def_readwrite("min_relative_gap", &IsospinDoubletConfig::minRelativeGap)
      .def_readwrite("contour_nodes", &IsospinDoubletConfig::contourNodes)
      .def_readwrite("projector_tolerance", &IsospinDoubletConfig::projectorTolerance)
      .def_readwrite("invariance_tolerance", &IsospinDoubletConfig::invarianceTolerance)
      .def_readwrite("commutant_tolerance", &IsospinDoubletConfig::commutantTolerance)
      .def_readwrite("track_overlap_threshold",
                     &IsospinDoubletConfig::trackOverlapThreshold)
      .def_readwrite("min_frames", &IsospinDoubletConfig::minFrames)
      .def_readwrite("transport_leakage_tolerance",
                     &IsospinDoubletConfig::transportLeakageTolerance)
      .def_readwrite("intertwining_tolerance",
                     &IsospinDoubletConfig::intertwiningTolerance)
      .def_readwrite("condition_number_cap", &IsospinDoubletConfig::conditionNumberCap);

  py::class_<IsospinFrame>(m, "IsospinFrame",
      "One frame of a cluster's lifetime: the operator the fiber is read on and "
      "the transfer from the previous frame's cells (empty: the identity).")
      .def(py::init([](std::string label, Eigen::MatrixXcd op, Eigen::MatrixXcd transfer) {
             return IsospinFrame{std::move(label), std::move(op), std::move(transfer)};
           }),
           py::arg("label"), py::arg("operator"),
           py::arg("transfer_from_previous") = Eigen::MatrixXcd())
      .def_readwrite("label", &IsospinFrame::label)
      .def_readwrite("operator", &IsospinFrame::operatorMatrix)
      .def_readwrite("transfer_from_previous", &IsospinFrame::transferFromPrevious);

  py::class_<IsospinResolution>(m, "IsospinResolution",
      "A further resolution of the first frame: its operator, the prolongation "
      "from the first frame's cells, and its sheet, base-cell and symmetry "
      "declarations.")
      .def(py::init<>())
      .def_readwrite("label", &IsospinResolution::label)
      .def_readwrite("operator", &IsospinResolution::operatorMatrix)
      .def_readwrite("prolongation", &IsospinResolution::prolongation)
      .def_readwrite("sheet_of_cell", &IsospinResolution::sheetOfCell)
      .def_readwrite("base_cell_of_cell", &IsospinResolution::baseCellOfCell)
      .def_readwrite("symmetry", &IsospinResolution::symmetry);

  py::class_<IsospinDoubletDeclaration>(m, "IsospinDoubletDeclaration",
      "What the isospin-doublet detector reads: the operator's name, the cells, "
      "the certified sheeting, the symmetry action, the frames and "
      "resolutions, and the optional lineage, member-splitting operator and "
      "three-quark one-body density.")
      .def(py::init<>())
      .def_readwrite("operator_name", &IsospinDoubletDeclaration::operatorName)
      .def_readwrite("degree", &IsospinDoubletDeclaration::degree)
      .def_readwrite("cells", &IsospinDoubletDeclaration::cells)
      .def_readwrite("sheet_of_cell", &IsospinDoubletDeclaration::sheetOfCell)
      .def_readwrite("base_cell_of_cell", &IsospinDoubletDeclaration::baseCellOfCell)
      .def_readwrite("symmetry", &IsospinDoubletDeclaration::symmetry)
      .def_readwrite("symmetry_name", &IsospinDoubletDeclaration::symmetryName)
      .def_readwrite("spinorial", &IsospinDoubletDeclaration::spinorial)
      .def_readwrite("frames", &IsospinDoubletDeclaration::frames)
      .def_readwrite("resolutions", &IsospinDoubletDeclaration::resolutions)
      .def_readwrite("member_splitting", &IsospinDoubletDeclaration::memberSplitting)
      .def_readwrite("lineage", &IsospinDoubletDeclaration::lineage)
      .def_readwrite("three_quark_density", &IsospinDoubletDeclaration::threeQuarkDensity);

  py::class_<IsospinBandRead>(m, "IsospinBandRead",
      "One Riesz band with its representation content under the acting "
      "symmetry and sheet algebra.")
      .def_readonly("index", &IsospinBandRead::index)
      .def_readonly("eigenvalues", &IsospinBandRead::eigenvalues)
      .def_readonly("center", &IsospinBandRead::center)
      .def_readonly("rank", &IsospinBandRead::rank)
      .def_readonly("gap", &IsospinBandRead::gap)
      .def_readonly("contour_center", &IsospinBandRead::contourCenter)
      .def_readonly("contour_radius", &IsospinBandRead::contourRadius)
      .def_readonly("projector_residual", &IsospinBandRead::projectorResidual)
      .def_readonly("projector_norm", &IsospinBandRead::projectorNorm)
      .def_readonly("resolvent_max", &IsospinBandRead::resolventMax)
      .def_readonly("isolated", &IsospinBandRead::isolated)
      .def_readonly("sheet_count", &IsospinBandRead::sheetCount)
      .def_readonly("sheet_invariance_residual", &IsospinBandRead::sheetInvarianceResidual)
      .def_readonly("colour_acts", &IsospinBandRead::colourActs)
      .def_readonly("symmetry_declared", &IsospinBandRead::symmetryDeclared)
      .def_readonly("symmetry_invariance_residual",
                    &IsospinBandRead::symmetryInvarianceResidual)
      .def_readonly("symmetry_acts", &IsospinBandRead::symmetryActs)
      .def_readonly("commutant_dimension", &IsospinBandRead::commutantDimension)
      .def_readonly("isotype_count", &IsospinBandRead::isotypeCount)
      .def_readonly("irreducible_dimensions", &IsospinBandRead::irreducibleDimensions)
      .def_readonly("multiplicities", &IsospinBandRead::multiplicities)
      .def_readonly("content", &IsospinBandRead::content)
      .def_readonly("spin_doublet", &IsospinBandRead::spinDoublet)
      .def_readonly("doublet_candidate", &IsospinBandRead::doubletCandidate)
      .def_readonly("unexplained_multiplicity", &IsospinBandRead::unexplainedMultiplicity)
      .def_readonly("classification", &IsospinBandRead::classification)
      .def_readonly("fiber", &IsospinBandRead::fiber);

  py::class_<IsospinFrameRead>(m, "IsospinFrameRead", "The bands of one frame.")
      .def_readonly("label", &IsospinFrameRead::label)
      .def_readonly("spectrum", &IsospinFrameRead::spectrum)
      .def_readonly("bands", &IsospinFrameRead::bands);

  py::class_<IsospinTransportStep>(m, "IsospinTransportStep",
      "One frame-to-frame transport of a doublet candidate: the library "
      "transport, its intertwining residual against the rotation and colour "
      "actions, and the flavour factor's singular values.")
      .def_readonly("from_frame", &IsospinTransportStep::fromFrame)
      .def_readonly("to_frame", &IsospinTransportStep::toFrame)
      .def_readonly("transport", &IsospinTransportStep::transport)
      .def_readonly("intertwining_residual", &IsospinTransportStep::intertwiningResidual)
      .def_readonly("flavour_singular_values", &IsospinTransportStep::flavourSingularValues);

  py::class_<IsospinChargeRead>(m, "IsospinChargeRead",
      "Isospin, charge and occupation pattern of an observed doublet.")
      .def_readonly("member_source", &IsospinChargeRead::memberSource)
      .def_readonly("isospin", &IsospinChargeRead::isospin)
      .def_readonly("member_projectors", &IsospinChargeRead::memberProjectors)
      .def_readonly("baryon_number", &IsospinChargeRead::baryonNumber)
      .def_readonly("charges", &IsospinChargeRead::charges)
      .def_readonly("member_occupations", &IsospinChargeRead::memberOccupations)
      .def_readonly("occupation_pattern", &IsospinChargeRead::occupationPattern)
      .def_readonly("notes", &IsospinChargeRead::notes);

  py::class_<IsospinCandidateRead>(m, "IsospinCandidateRead",
      "The read of one flavour-doublet candidate: its track, its transports, "
      "the three conditions and, when observed, its isospin and charges.")
      .def_readonly("band_index", &IsospinCandidateRead::bandIndex)
      .def_readonly("tracked_bands", &IsospinCandidateRead::trackedBands)
      .def_readonly("min_track_overlap", &IsospinCandidateRead::minTrackOverlap)
      .def_readonly("resolution_found", &IsospinCandidateRead::resolutionFound)
      .def_readonly("resolution_overlap", &IsospinCandidateRead::resolutionOverlap)
      .def_readonly("transports", &IsospinCandidateRead::transports)
      .def_readonly("lifetime_singular_values", &IsospinCandidateRead::lifetimeSingularValues)
      .def_readonly("conditions", &IsospinCandidateRead::conditions)
      .def_readonly("observed", &IsospinCandidateRead::observed)
      .def_readonly("charges", &IsospinCandidateRead::charges);

  py::class_<IsospinDoubletRead>(m, "IsospinDoubletRead",
      "The whole isospin-doublet read: the bands of every frame and "
      "resolution, the candidates, the three conditions, falsifiers 8 and 10, "
      "and a summary.")
      .def_readonly("operator_name", &IsospinDoubletRead::operatorName)
      .def_readonly("symmetry_name", &IsospinDoubletRead::symmetryName)
      .def_readonly("frames", &IsospinDoubletRead::frames)
      .def_readonly("resolutions", &IsospinDoubletRead::resolutions)
      .def_readonly("candidates", &IsospinDoubletRead::candidates)
      .def_readonly("conditions", &IsospinDoubletRead::conditions)
      .def_readonly("doublet_observed", &IsospinDoubletRead::doubletObserved)
      .def_readonly("no_isospin_doublet", &IsospinDoubletRead::noIsospinDoublet)
      .def_readonly("unexplained_multiplicities",
                    &IsospinDoubletRead::unexplainedMultiplicities)
      .def_readonly("multiplicity_refinement_measured",
                    &IsospinDoubletRead::multiplicityRefinementMeasured)
      .def_readonly("summary", &IsospinDoubletRead::summary);

  py::class_<IsospinDoublet>(m, "IsospinDoublet",
      R"doc(The observation of flavour as the whitepaper (v16, Section 10)
prescribes it: an unlabeled two-dimensional spectral band that is neither the
sheet (colour) multiplicity nor a spin irreducible of the support's symmetry
group, read on the operators a caller supplies, with the three conditions
(emergence, coherent transport, agreement with the Ward flux) each reported as
Passed, Failed or NotEvaluable.  The third is not evaluable: Section 13.4 is
deferred.)doc")
      .def_property_readonly_static("kConditionCount",
                                    [](py::object) { return IsospinDoublet::kConditionCount; })
      .def_static("condition_names", &IsospinDoublet::conditionNames)
      .def_static("statement", &IsospinDoublet::statement, py::arg("number"))
      .def_static("bands", &IsospinDoublet::bands, py::arg("operator"),
                  py::arg("cells") = std::vector<std::vector<std::uint64_t>>{},
                  py::arg("sheet_of_cell") = std::vector<std::size_t>{},
                  py::arg("base_cell_of_cell") = std::vector<std::size_t>{},
                  py::arg("symmetry") = std::vector<Eigen::MatrixXcd>{},
                  py::arg("spinorial") = false, py::arg("degree") = 1,
                  py::arg("config") = IsospinDoubletConfig{},
                  "The bands of one operator with their representation content.")
      .def_static("observe", &IsospinDoublet::observe, py::arg("declaration"),
                  py::arg("config") = IsospinDoubletConfig{}, "The whole read.");
}
