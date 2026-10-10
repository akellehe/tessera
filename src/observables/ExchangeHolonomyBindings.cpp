// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: the exchange holonomy. One
// of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_exchange_holonomy(py::module_ &m) {
  // ==========================================================================
  // ExchangeHolonomy: Berry-cancelled exchange statistics, the
  // constructed total-space spin holonomy cycle, and the conditional
  // SO(d) -> Spin(d) lift.  Read-only; nothing enters any emergence
  // objective; no Kasteleyn orientation is required anywhere.
  // ==========================================================================
  py::class_<ExchangeHolonomyConfig>(m, "ExchangeHolonomyConfig",
      "Analysis parameters of the exchange and rotation holonomy reads. "
      "Thresholds select which reads are certified; a failed threshold "
      "yields an uncertified read, never a different "
      "sign.")
      .def(py::init<>())
      .def_readwrite("leakFloor", &ExchangeHolonomyConfig::leakFloor,
                     "Minimum overlap singular value of a certified step "
                     "(a leaking transfer is rejected before polar "
                     "normalization).")
      .def_readwrite("conditionCap", &ExchangeHolonomyConfig::conditionCap,
                     "Maximum overlap conditioning of a certified step.")
      .def_readwrite("unitaryTolerance",
                     &ExchangeHolonomyConfig::unitaryTolerance,
                     "Certificate tolerance on loop unitarity / character "
                     "modulus.")
      .def_readwrite("signTolerance",
                     &ExchangeHolonomyConfig::signTolerance,
                     "Distance from +-1 within which a definite "
                     "characterSign is reported.")
      .def_readwrite("blockMatchThreshold",
                     &ExchangeHolonomyConfig::blockMatchThreshold,
                     "Minimum subspace overlap of a certified block "
                     "continuation (mirrors the band-tracker threshold).")
      .def_readwrite("liftAngleMargin",
                     &ExchangeHolonomyConfig::liftAngleMargin,
                     "Lifted loop steps must stay this far below the pi "
                     "branch cut.")
      .def_readwrite("cocycleTolerance",
                     &ExchangeHolonomyConfig::cocycleTolerance,
                     "Cap on the verified SO(d) cocycle residual of "
                     "spinLift.");

  py::enum_<HolonomyChannel>(m, "HolonomyChannel",
      "Which physical question a Berry-cancelled character answers; "
      "particle exchange and physical rotation are separate channels by "
      "construction and doublyCancelledRatio refuses mislabeled inputs.")
      .value("ParticleExchange", HolonomyChannel::ParticleExchange)
      .value("PhysicalRotation", HolonomyChannel::PhysicalRotation);

  py::class_<TransportStepRead>(m, "TransportStepRead",
      "One overlap-transport step: singular-value data of the r x r frame "
      "overlap before polar normalization, and whether the step met the "
      "leak/conditioning thresholds.")
      .def_readonly("fromIndex", &TransportStepRead::fromIndex)
      .def_readonly("toIndex", &TransportStepRead::toIndex)
      .def_readonly("minSingularValue", &TransportStepRead::minSingularValue)
      .def_readonly("maxSingularValue", &TransportStepRead::maxSingularValue)
      .def_readonly("conditioning", &TransportStepRead::conditioning)
      .def_readonly("certified", &TransportStepRead::certified);

  py::class_<LoopHolonomyRead>(m, "LoopHolonomyRead",
      R"doc(Certified cyclic overlap transport of one tracked frame around a
closed loop: R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t),
U_gamma = R_{T-1} ... R_0.  `determinant` is the raw chi_raw = det U_gamma;
it contains the ordinary Berry phase of the reference motion and is never
an exchange sign by itself.  Only the interferometric ratio against a
matched reference loop is the dynamical certificate.  An uncertified band
on the loop (gap closure), a leak, or ill-conditioning yields an
uncertified read, never a sign.)doc")
      .def_readonly("holonomy", &LoopHolonomyRead::holonomy)
      .def_readonly("determinant", &LoopHolonomyRead::determinant)
      .def_readonly("steps", &LoopHolonomyRead::steps)
      .def_readonly("rank", &LoopHolonomyRead::rank)
      .def_readonly("stepReads", &LoopHolonomyRead::stepReads)
      .def_readonly("unitarityResidual",
                    &LoopHolonomyRead::unitarityResidual)
      .def_readonly("minStepSingularValue",
                    &LoopHolonomyRead::minStepSingularValue)
      .def_readonly("conditioning", &LoopHolonomyRead::conditioning)
      .def_readonly("uncertifiedBand", &LoopHolonomyRead::uncertifiedBand)
      .def_readonly("certificate", &LoopHolonomyRead::certificate);

  py::class_<HolonomyCharacterRead>(m, "HolonomyCharacterRead",
      R"doc(The interferometric (Berry-cancelled) character chi_hat =
det U_loop / det U_reference, with the phase channels kept separate:
rawLoopDeterminant (exchange or rotation, plus Berry), referenceDeterminant
(the Berry reference motion alone), character (the cancelled ratio).
characterSign is -1/+1 only when the certificate holds and the character
sits within signTolerance of -+1; an uncertified read never emits a
sign.)doc")
      .def_readonly("channel", &HolonomyCharacterRead::channel)
      .def_readonly("rawLoopDeterminant",
                    &HolonomyCharacterRead::rawLoopDeterminant)
      .def_readonly("referenceDeterminant",
                    &HolonomyCharacterRead::referenceDeterminant)
      .def_readonly("character", &HolonomyCharacterRead::character)
      .def_readonly("characterSign", &HolonomyCharacterRead::characterSign)
      .def_readonly("signResidual", &HolonomyCharacterRead::signResidual)
      .def_readonly("timingMatched", &HolonomyCharacterRead::timingMatched)
      .def_readonly("ranksMatched", &HolonomyCharacterRead::ranksMatched)
      .def_readonly("certificate", &HolonomyCharacterRead::certificate);

  py::class_<ClusterOccupancy>(m, "ClusterOccupancy",
      R"doc(The occupancy declaration of one tracked cluster block: how many
one-particle modes of the block's fibre the state occupies, and how many
sheets the block's support carries.  The exchange statistic is occupation
parity, so the occupation is the number that enters it; the sheet count
decides whether the rank-parity cross-check applies at all, because a
sheeted fibre has even rank and exchanging whole frames of even rank gives
+1 whatever the occupations are.)doc")
      .def(py::init([](std::size_t occupation, std::size_t sheetCount) {
             return ClusterOccupancy{occupation, sheetCount};
           }),
           py::arg("occupation") = std::size_t{1},
           py::arg("sheetCount") = std::size_t{1})
      .def_readwrite("occupation", &ClusterOccupancy::occupation)
      .def_readwrite("sheet_count", &ClusterOccupancy::sheetCount);

  py::class_<BlockPermutationRead>(m, "BlockPermutationRead",
      R"doc(The structural exchange channel: the permutation of persistent
localized blocks around the loop (matching delegated to
SpectralFiberTracker.matchFibers), its exact parities through the exterior
grading, and the residual in-block motion after reference cancellation.

occupationParity is the exchange statistic: the graded sign the exterior
Fock functor attaches to the reordering, computed from the declared
occupations.  rankParity is the independent odd-rank determinant
cross-check computed the same way from the fibre ranks; it is retired, and
reported as 0 with rankParityRetired set, on a sheeted support, and it is
never multiplied into the statistic.  blockParity is the block-label sign
and compositeParity the optional composite-level sign.

Parities are exact integers given the verified matching premise; a failed
premise (gap closure, rank change, ambiguous matching) yields an
uncertified read with no parities.)doc")
      .def_readonly("blockPermutation",
                    &BlockPermutationRead::blockPermutation)
      .def_readonly("blockRanks", &BlockPermutationRead::blockRanks)
      .def_readonly("blockOccupations",
                    &BlockPermutationRead::blockOccupations)
      .def_readonly("blockSheetCounts",
                    &BlockPermutationRead::blockSheetCounts)
      .def_readonly("blockParity", &BlockPermutationRead::blockParity)
      .def_readonly("occupationParity",
                    &BlockPermutationRead::occupationParity,
                    "The exchange statistic: the graded sign of the "
                    "reordering read off the declared occupations.")
      .def_readonly("rankParity", &BlockPermutationRead::rankParity,
                    "The odd-rank determinant cross-check, reported "
                    "independently and never multiplied into the statistic.")
      .def_readonly("rankParityRetired",
                    &BlockPermutationRead::rankParityRetired,
                    "Whether the cross-check was retired, as it is on a "
                    "sheeted support.")
      .def_readonly("rankParityAgrees",
                    &BlockPermutationRead::rankParityAgrees)
      .def_readonly("compositePermutation",
                    &BlockPermutationRead::compositePermutation)
      .def_readonly("compositeParity",
                    &BlockPermutationRead::compositeParity)
      .def_readonly("minMatchOverlap",
                    &BlockPermutationRead::minMatchOverlap)
      .def_readonly("residualInBlockMotion",
                    &BlockPermutationRead::residualInBlockMotion)
      .def_readonly("certificate", &BlockPermutationRead::certificate);

  py::class_<LoopLiftRead>(m, "LoopLiftRead",
      "The Z2 character of a closed SO(d) loop lifted step-by-step to "
      "Spin(d): +1 contractible, -1 the double-cover generator; 0 "
      "(uncertified) when a step approached the pi branch cut or the "
      "lifted product failed to close on +-I.")
      .def_readonly("character", &LoopLiftRead::character)
      .def_readonly("maxStepAngle", &LoopLiftRead::maxStepAngle)
      .def_readonly("closureResidual", &LoopLiftRead::closureResidual)
      .def_readonly("certificate", &LoopLiftRead::certificate);

  py::class_<SpinLiftRead>(m, "SpinLiftRead",
      R"doc(The SO(d) -> Spin(d) lift decision over Cech transition data
with the second Stiefel-Whitney obstruction: per-triangle lift signs, the
exact GF(2) coboundary decision, and (when the lift exists) a consistent
per-edge sign choice.  Needed only for a continuum claim: the abstract
CAR/Fock algebra needs no spin structure and no Kasteleyn orientation.)doc")
      .def_readonly("liftExists", &SpinLiftRead::liftExists)
      .def_readonly("obstructed", &SpinLiftRead::obstructed)
      .def_readonly("triangleSigns", &SpinLiftRead::triangleSigns)
      .def_readonly("edgeSigns", &SpinLiftRead::edgeSigns)
      .def_readonly("maxCocycleResidual",
                    &SpinLiftRead::maxCocycleResidual)
      .def_readonly("maxLiftResidual", &SpinLiftRead::maxLiftResidual)
      .def_readonly("certificate", &SpinLiftRead::certificate)
      .def("describe", &SpinLiftRead::describe)
      .def("__repr__", &SpinLiftRead::describe);

  py::class_<ExchangeHolonomy>(m, "ExchangeHolonomy",
      R"doc(Berry-cancelled exchange statistics, the constructed total-space
spin holonomy cycle, and the conditional SO(d) -> Spin(d) lift.

Identities: certified cyclic overlap transport
R_t = polar(Phi_{t+1}^dagger W_t Phi_t), U_gamma = R_{T-1}...R_0 over
SpectralFiber frames; the interferometric exchange character
chi_hat = det U_exchange / det U_reference (the raw determinant contains
Berry phase and is never the sign); the structural block permutation with
exact graded parities; the total-space spin holonomy cycle with its
co-moving reference (one global rotation of the whole carried frame, never
per-hole Bloch products); the total-space J^2 reference values (proton
eigenstate -> 3/4, Delta -> 15/4); the principal rotation logarithm, the
Spin(d) lift, the Z2 loop character, and the w2 obstruction over Cech data.

Channels kept separate in API and report: simplex reorientation
(reorientedFrames, exactly invariant), compilation ordering
(permutedCellFrames / cell-tuple matching, exactly invariant), particle
exchange, Berry reference motion, physical rotation.

Read-only and stateless: never calls a solver, never mutates what it
reads, and nothing here may enter any emergence objective.)doc")
      .def_static("polarUnitary", &ExchangeHolonomy::polarUnitary,
                  py::arg("overlap"),
                  "The unitary polar factor U V^dagger of an overlap "
                  "matrix (the normative transport primitive).")
      .def_static("loopHolonomy", &ExchangeHolonomy::loopHolonomy,
                  py::arg("frames"), py::arg("weights"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy of an explicit frame path under a "
                  "constant diagonal metric.")
      .def_static("loopHolonomyPerStep",
                  &ExchangeHolonomy::loopHolonomyPerStep,
                  py::arg("frames"), py::arg("stepWeights"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy with per-step diagonal metrics "
                  "W_t.")
      .def_static("fiberLoopHolonomy",
                  &ExchangeHolonomy::fiberLoopHolonomy, py::arg("loop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy of a spectral-fiber track (shared "
                  "cells matched by vertex tuple; an uncertified band or "
                  "rank change yields an uncertified read).")
      .def_static("exchangeCharacter",
                  &ExchangeHolonomy::exchangeCharacter,
                  py::arg("exchangeLoop"), py::arg("referenceLoop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "chi_hat_F = det U_exchange / det U_reference "
                  "(ParticleExchange channel).")
      .def_static("rotationCharacter",
                  &ExchangeHolonomy::rotationCharacter,
                  py::arg("rotationLoop"), py::arg("referenceLoop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "chi_hat(2 pi) against the matched co-moving "
                  "non-rotating reference (PhysicalRotation channel).")
      .def_static("doublyCancelledRatio",
                  &ExchangeHolonomy::doublyCancelledRatio,
                  py::arg("exchange"), py::arg("rotation"),
                  "chi_hat(exchange) * chi_hat(2 pi)^{-1}; requires the "
                  "correct channel tags (ValueError otherwise).")
      .def_static("blockPermutation", &ExchangeHolonomy::blockPermutation,
                  py::arg("steps"),
                  py::arg("referenceSteps") =
                      std::vector<std::vector<SpectralFiber>>{},
                  py::arg("composites") =
                      std::vector<std::vector<std::size_t>>{},
                  py::arg("occupancies") = std::vector<ClusterOccupancy>{},
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Structural block tracking around the loop: permutation, "
                  "the exchange statistic from the declared occupations, the "
                  "rank-parity cross-check, and the reference-cancelled "
                  "in-block residual.  An empty occupancy list declares one "
                  "occupied mode on an unsheeted support for every block.")
      .def_static("frameExchangeDeterminant",
                  &ExchangeHolonomy::frameExchangeDeterminant,
                  py::arg("rankA"), py::arg("rankB"),
                  "det pi_AB = (-1)^{r_A r_B}, the determinant of exchanging "
                  "two complete fibre frames.  Exact as an identity about "
                  "frames; promoting it to particle statistics is the "
                  "hypothesis the construction does not adopt.")
      .def_static("spinorDimension", &ExchangeHolonomy::spinorDimension,
                  py::arg("d"))
      .def_static("gamma", &ExchangeHolonomy::gamma, py::arg("a"),
                  py::arg("d"),
                  "Euclidean gamma_a with {gamma_a, gamma_b} = 2 delta_ab "
                  "(Pauli at d = 3; the documented Dirac layer at d = 4).")
      .def_static("spinGenerator", &ExchangeHolonomy::spinGenerator,
                  py::arg("a"), py::arg("b"), py::arg("d"),
                  "Sigma_ab = [gamma_a, gamma_b]/4, eigenvalues -+i/2.")
      .def_static("spinorRotation", &ExchangeHolonomy::spinorRotation,
                  py::arg("theta"), py::arg("a"), py::arg("b"),
                  py::arg("d"),
                  "exp(theta Sigma_ab) in closed form; theta = 2 pi gives "
                  "exactly -I (the double cover).")
      .def_static("transverseSpinorFrame",
                  &ExchangeHolonomy::transverseSpinorFrame, py::arg("a"),
                  py::arg("b"), py::arg("d"),
                  "The canonical transverse rank-1 spinor frame of the "
                  "(a, b) plane (deterministic conventions).")
      .def_static("rotationLoopFrames",
                  &ExchangeHolonomy::rotationLoopFrames, py::arg("frame0"),
                  py::arg("a"), py::arg("b"), py::arg("d"),
                  py::arg("turns"), py::arg("steps"),
                  "The constructed total-space spin holonomy cycle as an "
                  "explicit closed frame path (one global rotation of the "
                  "whole carried frame).")
      .def_static("referenceLoopFrames",
                  &ExchangeHolonomy::referenceLoopFrames, py::arg("frame0"),
                  py::arg("steps"),
                  "The matched co-moving non-rotating reference (same "
                  "timing, no rotation).")
      .def_static("vectorLoopFrames", &ExchangeHolonomy::vectorLoopFrames,
                  py::arg("frame0"), py::arg("a"), py::arg("b"),
                  py::arg("d"), py::arg("turns"), py::arg("steps"),
                  "The vector-representation rotation loop (the +1 "
                  "control).")
      .def_static("totalJSquaredOperator",
                  &ExchangeHolonomy::totalJSquaredOperator,
                  py::arg("constituents"),
                  "J^2 = sum_a (sum_i S_a^(i))^2 on (C^2)^(tensor n) — the "
                  "total-space operator on the whole composite state.")
      .def_static("totalJSquared", &ExchangeHolonomy::totalJSquared,
                  py::arg("state"),
                  "<J^2> of a composite state (exact oracles: proton "
                  "eigenstate 3/4, Delta 15/4, product |uud> 7/4).")
      .def_static("rotationLog", &ExchangeHolonomy::rotationLog,
                  py::arg("rotation"),
                  "The principal antisymmetric logarithm via the real "
                  "Schur plane decomposition (pi branch by the documented "
                  "axis rule).")
      .def_static("rotationToSpin", &ExchangeHolonomy::rotationToSpin,
                  py::arg("rotation"), py::arg("d"),
                  "The principal Spin(d) lift of an SO(d) rotation "
                  "(half-angle plane factors; d = 3, 4).")
      .def_static("loopLiftCharacter",
                  &ExchangeHolonomy::loopLiftCharacter, py::arg("loop"),
                  py::arg("d"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "The Z2 character of a closed SO(d) loop by incremental "
                  "principal lifts (uncertified near the pi branch cut).")
      .def_static("spinLift", &ExchangeHolonomy::spinLift,
                  py::arg("edges"), py::arg("edgeRotations"),
                  py::arg("triangles"), py::arg("d"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "The SO(d) -> Spin(d) lift decision over Cech data with "
                  "the w2 obstruction (exact GF(2) coboundary decision "
                  "given the verified cocycle premise).")
      .def_static("reorientedFrames", &ExchangeHolonomy::reorientedFrames,
                  py::arg("frames"), py::arg("cellSigns"),
                  "The simplex-reorientation gauge (common row sign "
                  "flips); every read is exactly invariant.")
      .def_static("permutedCellFrames",
                  &ExchangeHolonomy::permutedCellFrames, py::arg("frames"),
                  py::arg("rowPermutation"),
                  "The compilation-ordering gauge (common row "
                  "permutation); every read is exactly invariant.");
}
