// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Spectral readings of the U(1) connection Laplacian: the first gap and the
/// dimension of its kernel.
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379

#include "observables/Spectral.h"

#include <cstddef>
#include <vector>

#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

double SpectralGap::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // The U(1) connection Laplacian D - A, not the Hodge Laplacian L_0. The
  // content of this observable is Aharonov-Bohm: the gap collapses at flux pi,
  // a property of the connection operator. L_0's gap is a different number and
  // carries no flux dependence. The connection operator is Hermitian, so its
  // eigenvalues are real and ascending and the first gap is lambda_1 - lambda_0;
  // connectionEigenvalues() is complex-typed for parity with the L_k family.
  const std::vector<std::complex<double>> evals =
      ::tessera::cobordism::HodgeLaplacian(spacetime).connectionEigenvalues();
  if (evals.size() < 2) return 0.0;
  return (evals[1] - evals[0]).real();
}

double HarmonicDimension::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  // The kernel dimension of the U(1) connection Laplacian, not of L_0. A
  // nonzero flux lifts this zero mode, which is the content of this observable;
  // dim ker L_0 is always b_0, reported by ChainComplex::bettiNumbers.
  return static_cast<double>(
      ::tessera::cobordism::HodgeLaplacian(spacetime)
          .connectionHarmonics().size());
}

}  // namespace tessera::observables
