// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/DenseReference.h"

#include <Eigen/Dense>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "cobordism/OccupationSpectra.h"

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;

Eigen::MatrixXcd toMatrix(const std::vector<cd> &flat, int dim,
                          const char *name) {
  if (dim < 0 || flat.size() != static_cast<std::size_t>(dim) *
                                    static_cast<std::size_t>(dim))
    throw std::invalid_argument(std::string(name) +
                                ": flat size does not match dimension");
  Eigen::MatrixXcd matrix(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      matrix(i, j) = flat[static_cast<std::size_t>(i) * dim + j];
  return matrix;
}

void sortAscendingReIm(std::vector<cd> &values) {
  std::sort(values.begin(), values.end(), [](const cd &x, const cd &y) {
    if (x.real() != y.real())
      return x.real() < y.real();
    return x.imag() < y.imag();
  });
}

} // namespace

DenseReference::DenseReference(int crossoverDimension) {
  setCrossoverDimension(crossoverDimension);
}

void DenseReference::setCrossoverDimension(int crossoverDimension) {
  if (crossoverDimension < 1)
    throw std::invalid_argument(
        "DenseReference: crossover dimension must be positive");
  crossover_ = crossoverDimension;
}

void DenseReference::requireBelowCrossover(int dim, const char *kernel) const {
  if (!belowCrossover(dim))
    throw std::length_error(
        std::string(kernel) +
        ": dimension at or above the dense crossover (" +
        std::to_string(crossover_) +
        "); the dense reference is a fixture kernel, use the structured path");
}

CertifiedVector DenseReference::solve(const std::vector<cd> &matrix, int dim,
                                      const std::vector<cd> &rhs,
                                      double tolerance) const {
  requireBelowCrossover(dim, "DenseReference::solve");
  const Eigen::MatrixXcd a = toMatrix(matrix, dim, "solve: matrix");
  if (rhs.size() != static_cast<std::size_t>(dim))
    throw std::invalid_argument("solve: rhs size does not match dimension");
  Eigen::VectorXcd b(dim);
  for (int i = 0; i < dim; ++i)
    b(i) = rhs[static_cast<std::size_t>(i)];

  const Eigen::PartialPivLU<Eigen::MatrixXcd> factorization(a);
  const Eigen::VectorXcd x = factorization.solve(b);
  const double scale = b.norm();
  const double residual =
      scale > 0.0 ? (a * x - b).norm() / scale : (a * x - b).norm();
  const double rcond = factorization.rcond();

  CertifiedVector result;
  result.values.assign(x.data(), x.data() + x.size());
  result.certificate = Certificate::structureExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      rcond > 0.0 ? 1.0 / rcond : std::numeric_limits<double>::infinity(),
      tolerance);
  return result;
}

CertifiedVector DenseReference::spectrum(const std::vector<cd> &matrix, int dim,
                                         bool selfAdjoint,
                                         double tolerance) const {
  requireBelowCrossover(dim, "DenseReference::spectrum");
  const Eigen::MatrixXcd a = toMatrix(matrix, dim, "spectrum: matrix");
  const double scale = std::max(a.norm(), 1e-300);

  CertifiedVector result;
  // The self-adjoint request is honoured only after Hermiticity is verified.
  const double hermitianDefect = (a - a.adjoint()).norm() / scale;
  if (selfAdjoint && hermitianDefect <= tolerance) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(a);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("spectrum: self-adjoint eigensolve failed");
    double residual = hermitianDefect;
    for (int i = 0; i < dim; ++i) {
      const Eigen::VectorXcd v = solver.eigenvectors().col(i);
      residual = std::max(residual,
                          (a * v - solver.eigenvalues()(i) * v).norm() / scale);
    }
    result.values.reserve(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i)
      result.values.emplace_back(solver.eigenvalues()(i), 0.0);
    // Ascending real order already; unitary eigenvectors: conditioning 1.
    result.certificate = Certificate::certifiedNumerical(
        CertificateDomain::Static, CertificateRegime::HermitianIndefinite,
        residual, 1.0, tolerance);
    return result;
  }

  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(a);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("spectrum: eigensolve failed");
  double residual = 0.0;
  for (int i = 0; i < dim; ++i) {
    const Eigen::VectorXcd v = solver.eigenvectors().col(i);
    residual = std::max(residual,
                        (a * v - solver.eigenvalues()(i) * v).norm() /
                            (scale * std::max(v.norm(), 1e-300)));
  }
  // Conditioning of the eigenbasis: kappa_2 estimated from the extreme
  // singular values of the eigenvector matrix.
  double conditioning = std::numeric_limits<double>::infinity();
  if (dim > 0) {
    const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(solver.eigenvectors());
    const double smallest = svd.singularValues()(dim - 1);
    if (smallest > 0.0)
      conditioning = svd.singularValues()(0) / smallest;
  } else {
    conditioning = 1.0;
  }
  result.values.assign(solver.eigenvalues().data(),
                       solver.eigenvalues().data() + dim);
  sortAscendingReIm(result.values);
  result.certificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      conditioning, tolerance);
  return result;
}

CertifiedVector DenseReference::fockSpectrum(const std::vector<cd> &oneParticle,
                                             int dim, int particles,
                                             bool selfAdjoint,
                                             double tolerance) const {
  CertifiedVector oneParticleSpectrum =
      spectrum(oneParticle, dim, selfAdjoint, tolerance);
  CertifiedVector result;
  result.values = OccupationSpectra::subsetSums(oneParticleSpectrum.values,
                                                particles);
  result.certificate = oneParticleSpectrum.certificate;
  return result;
}

} // namespace tessera::cobordism
