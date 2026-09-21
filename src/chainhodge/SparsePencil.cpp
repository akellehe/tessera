// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/SparsePencil.h"

#include <stdexcept>
#include <vector>

namespace tessera::chainhodge {

namespace {

using Triplets = std::vector<Eigen::Triplet<Complex>>;

void requireSquarePencil(const SparsePencil &p, const char *who) {
  if (p.A.rows() != p.A.cols() || p.M.rows() != p.M.cols() || p.A.rows() != p.M.rows())
    throw std::invalid_argument(std::string(who) +
                                ": a pencil's two matrices must be square of one size");
}

void append(Triplets &out, const SparseMatrix &block, Eigen::Index rowOffset, Eigen::Index colOffset) {
  for (int c = 0; c < block.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(block, c); it; ++it)
      out.emplace_back(static_cast<int>(it.row() + rowOffset), static_cast<int>(it.col() + colOffset),
                       it.value());
}

SparseMatrix assembled(const Triplets &entries, Eigen::Index n) {
  SparseMatrix out(n, n);
  out.setFromTriplets(entries.begin(), entries.end());
  out.makeCompressed();
  return out;
}

}  // namespace

SparsePencil SparsePencilComposition::directSum(const SparsePencil &a, const SparsePencil &b) {
  return hoppingBlock(a, b, SparseMatrix(a.A.rows(), b.A.rows()), SparseMatrix(b.A.rows(), a.A.rows()));
}

SparsePencil SparsePencilComposition::hoppingBlock(const SparsePencil &a, const SparsePencil &b,
                                                   const SparseMatrix &coupling,
                                                   const SparseMatrix &couplingReverse) {
  requireSquarePencil(a, "SparsePencilComposition");
  requireSquarePencil(b, "SparsePencilComposition");
  const Eigen::Index na = a.A.rows(), nb = b.A.rows();
  if (coupling.rows() != na || coupling.cols() != nb)
    throw std::invalid_argument("SparsePencilComposition::hoppingBlock: the coupling block must be n_a x n_b");
  const bool hermitian = couplingReverse.rows() == 0 && couplingReverse.cols() == 0;
  if (!hermitian && (couplingReverse.rows() != nb || couplingReverse.cols() != na))
    throw std::invalid_argument(
        "SparsePencilComposition::hoppingBlock: the reverse coupling block must be n_b x n_a");
  const SparseMatrix reverse = hermitian ? SparseMatrix(coupling.adjoint()) : couplingReverse;

  Triplets left, right;
  left.reserve(static_cast<std::size_t>(a.A.nonZeros() + b.A.nonZeros() + coupling.nonZeros() +
                                        reverse.nonZeros()));
  right.reserve(static_cast<std::size_t>(a.M.nonZeros() + b.M.nonZeros()));
  append(left, a.A, 0, 0);
  append(left, b.A, na, na);
  append(left, coupling, 0, na);
  append(left, reverse, na, 0);
  append(right, a.M, 0, 0);
  append(right, b.M, na, na);

  SparsePencil out;
  out.degree = a.degree;
  out.A = assembled(left, na + nb);
  out.M = assembled(right, na + nb);
  return out;
}

}  // namespace tessera::chainhodge
