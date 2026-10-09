// N-qubit density-matrix utilities: the reduced state on any subset of the
// qubits, and a random correlated mixed state.
//
// An n-qubit density matrix here is a 2^n x 2^n Hermitian, positive
// semidefinite, unit-trace matrix in the computational basis, with qubit 0
// the most significant bit of the row index. That is the ordering of the
// Kronecker product rho_0 (x) rho_1 (x) ... (x) rho_{n-1}.

#pragma once

#include <Eigen/Dense>

#include <random>
#include <vector>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

/// Reduced density matrix of an n-qubit state on the qubits in `keep`.
///
/// Every qubit not listed in `keep` is traced out. The kept qubits keep
/// their listed order, so keep = {i, j} returns the (qubit i (x) qubit j)
/// joint state and keep = {j, i} returns it with the two factors swapped.
/// An empty `keep` returns the 1 x 1 matrix holding Tr(rho).
///
/// @throws std::invalid_argument if n is not in [1, 30], rho is not
///   2^n x 2^n, or `keep` holds an index outside [0, n) or a repeated index.
[[nodiscard]] Eigen::MatrixXcd
partialTrace(const Eigen::MatrixXcd& rho, int n, const std::vector<int>& keep);

/// A random correlated mixed state on n qubits: rho = M M^dagger /
/// Tr(M M^dagger), where M is a 2^n x 2^n matrix whose entries have
/// independent standard normal real and imaginary parts (a Ginibre matrix).
/// The state is generic: every pair of qubits shares mutual information.
///
/// @throws std::invalid_argument if n is not in [1, 30].
[[nodiscard]] Eigen::MatrixXcd randomCorrelatedState(int n, std::mt19937& rng);

} // namespace tessera::quantum
