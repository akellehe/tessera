#include "quantum/DensityMatrix.hpp"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>

namespace tessera::quantum {

namespace {

void requireQubitCount(int n) {
    if (n < 1 || n > 30)
        throw std::invalid_argument(
            "DensityMatrix: the qubit count must be in [1, 30], got " +
            std::to_string(n));
}

} // namespace

Eigen::MatrixXcd partialTrace(const Eigen::MatrixXcd& rho, int n,
                              const std::vector<int>& keep) {
    requireQubitCount(n);
    const int dim = 1 << n;
    if (rho.rows() != dim || rho.cols() != dim)
        throw std::invalid_argument(
            "partialTrace: rho must be 2^n x 2^n for n = " + std::to_string(n));
    std::vector<bool> seen(n, false);
    for (int q : keep) {
        if (q < 0 || q >= n)
            throw std::invalid_argument(
                "partialTrace: kept qubit " + std::to_string(q) +
                " is outside [0, " + std::to_string(n) + ")");
        if (seen[q])
            throw std::invalid_argument(
                "partialTrace: kept qubit " + std::to_string(q) +
                " is listed twice");
        seen[q] = true;
    }

    std::vector<int> traced;
    for (int b = 0; b < n; ++b)
        if (!seen[b]) traced.push_back(b);
    const int k = static_cast<int>(keep.size());
    const int t = static_cast<int>(traced.size());
    const int dimK = 1 << k;
    const int dimT = 1 << t;
    // Row index of rho for kept value kv and traced value tv; keep[0] and
    // traced[0] are the most significant bits of kv and tv.
    auto fullIndex = [&](int kv, int tv) {
        int idx = 0;
        for (int i = 0; i < k; ++i)
            if (kv & (1 << (k - 1 - i))) idx |= 1 << (n - 1 - keep[i]);
        for (int i = 0; i < t; ++i)
            if (tv & (1 << (t - 1 - i))) idx |= 1 << (n - 1 - traced[i]);
        return idx;
    };
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(dimK, dimK);
    for (int r = 0; r < dimK; ++r)
        for (int c = 0; c < dimK; ++c)
            for (int tv = 0; tv < dimT; ++tv)
                out(r, c) += rho(fullIndex(r, tv), fullIndex(c, tv));
    return out;
}

Eigen::MatrixXcd randomCorrelatedState(int n, std::mt19937& rng) {
    requireQubitCount(n);
    const int dim = 1 << n;
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::MatrixXcd m(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            m(i, j) = std::complex<double>(g(rng), g(rng));
    Eigen::MatrixXcd rho = m * m.adjoint();
    return rho / rho.trace().real();
}

} // namespace tessera::quantum
