// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
#define TESSERA_COBORDISM_LEVENBERGMARQUARDT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include <Eigen/Dense>

namespace tessera::cobordism {

/// A bounded least-squares Levenberg-Marquardt solver with reproducible random
/// restarts. Minimizes \f$ \|r(x)\|^2 \f$ over parameters kept inside a
/// caller-supplied feasible set.
///
/// Reference: Marquardt, "An algorithm for least-squares estimation of
/// nonlinear parameters", 1963.
class LevenbergMarquardt {
  public:
    /// The residual vector \f$ r(x) \f$ whose squared norm is the cost.
    using Residual = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    /// Projects a parameter vector back into the feasible set. Applied to the
    /// start point and to every trial step and finite-difference probe.
    using Clamp = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    /// Draws a random start point from a seeded engine.
    using Sample = std::function<Eigen::VectorXd(std::mt19937_64 &)>;

    /// The best point found and its cost.
    struct Result {
      /// The parameter vector, already clamped to the feasible set.
      Eigen::VectorXd parameters{};
      /// \f$ \|r(\text{parameters})\|^2 \f$; infinite until a point is
      /// evaluated.
      double cost{std::numeric_limits<double>::infinity()};
    };

    /// \param maxIterations Cap on Levenberg-Marquardt iterations per solve.
    /// \param epsilon Cost target; iteration stops once the cost is at or below
    ///   it.
    explicit LevenbergMarquardt(int maxIterations = 200,
                                double epsilon = 0.0);

    /// Minimize \f$ \|r(x)\|^2 \f$ from the start point \a x0. The Jacobian is
    /// central finite differences through \a clamp, and each iteration damps the
    /// normal equations until a step lowers the cost. Stops at the iteration
    /// cap, at the cost target, or when no damped step improves.
    [[nodiscard]] Result minimize(const Residual &residual, const Clamp &clamp,
                                  Eigen::VectorXd x0) const;

    /// `minimize` from \a restarts random start points drawn by \a sample from
    /// an engine seeded with \a seed, so a run is reproducible. Returns the
    /// lowest-cost result, stopping early once a cost below \a epsilon is
    /// reached. With \a numParams zero the empty parameter vector is returned
    /// with its cost. At least one restart always runs.
    [[nodiscard]] Result multiRestart(
        const Residual &residual, const Clamp &clamp, const Sample &sample,
        std::size_t numParams, int restarts, std::uint64_t seed,
        double epsilon) const;

  private:
    int maxIterations_;
    double epsilon_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LEVENBERGMARQUARDT_H
