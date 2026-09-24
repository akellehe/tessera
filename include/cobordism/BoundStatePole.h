// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_BOUNDSTATEPOLE_H
#define TESSERA_COBORDISM_BOUNDSTATEPOLE_H

#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/PencilSchur.h"
#include "cobordism/PencilLayer.h"

namespace tessera::cobordism {

/// # BoundStatePoleConfig
///
/// Every declared parameter of the pole search, echoed on each read so a
/// stored result carries the configuration that produced it.
struct BoundStatePoleConfig {
  /// Quadrature nodes on the contour. The trapezoid rule on a circle is
  /// spectrally accurate for a function holomorphic in an annulus around it,
  /// which \f$ D_C'/D_C \f$ is once the contour separates the enclosed zeros
  /// from every other zero and from every interior pole.
  int contourNodes = 128;
  /// Nodes of the second quadrature the refinement continuation is read at.
  /// The pole is recomputed with this node count and the movement between the
  /// two answers is reported as the continuation's stability.
  int refinementNodes = 256;
  /// The largest enclosed zero count the moment method is run for. A contour
  /// enclosing more zeros than this is refused by name rather than answered
  /// with a truncated root set.
  std::size_t maxZeros = 16;
  /// Newton steps on \f$ D_C \f$ allowed per root.
  int maxNewtonSteps = 64;
  /// A Newton step smaller than this, relative to the contour radius, ends the
  /// refinement of a root.
  double newtonTolerance = 1e-13;
  /// \f$ |N-\operatorname{round}(N)| \f$ of the argument-principle count must
  /// be at or below this for the count to be read as an integer. Above it the
  /// contour is too close to a zero or to an interior pole for the quadrature
  /// to resolve, and the read refuses by name.
  double zeroCountTolerance = 1e-3;
  /// The radius, as a fraction of the distance to the nearest other root, of
  /// the small contour a root's algebraic multiplicity and residue are read
  /// on. With a single root the fraction is taken of the main contour radius.
  double localRadiusFraction = 0.25;
  /// Singular values at or below this fraction of the largest are treated as
  /// zero when the Hankel rank and the residue rank are counted.
  double rankTolerance = 1e-10;
  /// The free threshold the binding shift is measured against: the complex
  /// spectral value at which the cluster's content is unbound. Empty leaves
  /// the binding shift unreported, since no threshold is derivable from the
  /// pencil alone.
  std::optional<std::complex<double>> freeThreshold{};
};

/// # BoundStatePoleRead
///
/// The zeros of \f$ D_C(s)=\det F_C(s) \f$ inside one declared contour, with
/// the certificates Section 13.3 attaches to a bound-state pole.
struct BoundStatePoleRead {
  /// The centre of the contour the search was run on.
  std::complex<double> centre{0.0, 0.0};
  /// Its radius.
  double radius = std::numeric_limits<double>::quiet_NaN();
  /// Its quadrature node count.
  int nodes = 0;

  /// \f$ (2\pi i)^{-1}\oint D_C'/D_C\,ds \f$ as the quadrature produced it:
  /// complex, and reported before it is rounded, so a contour that does not
  /// resolve its zeros shows it.
  std::complex<double> zeroCount{0.0, 0.0};
  /// The integer nearest `zeroCount`: the total algebraic multiplicity
  /// enclosed.
  std::size_t zeros = 0;
  /// \f$ |{\rm zeroCount}-{\rm zeros}| \f$.
  double zeroCountDefect = std::numeric_limits<double>::quiet_NaN();

  /// \f$ (2\pi i)^{-1}\oint \frac{d}{ds}\log\det P_{II}\,ds \f$ as the
  /// quadrature produced it: the unretained interior poles of \f$ F_C \f$
  /// inside the contour.
  std::complex<double> interiorPoleCount{0.0, 0.0};
  /// The integer nearest `interiorPoleCount`, floored at zero.
  std::size_t interiorPolesEnclosed = 0;

  /// The distinct zeros \f$ s_C \f$ found inside the contour, in the order the
  /// moment problem produced them. No ordering convention is imposed on them.
  /// Every reported zero lies strictly inside the declared contour: a zero
  /// the refinement would place outside it is not reported, and the read
  /// names "pole-outside-contour". A zero of multiplicity above one is the
  /// mean of the zeros its own small contour encloses (the first moment of
  /// the argument principle there over the count), which is the zero itself
  /// when it is exactly multiple and the centroid of a cluster the
  /// quadrature does not resolve.
  std::vector<std::complex<double>> poles{};
  /// The algebraic multiplicity of each zero, read as the argument-principle
  /// count on a small contour around it. Parallel to `poles`.
  std::vector<std::size_t> multiplicity{};
  /// \f$ D_C(s_C) \f$ at the refined root, parallel to `poles`. The simple
  /// isolated zero of Section 13.3 is specified by this vanishing.
  ///
  /// It is read as the Taylor coefficient
  /// \f$ (2\pi i)^{-1}\oint D_C(s)/(s-s_C)\,ds \f$ on the same small contour
  /// the residue is taken on, rather than as \f$ D_C \f$ evaluated at a point
  /// where the response is singular.
  std::vector<std::complex<double>> determinantAtPole{};
  /// \f$ D_C'(s_C) \f$, parallel to `poles`, read as the Taylor coefficient
  /// \f$ (2\pi i)^{-1}\oint D_C(s)/(s-s_C)^2\,ds \f$. The second half of the
  /// specification is that this does not vanish.
  std::vector<std::complex<double>> derivativeAtPole{};
  /// Whether the zero met the simple-isolated specification: algebraic
  /// multiplicity one and a nonvanishing derivative. For a zero of a
  /// holomorphic function those two are the same statement, and the
  /// multiplicity is the one the argument principle counted.
  std::vector<bool> simple{};
  /// The last Newton step taken on each root, in absolute units: the
  /// convergence of the refinement. NaN for a zero of multiplicity above
  /// one, which is read as its local mean and not refined by Newton, and for
  /// a simple zero whose Newton iterate left its small contour.
  std::vector<double> newtonStep{};
  /// The distance from each root to the nearest other root found inside the
  /// contour; infinite when it is the only one.
  std::vector<double> separation{};

  /// The residue \f$ (2\pi i)^{-1}\oint F_C(s)^{-1}ds \f$ of the supported
  /// resolvent at each root, flat row-major over the interface coordinates.
  /// Parallel to `poles`. This is the pole residue the whitepaper reports
  /// beside the pole, and it is a matrix rather than a number so that a
  /// multiple root's local data is retained rather than summarized.
  std::vector<std::vector<std::complex<double>>> residue{};
  /// The Frobenius norm of each residue.
  std::vector<double> residueNorm{};
  /// The numerical rank of each residue at `rankTolerance`, which for a simple
  /// isolated zero of a generic pencil is one.
  std::vector<std::size_t> residueRank{};

  /// Each root recomputed with `refinementNodes` quadrature nodes: the
  /// refinement continuation. Parallel to `poles`.
  std::vector<std::complex<double>> continuedPole{};
  /// \f$ |s_C^{\rm refined}-s_C| \f$, the movement under that continuation.
  std::vector<double> continuationMovement{};

  /// \f$ s_C-s_{\rm free} \f$ against the declared free threshold, parallel to
  /// `poles`; empty when no threshold is declared. The whitepaper reports the
  /// binding shift below the free threshold beside the pole and takes no
  /// square root of either.
  std::vector<std::complex<double>> bindingShift{};

  /// Whether the interior block \f$ P_{II}(s) \f$ was singular at some node of
  /// the contour, i.e. an unretained interior pole sits on it. The domain
  /// Section 13.3 continues \f$ F_C \f$ on excludes unretained interior poles,
  /// so this is a statement that the declared contour leaves that domain.
  bool interiorResonance = false;

  /// Named failures: "empty-interface", "nonintegral-zero-count",
  /// "too-many-zeros", "interior-resonance", "interior-pole-enclosed",
  /// "roots-not-separated", "no-zero-enclosed", and the three agreement
  /// certificates between the argument-principle count, the moment-method
  /// roots and the refinement: "moment-root-without-zero" (a moment-method
  /// root whose own small contour encloses no zero; it is not reported),
  /// "multiplicity-count-mismatch" (the local multiplicities do not add up to
  /// the count on the declared contour), "newton-left-local-contour" (the
  /// Newton refinement of a simple zero left the small contour that
  /// certified it; the zero is reported at its local mean), and
  /// "pole-outside-contour" (a refined zero outside the declared contour; it
  /// is not reported).
  std::vector<std::string> failedCertificates{};
};

/// # BoundStatePole
///
/// Mass as the complex bound-state pole of Section 13.3: the zeros of
/// \f$ D_C(s)=\det F_C(s) \f$, with \f$ F_C \f$ the exact meromorphic Feshbach
/// response pencil of a persistent bound cluster \f$ C \f$, continued in the
/// complex spectral parameter \f$ s \f$.
///
/// Mass is not defined here by an incoherent sum of moduli. The crossing sum
/// \f$ \kappa\sum_c|\pi_\perp(c)| \f$ of `observables::CrossingReadouts` is a
/// crossing functional of a world tube and is a different object; it is never
/// read as this pole and this pole is never assembled from it.
///
/// ## The response pencil
///
/// The cluster's coordinates are the interface \f$ B \f$ and the rest of the
/// complex is the interior \f$ I \f$ that is eliminated. For the pencil
/// \f$ P(s)=A-sM \f$,
/// \f[
///   F_C(s)=P_{BB}(s)-P_{BI}(s)P_{II}(s)^{-1}P_{IB}(s),
///   \qquad D_C(s)=\det F_C(s),
/// \f]
/// which is `chainhodge::PencilSchur::feshbach` at the shift \f$ s \f$. By the
/// determinant factorization \f$ \det P=\det P_{II}\,\det F_C \f$, the zeros of
/// \f$ D_C \f$ on a finite complex are exactly the eigenvalues of the full
/// pencil that are not eigenvalues of the interior block, with matching
/// multiplicities. They therefore exist and are generically simple and
/// isolated: the content of the certificate is the value of \f$ s_C \f$ and of
/// the binding shift, not its existence.
///
/// ## How a zero is found
///
/// Every quantity below is analytic. The derivative
/// \f[
///   F_C'(s) = -M_{BB} + M_{BI}X + ZM_{IB} - ZM_{II}X,
///   \qquad X=P_{II}^{-1}P_{IB},\quad Z=P_{BI}P_{II}^{-1},
/// \f]
/// follows from \f$ dP/ds=-M \f$ in closed form, so the logarithmic derivative
/// \f$ D_C'/D_C=\operatorname{tr}(F_C^{-1}F_C') \f$ carries no finite
/// difference and no step size. It is the logarithmic derivative that is
/// evaluated rather than \f$ D_C \f$ and \f$ D_C' \f$ separately, because the
/// determinant of a large block overflows long before its logarithmic
/// derivative loses a digit.
///
/// On the declared contour the argument principle gives the enclosed count and
/// the power sums of the enclosed zeros,
/// \f[
///   m_p=\frac{1}{2\pi i}\oint s^{p}\,\frac{D_C'(s)}{D_C(s)}\,ds
///      =\sum_i z_i^{\,p},
/// \f]
/// and the zeros are recovered from those moments through the Hankel pencil
/// \f$ (H_{<},H) \f$, \f$ H_{ij}=m_{i+j} \f$,
/// \f$ (H_{<})_{ij}=m_{i+j+1} \f$, whose rank is the number of distinct zeros.
/// \f$ D_C \f$ is meromorphic and not entire — it carries a pole at every
/// eigenvalue of the interior block — so the contour is required to enclose
/// none of those, which is the domain Section 13.3 continues \f$ F_C \f$ on.
/// The enclosed interior poles are counted by the same argument principle
/// applied to \f$ \det P_{II} \f$ and a contour that encloses one is refused
/// by name, rather than answered with a count in which the zeros and the poles
/// have already cancelled.
/// Each root is then read on a small contour that encloses it alone: the
/// argument principle there gives its algebraic multiplicity and the mean of
/// the zeros the small disc holds. A simple zero is refined by Newton on
/// \f$ D_C \f$ through the logarithmic derivative, confined to its small
/// disc; a multiple zero is reported as that local mean, because Newton on a
/// cluster split below the quadrature's resolution overshoots it. Multiple
/// roots are retained with their algebraic multiplicity and their residue
/// matrix rather than split by an ordering convention. The count on the
/// declared contour, the moment-method roots and the refinement are required
/// to agree, and every reported zero to lie inside the declared contour;
/// each disagreement is named in `failedCertificates`.
///
/// Reference: Kravanja and Van Barel, "Computing the Zeros of Analytic
/// Functions", Lecture Notes in Mathematics 1727, Springer (2000).
///
/// ## Boundaries
///
/// Read-only: it reads a pencil and a declared contour, never calls a solver
/// on the geometry, and never enters an emergence objective. An unmeasured
/// value is NaN or an empty vector with the reason named, never zero. Nothing
/// here converts \f$ s_C \f$ into a mass: the theory carries \f$ s_C \f$ and
/// takes no square root of it, and the identification of \f$ s_C \f$ with an
/// invariant momentum square needs a refinement regime that supplies a
/// nondegenerate complex Lorentzian momentum pairing.
class BoundStatePole {
 public:
  /// The Feshbach response \f$ F_C(s) \f$ of the pencil \f$ (A,M) \f$ onto the
  /// \p interface coordinates, as the framework's own Schur complement
  /// supplies it.
  /// @throws std::invalid_argument when \p A and \p M are not square of one
  ///   size, or when an interface index is out of range.
  [[nodiscard]] static chainhodge::FeshbachResult response(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s,
      double rankTolerance = 1e-12);

  /// \f$ D_C(s)=\det F_C(s) \f$.
  /// @throws std::invalid_argument as `response` does.
  [[nodiscard]] static std::complex<double> determinant(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s);

  /// \f$ F_C'(s) \f$, the exact analytic derivative of the response in the
  /// spectral parameter.
  /// @throws std::invalid_argument as `response` does.
  [[nodiscard]] static Eigen::MatrixXcd responseDerivative(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s);

  /// \f$ D_C'(s)/D_C(s)=\operatorname{tr}(F_C^{-1}F_C') \f$. NaN when the
  /// interior block is singular at \p s, since the response is then not
  /// defined there.
  /// @throws std::invalid_argument as `response` does.
  [[nodiscard]] static std::complex<double> logarithmicDerivative(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s);

  /// The zeros of \f$ D_C \f$ inside the circle of radius \p radius about
  /// \p centre, with their multiplicities, residues, separations and
  /// refinement continuation.
  ///
  /// @param A The pencil's operator block.
  /// @param M The pencil's metric block.
  /// @param interface The cluster's coordinates, the ones the response is
  ///   taken onto. Duplicates are ignored and the set is used ascending.
  /// @param centre The centre of the contour.
  /// @param radius Its radius, which must be positive.
  /// @param cfg The declared parameters.
  /// @throws std::invalid_argument when \p A and \p M are not square of one
  ///   size, when an interface index is out of range, or when \p radius is not
  ///   positive.
  [[nodiscard]] static BoundStatePoleRead poles(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> centre,
      double radius, const BoundStatePoleConfig &cfg = {});

  /// The same search run on an assembled pencil's degree-\p k block, with
  /// \p clusterCells the canonical indices of the cluster's cells. The pencil
  /// is `PencilLayer::pencil(assembled, k)`, so the operator and the metric
  /// are the framework's dressed \f$ (\tilde A_k^U, M_k^U) \f$ and nothing is
  /// reassembled here.
  /// @throws std::invalid_argument as `poles` does.
  [[nodiscard]] static BoundStatePoleRead clusterPoles(
      const AssembledPencil &assembled, int k,
      const std::vector<int> &clusterCells, std::complex<double> centre,
      double radius, const BoundStatePoleConfig &cfg = {});
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BOUNDSTATEPOLE_H
