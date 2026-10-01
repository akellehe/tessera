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
/// Every declared parameter of the pole read, echoed on each read so a stored
/// result carries the configuration that produced it.
struct BoundStatePoleConfig {
  /// The relative tolerance of every decision the read makes, as a fraction
  /// of the largest singular value of the block whose spectrum is read
  /// (`BoundStatePoleRead::scale`): a singular value at or below this fraction
  /// of that scale counts as zero in every rank decision (the rank of each
  /// residue, the ranks of the powers of each cluster's nilpotent part, the
  /// invertibility of the metric), and two eigenvalues whose distance is at or
  /// below this fraction of that scale belong to one cluster and form one
  /// pole. No decision is made at any other tolerance.
  double rankTolerance = 1e-15;
  /// The free threshold the binding shift is measured against: the complex
  /// spectral value at which the cluster's content is unbound. Empty leaves
  /// the binding shift unreported, since no threshold is derivable from the
  /// pencil alone.
  std::optional<std::complex<double>> freeThreshold{};
};

/// # BoundStatePoleRead
///
/// The zeros of \f$ D_C(s)=\det F_C(s) \f$, read exactly from the spectrum of
/// the pencil, with the certificates Section 13.3 attaches to a bound-state
/// pole. Every vector below that is described as parallel to `poles` has one
/// entry per reported pole, in the order of `poles`.
struct BoundStatePoleRead {
  /// The largest singular value of the block \f$ T=M^{-1}A \f$ whose
  /// spectrum is read: the reference of every rank decision and of the
  /// clustering. Two eigenvalues at distance at or below `rankTolerance`
  /// times this form one pole.
  double scale = std::numeric_limits<double>::quiet_NaN();

  /// The clusters of eigenvalues of the pencil \f$ (A,M) \f$, ascending by
  /// (real part, imaginary part). A cluster that no eigenvalue of the
  /// interior block meets is a zero \f$ s_C \f$ of \f$ D_C \f$. A cluster
  /// that an interior eigenvalue meets is reported with the others and
  /// flagged in `atInteriorPole`. A cluster is a set of eigenvalues connected
  /// by distances at or below `rankTolerance` times `scale`; its value is the
  /// trace of its Schur block over its size, which is exactly the repeated
  /// eigenvalue when the cluster is one and lies within `clusterSpread` of
  /// every member otherwise.
  std::vector<std::complex<double>> poles{};
  /// Whether an eigenvalue of the interior pencil \f$ (A_{II},M_{II}) \f$
  /// lies within `rankTolerance` times `scale` of a member of each pole's
  /// cluster. Such a point is a pole of \f$ F_C \f$ and lies outside the
  /// domain Section 13.3 continues \f$ F_C \f$ on, so the read does not
  /// establish what order \f$ D_C \f$ has there: the cluster's value, its
  /// multiplicity in the pencil, its Jordan structure and its residue matrix
  /// are those of the pencil's eigenvalue and are reported as such, and
  /// "eigenvalue-at-interior-pole" is named in `failedCertificates`. Parallel
  /// to `poles`.
  std::vector<bool> atInteriorPole{};
  /// The algebraic multiplicity of each pole: the size of its cluster, which
  /// is the dimension of the generalized eigenspace the pole's spectral
  /// projector projects onto. Parallel to `poles`.
  std::vector<std::size_t> multiplicity{};
  /// The geometric multiplicity of each pole: the number of its Jordan blocks,
  /// which is the size of the cluster less the rank of its nilpotent part at
  /// `rankTolerance`. Parallel to `poles`.
  std::vector<std::size_t> geometricMultiplicity{};
  /// The sizes of the Jordan blocks of each pole, descending, read from the
  /// ranks of the powers of the cluster's nilpotent part at `rankTolerance`:
  /// the number of blocks of size at least \f$ j \f$ is
  /// \f$ \operatorname{rank}N^{j-1}-\operatorname{rank}N^{j} \f$. They sum to
  /// the algebraic multiplicity. Parallel to `poles`.
  std::vector<std::vector<std::size_t>> jordanBlocks{};
  /// The diameter of each cluster: the largest distance between two of its
  /// eigenvalues, zero for a cluster of one eigenvalue or of an exactly
  /// repeated one. Parallel to `poles`.
  std::vector<double> clusterSpread{};
  /// Whether the pole met the simple-isolated specification of Section 13.3,
  /// \f$ D_C(s_C)=0 \f$ and \f$ D_C'(s_C)\neq 0 \f$, which for a zero of a
  /// holomorphic function is the statement that its algebraic multiplicity is
  /// one. Parallel to `poles`.
  std::vector<bool> simple{};
  /// The distance from each pole to the nearest other reported pole; infinite
  /// when it is the only one. Parallel to `poles`.
  std::vector<double> separation{};
  /// The residual \f$ \|TV-VU_{11}\|_F \f$ of each pole's invariant subspace,
  /// with \f$ V \f$ the orthonormal Schur basis of the cluster's generalized
  /// eigenspace and \f$ U_{11}=V^HTV \f$ the block of \f$ T \f$ on it: zero
  /// exactly when \f$ V \f$ spans an invariant subspace of \f$ T \f$, and the
  /// rounding of the Schur form otherwise. It is absolute; `scale` is the
  /// size it is measured against. Parallel to `poles`.
  std::vector<double> subspaceResidual{};

  /// The residue of the supported resolvent \f$ F_C(s)^{-1} \f$ at each pole,
  /// flat row-major over the interface coordinates, parallel to `poles`. It is
  /// \f$ -(\Pi M^{-1})_{BB} \f$ with \f$ \Pi \f$ the spectral projector of
  /// \f$ T=M^{-1}A \f$ onto the cluster's generalized eigenspace, since
  /// \f$ F_C(s)^{-1} \f$ is the interface block of \f$ (A-sM)^{-1}
  /// =-(s-T)^{-1}M^{-1} \f$; for the identity metric on a block with no
  /// interior it is minus the spectral projector itself. This is the pole
  /// residue the whitepaper reports beside the pole, and it is a matrix
  /// rather than a number so that a multiple pole's local data is retained
  /// rather than summarized. For a pole flagged in `atInteriorPole` it is the
  /// same matrix, the interface block of the residue of the full resolvent
  /// \f$ (A-sM)^{-1} \f$ at the pencil's eigenvalue.
  std::vector<std::vector<std::complex<double>>> residue{};
  /// The Frobenius norm of each residue. Parallel to `poles`.
  std::vector<double> residueNorm{};
  /// The rank of each residue at `rankTolerance`, which for a simple pole of
  /// a pencil with no interior is one and for a pole of a block with no
  /// interior is its algebraic multiplicity. Parallel to `poles`.
  std::vector<std::size_t> residueRank{};

  /// \f$ s_C-s_{\rm free} \f$ against the declared free threshold, parallel to
  /// `poles`; empty when no threshold is declared. The whitepaper reports the
  /// binding shift below the free threshold beside the pole and takes no
  /// square root of either.
  std::vector<std::complex<double>> bindingShift{};

  /// The distinct eigenvalues of the interior pencil \f$ (A_{II},M_{II}) \f$,
  /// clustered at `rankTolerance` times `scale` and ascending by (real part,
  /// imaginary part): the poles of \f$ F_C \f$, which are the points the
  /// domain Section 13.3 continues \f$ F_C \f$ on excludes. Empty when there
  /// is no interior.
  std::vector<std::complex<double>> interiorPoles{};
  /// The algebraic multiplicity of each interior pole. Parallel to
  /// `interiorPoles`.
  std::vector<std::size_t> interiorMultiplicity{};

  /// Named failures: "empty-interface" (no interface coordinate, so there is
  /// no response to read); "eigenvalue-at-interior-pole" (an eigenvalue of
  /// the pencil within `rankTolerance` times `scale` of an eigenvalue of the
  /// interior block: the point lies outside the domain Section 13.3 continues
  /// \f$ F_C \f$ on, so the read does not establish what order \f$ D_C \f$
  /// has there; the cluster is reported among `poles` with its entry of
  /// `atInteriorPole` true); and "jordan-structure-unresolved" (the
  /// ranks of the powers of a cluster's nilpotent part at `rankTolerance` are
  /// not the rank sequence of a nilpotent matrix, so that cluster's Jordan
  /// block sizes are unmeasured and its `jordanBlocks` entry is empty, while
  /// its pole, multiplicity and residue stand).
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
/// ## How a zero is read
///
/// The pencil is a finite matrix problem, so its zeros are read exactly from
/// its spectrum rather than searched for. With \f$ M \f$ invertible at the
/// declared rank tolerance, the eigenvalues of the pencil are those of
/// \f$ T=M^{-1}A \f$, which the complex Schur form \f$ T=QUQ^H \f$ carries on
/// its diagonal. The diagonal is clustered at the declared tolerance: two
/// eigenvalues whose distance is at or below `rankTolerance` times the largest
/// singular value of \f$ T \f$ are one pole, and a cluster is the transitive
/// closure of that relation. The eigenvalues of the interior pencil
/// \f$ (A_{II},M_{II}) \f$ are read the same way; a cluster of \f$ T \f$ that
/// meets one of them lies outside the domain the response is continued on. It
/// is reported like every other cluster, with its entry of
/// `BoundStatePoleRead::atInteriorPole` true and the failure
/// "eigenvalue-at-interior-pole" named.
///
/// For each cluster the Schur form is reordered so that the cluster
/// leads and its spectral projector \f$ \Pi \f$ follows from one Sylvester
/// solve (`chainhodge::rieszProjector`); the projector's rank is the algebraic
/// multiplicity, and the residue of \f$ F_C^{-1} \f$ is its interface block
/// as `BoundStatePoleRead::residue` states. The Jordan structure of the
/// cluster is read from its Schur block \f$ U_{11} \f$: with \f$ \mu \f$ the
/// trace of \f$ U_{11} \f$ over its size, \f$ N=U_{11}-\mu I \f$ is the
/// nilpotent part, and the ranks of \f$ N,N^2,\dots \f$ at the declared
/// tolerance give the block sizes. A multiple pole is reported once, with its
/// multiplicity and its whole residue matrix, never split by an ordering
/// convention and never replaced by a summary.
///
/// ## Boundaries
///
/// Read-only: it reads a pencil, never calls a solver on the geometry, and
/// never enters an emergence objective. An unmeasured value is NaN or an
/// empty vector with the reason named, never zero. Nothing here converts
/// \f$ s_C \f$ into a mass: the theory carries \f$ s_C \f$ and takes no square
/// root of it, and the identification of \f$ s_C \f$ with an invariant
/// momentum square needs a refinement regime that supplies a nondegenerate
/// complex Lorentzian momentum pairing.
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

  /// \f$ D_C(s)=\det F_C(s) \f$. NaN when the interior block is singular at
  /// \p s, since the response is then not defined there.
  /// @throws std::invalid_argument as `response` does.
  [[nodiscard]] static std::complex<double> determinant(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s);

  /// \f$ F_C'(s) \f$, the exact analytic derivative of the response in the
  /// spectral parameter,
  /// \f[
  ///   F_C'(s) = -M_{BB} + M_{BI}X + ZM_{IB} - ZM_{II}X,
  ///   \qquad X=P_{II}^{-1}P_{IB},\quad Z=P_{BI}P_{II}^{-1},
  /// \f]
  /// which follows from \f$ dP/ds=-M \f$ in closed form. At a simple pole
  /// \f$ s_C \f$ the residue \f$ R \f$ of \f$ F_C^{-1} \f$ satisfies
  /// \f$ \operatorname{tr}(RF_C'(s_C))=1 \f$, the identity a residue is
  /// certified by.
  /// @throws std::invalid_argument as `response` does.
  [[nodiscard]] static Eigen::MatrixXcd responseDerivative(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface, std::complex<double> s);

  /// The zeros of \f$ D_C \f$ with their multiplicities, Jordan structure,
  /// residues, separations and subspace residuals, read exactly from the
  /// spectrum of the pencil.
  ///
  /// @param A The pencil's operator block.
  /// @param M The pencil's metric block, which must be invertible at the
  ///   declared rank tolerance, as must its interior block.
  /// @param interface The cluster's coordinates, the ones the response is
  ///   taken onto. Duplicates are ignored and the set is used ascending.
  /// @param cfg The declared parameters.
  /// @throws std::invalid_argument when \p A and \p M are not square of one
  ///   size, when an interface index is out of range, or when \p M or its
  ///   interior block is singular at the declared rank tolerance, in which
  ///   case the pencil's spectrum is not the spectrum of a matrix and the
  ///   read refuses by name.
  /// @throws std::runtime_error when the complex Schur form of the block or
  ///   of its interior block does not converge.
  [[nodiscard]] static BoundStatePoleRead poles(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<int> &interface,
      const BoundStatePoleConfig &cfg = {});

  /// The same read on an assembled pencil's degree-\p k block, with
  /// \p clusterCells the canonical indices of the cluster's cells. The pencil
  /// is `PencilLayer::pencil(assembled, k)`, so the operator and the metric
  /// are the framework's dressed \f$ (\tilde A_k^U, M_k^U) \f$ and nothing is
  /// reassembled here.
  /// @throws std::invalid_argument and std::runtime_error as `poles` does.
  [[nodiscard]] static BoundStatePoleRead clusterPoles(
      const AssembledPencil &assembled, int k,
      const std::vector<int> &clusterCells,
      const BoundStatePoleConfig &cfg = {});
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_BOUNDSTATEPOLE_H
