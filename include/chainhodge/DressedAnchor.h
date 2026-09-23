// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_DRESSEDANCHOR_H
#define TESSERA_CHAINHODGE_DRESSEDANCHOR_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/FaceAnchor.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// # DeclaredPaths
///
/// The declared path rule of an anchor: one base vertex \f$ p \f$ and, for each
/// vertex \f$ v \f$ the rule reaches, one walk in the complex's 1-skeleton from
/// \f$ p \f$ to \f$ v \f$. The rule is declared once and the same base vertex
/// and the same rule are used for every face of the anchor's atlas, which is
/// what makes the faces' coordinates comparable with one another.
///
/// The transport of a coefficient sitting at \f$ v \f$ to \f$ p \f$ along the
/// walk \f$ p = u_0, u_1, \dots, u_k = v \f$ is the ordered product
/// \f[
///   T_p(v) = U_{u_0u_1}U_{u_1u_2}\cdots U_{u_{k-1}u_k},
/// \f]
/// evaluated with `Connection::link`, which reads \f$ U_{yx} \f$ as
/// \f$ U_{xy}^{-1} \f$ exactly. It is the open analogue of
/// `Connection::holonomy`, which requires a closed walk and therefore cannot
/// express a transport between two different vertices. Because a link
/// transforms as \f$ U_{xy}\mapsto g_x^{-1}U_{xy}g_y \f$ under a vertex gauge
/// \f$ g \f$, the product telescopes to
/// \f$ T_p(v)\mapsto g_p^{-1}T_p(v)g_v \f$: the walk's interior cancels and
/// only the two endpoints survive. That single property is what makes the
/// anchor's coordinate covariant, and it is why a transport declared as a bare
/// number rather than as a walk is refused by the covariance check.
class DeclaredPaths {
 public:
  /// The breadth-first path rule rooted at \p basePoint: the walk to each
  /// vertex is a shortest path in the 1-skeleton. The parent of a vertex is the
  /// first vertex adjacent to it that the breadth-first scan dequeues, and each
  /// vertex's neighbours are scanned in ascending vertex id, so the rule is
  /// fixed by the vertex ids alone and is reproducible. When
  /// \p support is non-empty the walks stay inside it, which is how a cluster's
  /// own support declares its own paths; an empty \p support means the whole
  /// complex.
  /// @throws std::invalid_argument when the complex has no edges, when
  ///   \p basePoint is not a vertex of the complex, or when \p support is
  ///   non-empty and does not contain \p basePoint.
  [[nodiscard]] static DeclaredPaths breadthFirst(const cobordism::ChainComplex &K,
                                                  std::uint64_t basePoint,
                                                  const std::vector<std::uint64_t> &support = {});

  /// A caller-declared path rule: `walks[v]` is the vertex sequence of the walk
  /// from \p basePoint to `v`, starting at \p basePoint and ending at `v`. The
  /// walks need not be shortest and need not lie in a tree; two vertices may be
  /// reached along paths whose loop holonomy differs, and that dependence is
  /// reported by the anchor rather than erased.
  /// @throws std::invalid_argument when a walk does not start at
  ///   \p basePoint, does not end at its own vertex, or steps along a pair that
  ///   is not an edge of the complex.
  [[nodiscard]] static DeclaredPaths fromWalks(
      const cobordism::ChainComplex &K, std::uint64_t basePoint,
      const std::map<std::uint64_t, std::vector<std::uint64_t>> &walks);

  /// A transport table declared as bare numbers, with no walk behind it:
  /// `transports[v]` is used as \f$ T_p(v) \f$ whatever the connection is.
  /// This is the raw restriction the whitepaper names: it does not transform
  /// under a vertex gauge, so it fails the connection-dressed covariance check
  /// and the anchor refuses rather than reporting a gauge-dependent number. It
  /// exists so that the refusal is reachable and testable, and for no other
  /// purpose.
  [[nodiscard]] static DeclaredPaths declaredTransports(
      std::uint64_t basePoint, const std::map<std::uint64_t, Complex> &transports);

  /// The base vertex \f$ p \f$ every walk starts at.
  [[nodiscard]] std::uint64_t basePoint() const noexcept { return basePoint_; }
  /// Whether the transports are evaluated from declared walks on the
  /// connection. False for `declaredTransports`.
  [[nodiscard]] bool derivedFromWalks() const noexcept { return derivedFromWalks_; }
  /// Whether the rule declares a transport for \p v. The base vertex always
  /// reaches itself, with transport 1.
  [[nodiscard]] bool reaches(std::uint64_t v) const;
  /// The declared walks, keyed by their end vertex. Empty for
  /// `declaredTransports`.
  [[nodiscard]] const std::map<std::uint64_t, std::vector<std::uint64_t>> &walks() const noexcept {
    return walks_;
  }
  /// \f$ T_p(v) \f$ evaluated on \p U.
  /// @throws std::invalid_argument when the rule does not reach \p v.
  [[nodiscard]] Complex transport(const Connection &U, std::uint64_t v) const;

 private:
  std::uint64_t basePoint_{0};
  bool derivedFromWalks_{true};
  std::map<std::uint64_t, std::vector<std::uint64_t>> walks_{};
  std::map<std::uint64_t, Complex> transports_{};
};

/// One oriented triangle's dressed restriction to its three ordered boundary
/// edges, transported to the base vertex.
struct FaceRestriction {
  /// Index of the triangle in `ChainComplex::kSimplexVertices(2)`.
  std::size_t faceIndex{0};
  /// The triangle's three canonical \f$ C_1 \f$ edge indices, in the cyclic
  /// order of the oriented triangle \f$ \tau = (v_0\to v_1\to v_2\to v_0) \f$:
  /// \f$ (v_0v_1), (v_1v_2), (v_0v_2) \f$.
  std::vector<int> edgeIndices{};
  /// The incidence sign of each of those edges in \f$ \partial\tau \f$, which
  /// for \f$ \partial[v_0,v_1,v_2] = [v_1,v_2]-[v_0,v_2]+[v_0,v_1] \f$ and the
  /// cyclic order above is \f$ (+1,+1,-1) \f$.
  std::vector<int> incidenceSigns{};
  /// The base vertex \f$ b(e) = \min e \f$ of each of those edges, the vertex
  /// at which the edge's coefficient sits under the repository's degree-one
  /// representation.
  std::vector<std::uint64_t> basePoints{};
  /// \f$ T_p(b(e_i)) \f$, the transport of each edge's coefficient to the base
  /// vertex along the declared walk.
  std::vector<Complex> transports{};
  /// \f$ \varepsilon_i\,T_p(b(e_i)) \f$: the three nonzero entries of
  /// \f$ \mathrm{res}_{\tau\to p}(U) \f$, one per row.
  std::vector<Complex> factors{};
};

/// The anchor certificate of one base band on one atlas of faces.
struct DressedAnchorRead {
  /// The base vertex \f$ p \f$ of the declared path rule.
  std::uint64_t basePoint{0};
  /// The rank \f$ r \f$ of the base band, between one and three.
  int bandRank{0};
  /// The canonical triangle indices of the atlas, in the order given.
  std::vector<std::size_t> faceIndices{};
  /// `coordinates[i]` holds the \f$ \binom{3}{r} \f$ exterior-power
  /// coordinates of face `faceIndices[i]`, in lexicographic order of the row
  /// subset. For \f$ r = 3 \f$ there is exactly one and it is
  /// \f$ \Delta_\tau \f$.
  std::vector<std::vector<Complex>> coordinates{};
  /// The invariant coordinate \f$ \alpha_\tau \f$ of each face of the atlas,
  /// when it has been attached by `withInvariantCoordinates`; empty otherwise.
  std::vector<Complex> invariantCoordinates{};
  /// The number of faces of the atlas carrying at least one coordinate above
  /// the numerical zero threshold.
  std::size_t anchoringFaces{0};
  /// The largest Hadamard bound over the atlas's coordinates: the product, over
  /// the rows a coordinate selects, of the Euclidean norms of the dressed
  /// restriction's rows. It is the scale the numerical zero threshold is taken
  /// against and is an explicitly labeled numerical certificate, never part of
  /// the physical definition.
  double coordinateScale{0.0};
  /// The largest residual of the connection-dressed covariance law over the
  /// atlas, measured against the declared verification gauge.
  double covarianceResidual{0.0};
  /// The largest residual of the transition cocycle
  /// \f$ t_{ab}t_{bc} = t_{ac} \f$ over the triples of anchoring faces whose
  /// first coordinate does not vanish; \f$ 0 \f$ when fewer than three such
  /// faces exist.
  double transitionCocycleResidual{0.0};
  /// The tolerance both residuals and the zero threshold were judged against.
  double tolerance{0.0};
  /// Whether the anchor certificate holds: a non-empty atlas, a verified
  /// covariance, and a profile that is not identically zero.
  bool anchored{false};
  /// The named reasons the anchor refuses; empty when `anchored` is true.
  std::vector<std::string> failedCertificates{};
};

/// # DressedAnchor
///
/// The anchor of a base band to oriented two-simplices by the dressed
/// coordinate: \f$ \Delta_\tau \f$, its \f$ \Lambda^r \f$ variant for a band of
/// rank below three, the profile those coordinates form, and the
/// determinant-line transition functions on overlaps.
///
/// ## The restriction
///
/// Choose one base vertex \f$ p \f$ in the cluster \f$ Q \f$. For an oriented
/// triangle \f$ \tau\subset Q \f$, \f$ \mathrm{res}_{\tau\to p}(U) \f$
/// restricts a one-chain to \f$ \tau \f$'s three ordered boundary edges and
/// parallel-transports those three coefficients to \f$ p \f$ along the declared
/// walks. As a matrix on chains it is the \f$ 3\times n_1 \f$ matrix
/// \f[
///   \bigl(\mathrm{res}_{\tau\to p}(U)\bigr)_{i,e}
///     = \varepsilon_i\,\delta_{e,e_i(\tau)}\,T_p\bigl(b(e_i)\bigr),
/// \f]
/// where \f$ e_1,e_2,e_3 \f$ are \f$ \tau \f$'s edges in the cyclic order of
/// the oriented triangle, \f$ \varepsilon_i \f$ their incidence signs in
/// \f$ \partial\tau \f$, \f$ b(e) = \min e \f$ the vertex at which an edge's
/// coefficient sits, and \f$ T_p \f$ the declared transport. Only three entries
/// are nonzero, so the matrix is never formed when a coordinate is wanted; it
/// is available for verifying the covariance law as written.
///
/// The base-point convention \f$ b(e) = \min e \f$ is not free: it is the
/// convention of the repository's degree-one representation
/// \f$ \rho_1(g) = \mathrm{diag}(g_{b(\sigma)}^{-1}) \f$
/// (`CovariantChainHodge::rho`) and of the dressing
/// \f$ (M_1^U)_{ee'} = (M_1)_{ee'}U_{b(e)b(e')} \f$, and it is what makes the
/// covariance law close. Under a vertex gauge \f$ g \f$, which acts on links by
/// \f$ U_{xy}\mapsto g_x^{-1}U_{xy}g_y \f$ and hence on transports by
/// \f$ T_p(v)\mapsto g_p^{-1}T_p(v)g_v \f$,
/// \f[
///   \mathrm{res}_{\tau\to p}(U^g)\,\rho_1(g) = g_p^{-1}\,\mathrm{res}_{\tau\to p}(U).
/// \f]
/// The whitepaper writes the same law with \f$ \rho_1(g)^{-1} \f$ because it
/// takes the opposite sign convention for \f$ \rho_1 \f$; the consequence, and
/// the only thing either convention is used for, is identical.
///
/// ## The coordinate and its profile
///
/// For a base band of rank three with chain frame \f$ \Phi_Q \f$
/// (\f$ n_1\times 3 \f$) the dressed coordinate is
/// \f[
///   \Delta_\tau = \det\bigl(\mathrm{res}_{\tau\to p}(U)\,\Phi_Q\bigr)\in\mathbb{C}.
/// \f]
/// A chain frame transforms under the gauge as
/// \f$ \Phi_Q\mapsto\rho_1(g)\Phi_Q g_Q \f$, so
/// \f$ \Delta_\tau\mapsto g_p^{-3}(\det g_Q)\,\Delta_\tau \f$: one common
/// factor, the same on every face. Hence the profile
/// \f[
///   [\Delta_Q] = [\Delta_\tau]_{\tau\subset Q}
/// \f]
/// is frame-independent as a point of a projective space wherever it is
/// nonzero. It records the complete complex interference pattern across the
/// candidate anchoring faces without selecting a largest modulus or fitting
/// convex weights after seeing the band. An extended fiber may be anchored by
/// an atlas of faces; concentration on one face is not required.
///
/// For a base band of rank \f$ r < 3 \f$ — the \f$ j = 1/2 \f$ doublet of an
/// odd-monopole support has \f$ r = 2 \f$ — the same construction uses the
/// \f$ r \f$-th exterior power,
/// \f$ \Lambda^r\,\mathrm{res}_{\tau\to p}(U)\,\Phi_Q\in\Lambda^r\mathbb{C}^3 \f$,
/// a point of \f$ \mathbb{P}(\Lambda^r\mathbb{C}^3) \f$, which for
/// \f$ r = 2 \f$ is again \f$ \mathbb{P}^2 \f$. Concretely these are the
/// \f$ \binom{3}{r} \f$ maximal minors of the \f$ 3\times r \f$ matrix
/// \f$ \mathrm{res}_{\tau\to p}(U)\Phi_Q \f$, and the transformation law is the
/// same with \f$ \det g_Q \f$ replaced by the induced action on
/// \f$ \Lambda^r \f$, which on a maximal exterior power is again multiplication
/// by \f$ \det g_Q \f$, and \f$ g_p^{-3} \f$ replaced by \f$ g_p^{-r} \f$.
///
/// No modulus, square root, free face weight or real-valued score enters the
/// physical definition. Numerical projective distances, Hadamard scales and
/// residuals are reported as explicitly labeled numerical certificates and are
/// never fed back into a physical statement.
///
/// ## The transitions
///
/// The determinant line of the band is trivialized on each face of the atlas
/// where its coordinate does not vanish. On the overlap of two such charts the
/// transition function is the ratio
/// \f[
///   t_{\tau\tau'} = \frac{\Lambda^r_\tau}{\Lambda^r_{\tau'}}\in\mathbb{C}^\times,
/// \f]
/// taken at the same exterior-power coordinate index. Both the common factor
/// \f$ g_p^{-r}\det g_Q \f$ of the microscopic gauge and the frame change
/// \f$ \Phi_Q\mapsto\Phi_Q g_Q \f$ cancel in the ratio, so the transitions are
/// fully invariant, and the cocycle identity
/// \f$ t_{ab}t_{bc} = t_{ac} \f$ holds by construction; its numerical residual
/// is reported. Changing the base vertex from \f$ p \f$ to \f$ p' \f$ is the
/// other chart change: when the walks from \f$ p' \f$ factor through \f$ p \f$
/// the ratio is the scalar \f$ T_{p'}(p)^r \f$ on every face, and when they do
/// not, the ratio is a product of microscopic face holonomies and varies from
/// face to face. That dependence on the declared paths is exactly microscopic
/// face holonomy and is reported, not erased.
///
/// ## The second coordinate and the refusal
///
/// The anchor certificate of Section 10 is the persistent nonzero projective
/// profile \f$ [\Delta_Q] \f$, the complex profile \f$ \{\alpha_\tau\} \f$ of
/// invariant coordinates supplied by `FaceAnchor`, and the determinant-line
/// transition functions on overlaps; `withInvariantCoordinates` attaches the
/// second of the three to a read that already carries the first and the third.
///
/// The anchor refuses, rather than reporting a gauge-dependent raw restriction,
/// when the connection-dressed covariance cannot be verified — which is what
/// happens when the transports are declared as bare numbers instead of as walks
/// — and it refuses when the profile is identically zero, because a projective
/// point with no nonzero coordinate does not exist. The second refusal is not a
/// defect: by the anchoring theorem an exact band at flat connection has
/// \f$ \Delta_\tau = 0 \f$ on every face, so refusing is the correct reading of
/// a band that anchors nowhere.
///
/// ## The anchoring theorem
///
/// Fix an oriented triangle \f$ \tau = (x\to y\to z\to x) \f$. The restriction
/// to its three ordered boundary edges of the twisted coboundary of a vertex
/// potential, in the convention
/// \f$ (\delta_0^U f)_{xy} = U_{xy}f_y - f_x \f$, is
/// \f[
///   \begin{pmatrix} -1 & U_{xy} & 0\\ 0 & -1 & U_{yz}\\ U_{zx} & 0 & -1\end{pmatrix},
///   \qquad \det = U_{xy}U_{yz}U_{zx} - 1 = F_\tau - 1,
/// \f]
/// which `twistedCoboundaryBlock` returns. Consequently a rank-three band
/// contained in \f$ \mathrm{im}\,\delta_0^U \f$ — an exact band — has
/// \f$ \Delta_\tau \f$ proportional to \f$ F_\tau - 1 \f$ with a nonvanishing
/// factor fixed by the path rule, so an exact band anchors to a face only where
/// the face holonomy is nontrivial, with anchor strength equal to the
/// curvature, and at flat connection no exact band anchors anywhere. A band in
/// \f$ \mathrm{im}\,\partial_2 \f$ — a coexact band — anchors to \f$ \tau \f$
/// whenever the restrictions to \f$ \tau \f$'s edges of \f$ \partial\tau \f$
/// and of the boundaries of \f$ \tau \f$'s neighbouring faces span
/// \f$ \mathbb{C}^3 \f$, which holds on every face of the tetrahedron. On the
/// regular tetrahedron with trivial connection the exact band has
/// \f$ \Delta_\tau = 0 \f$ on all four faces and the coexact band has
/// \f$ \Delta_\tau \ne 0 \f$, equal in modulus on all four, so the anchor
/// certificate selects the coexact band uniquely. Since coexact bands are
/// supported by 2-cells and not by cycles, anchorable fibers exist on
/// contractible supports, the solid tetrahedron being one.
///
/// Because every 2-simplex has exactly three boundary edges, the construction
/// is independent of the ambient spectral dimension.
class DressedAnchor {
 public:
  // ---- the restriction ----

  /// The triangle's dressed restriction: its three ordered boundary edges,
  /// their incidence signs, their base vertices and the transports of those
  /// base vertices to \f$ p \f$.
  /// @throws std::invalid_argument on a bad face index, a complex with no
  ///   triangles, a missing edge, or a base vertex the path rule does not
  ///   reach.
  [[nodiscard]] static FaceRestriction faceRestriction(const cobordism::ChainComplex &K,
                                                       const Connection &U,
                                                       const DeclaredPaths &paths,
                                                       std::size_t faceIndex);

  /// \f$ \mathrm{res}_{\tau\to p}(U) \f$ as a dense \f$ 3\times n_1 \f$ matrix.
  /// Three of its entries are nonzero. It is materialized for verifying the
  /// covariance law as written and for nothing else; `restrictedFrame` is the
  /// working route and forms no \f$ n_1 \f$-wide object.
  [[nodiscard]] static Eigen::MatrixXcd restriction(const cobordism::ChainComplex &K,
                                                    const Connection &U,
                                                    const DeclaredPaths &paths,
                                                    std::size_t faceIndex);

  /// \f$ \mathrm{res}_{\tau\to p}(U)\,\Phi_Q \f$, an \f$ 3\times r \f$ matrix,
  /// formed by scaling three rows of \p Phi.
  /// @throws std::invalid_argument when \p Phi does not have one row per
  ///   1-simplex of the complex.
  [[nodiscard]] static Eigen::MatrixXcd restrictedFrame(const cobordism::ChainComplex &K,
                                                        const Connection &U,
                                                        const DeclaredPaths &paths,
                                                        std::size_t faceIndex,
                                                        const Eigen::MatrixXcd &Phi);

  /// The restriction to \f$ \tau \f$'s three ordered boundary edges of the
  /// twisted coboundary of a vertex potential, rows in the cyclic order
  /// \f$ (x,y),(y,z),(z,x) \f$ and columns in the vertex order
  /// \f$ x < y < z \f$; its determinant is \f$ F_\tau - 1 \f$.
  /// @throws std::invalid_argument on a bad face index or a complex with no
  ///   triangles.
  [[nodiscard]] static Eigen::Matrix3cd twistedCoboundaryBlock(const cobordism::ChainComplex &K,
                                                               const Connection &U,
                                                               std::size_t faceIndex);

  // ---- the coordinates ----

  /// \f$ \Lambda^r A \f$ of a \f$ 3\times r \f$ matrix: its \f$ \binom{3}{r} \f$
  /// maximal minors, in lexicographic order of the row subset. For
  /// \f$ r = 3 \f$ the single entry is \f$ \det A \f$; for \f$ r = 2 \f$ the
  /// three entries are the minors on rows \f$ (0,1),(0,2),(1,2) \f$; for
  /// \f$ r = 1 \f$ they are \f$ A \f$'s three entries.
  /// @throws std::invalid_argument when \p A does not have three rows or its
  ///   column count is not one, two or three.
  [[nodiscard]] static std::vector<Complex> exteriorPower(const Eigen::MatrixXcd &A);

  /// The \f$ \Lambda^r \f$ coordinates of one face:
  /// `exteriorPower(restrictedFrame(...))`.
  [[nodiscard]] static std::vector<Complex> dressedCoordinates(const cobordism::ChainComplex &K,
                                                               const Connection &U,
                                                               const DeclaredPaths &paths,
                                                               std::size_t faceIndex,
                                                               const Eigen::MatrixXcd &Phi);

  /// \f$ \Delta_\tau = \det(\mathrm{res}_{\tau\to p}(U)\Phi_Q) \f$ for a
  /// rank-three band.
  /// @throws std::invalid_argument when \p Phi does not have exactly three
  ///   columns.
  [[nodiscard]] static Complex dressedCoordinate(const cobordism::ChainComplex &K,
                                                 const Connection &U, const DeclaredPaths &paths,
                                                 std::size_t faceIndex,
                                                 const Eigen::MatrixXcd &Phi);

  /// The triangles every one of whose three edge base vertices the path rule
  /// reaches: the faces an atlas may be built from.
  [[nodiscard]] static std::vector<std::size_t> anchorableFaces(const cobordism::ChainComplex &K,
                                                                const DeclaredPaths &paths);

  // ---- the certificate ----

  /// The anchor certificate of \p Phi on the atlas \p faceIndices: the
  /// \f$ \Lambda^r \f$ coordinates of every face, the covariance residual
  /// against the verification gauge, the transition cocycle residual, and the
  /// named refusals.
  ///
  /// @param gaugeSeed The seed of the deterministic vertex gauge the covariance
  ///   law is verified against, following the same convention as
  ///   `CovariantChainHodge`'s own verification gauge.
  /// @param tolerance The relative tolerance both residuals and the numerical
  ///   zero threshold are judged against.
  /// @throws std::invalid_argument when \p Phi does not have one row per
  ///   1-simplex, when its column count is not one, two or three, when a face
  ///   index is out of range, or when the path rule does not reach one of a
  ///   declared face's base vertices.
  [[nodiscard]] static DressedAnchorRead profile(const cobordism::ChainComplex &K,
                                                 const Connection &U, const DeclaredPaths &paths,
                                                 const std::vector<std::size_t> &faceIndices,
                                                 const Eigen::MatrixXcd &Phi,
                                                 double tolerance = 1e-9,
                                                 std::uint64_t gaugeSeed = 7);

  /// The same read with the invariant coordinates \f$ \alpha_\tau \f$ of its
  /// atlas attached, from the fiber's geometric images, completing the anchor
  /// certificate's three parts. The read's atlas and \p cov must describe the
  /// same complex.
  [[nodiscard]] static DressedAnchorRead withInvariantCoordinates(const DressedAnchorRead &read,
                                                                  const CovariantChainHodge &cov,
                                                                  const Eigen::MatrixXcd &Zdual,
                                                                  const Eigen::MatrixXcd &Z);

  /// The residual of the connection-dressed covariance law
  /// \f$ \mathrm{res}_{\tau\to p}(U^g)\rho_1(g) = g_p^{-1}\mathrm{res}_{\tau\to p}(U) \f$
  /// over the atlas, for the declared vertex gauge \p gauge: the largest
  /// entrywise difference of the two sides divided by the larger of one and the
  /// right-hand entry's modulus. Because both sides are supported on the same
  /// three entries and \f$ \rho_1(g) \f$ is diagonal, comparing those three
  /// entries is comparing the matrices.
  [[nodiscard]] static double covarianceResidual(const cobordism::ChainComplex &K,
                                                 const Connection &U, const DeclaredPaths &paths,
                                                 const std::vector<std::size_t> &faceIndices,
                                                 const std::map<std::uint64_t, Complex> &gauge);

  /// The deterministic verification gauge of \p seed: one nonzero complex
  /// number per vertex of the complex, reproducible from the seed alone.
  [[nodiscard]] static std::map<std::uint64_t, Complex> verificationGauge(
      const cobordism::ChainComplex &K, std::uint64_t seed);

  // ---- the transitions ----

  /// The transition function on the overlap of the charts of two faces of one
  /// read: the ratio of their \p coordinate-th exterior-power coordinates.
  /// @throws std::invalid_argument on a bad slot or coordinate index, and
  ///   std::runtime_error when the denominator vanishes, which means the second
  ///   face's chart is empty and the two charts do not overlap there.
  [[nodiscard]] static Complex faceTransition(const DressedAnchorRead &read, std::size_t faceSlot,
                                              std::size_t otherFaceSlot,
                                              std::size_t coordinate = 0);

  /// The transition between the charts of two base vertices on one face: the
  /// ratio of the coordinate of \p read to that of \p other, which must be
  /// reads of the same band on the same atlas taken at two base vertices. When
  /// the walks of \p read factor through the base vertex of \p other the ratio
  /// is the scalar \f$ T_{p}(p')^r \f$ on every face; otherwise it is a product
  /// of microscopic face holonomies and varies from face to face.
  /// @throws std::invalid_argument when the two reads do not share an atlas or
  ///   a rank, and std::runtime_error when the denominator vanishes.
  [[nodiscard]] static Complex basePointTransition(const DressedAnchorRead &read,
                                                   const DressedAnchorRead &other,
                                                   std::size_t faceSlot,
                                                   std::size_t coordinate = 0);

  /// The largest residual of the cocycle identity
  /// \f$ t_{ab}t_{bc} = t_{ac} \f$ over the triples of faces of \p read whose
  /// \p coordinate-th coordinate does not vanish; \f$ 0 \f$ when fewer than
  /// three such faces exist.
  [[nodiscard]] static double transitionCocycleResidual(const DressedAnchorRead &read,
                                                        std::size_t coordinate = 0);

  /// The chordal Fubini-Study distance between two reads' profiles, read as
  /// points of the same projective space: \f$ 0 \f$ when they are proportional
  /// and \f$ 1 \f$ when they are orthogonal. It is a reported numerical
  /// stability certificate. It is the one place a conjugate appears, and
  /// nothing physical is derived from it; the exact statements are made with
  /// the transition ratios, which are conjugate-free.
  /// @throws std::invalid_argument when the two reads do not share an atlas or
  ///   a rank, and std::runtime_error when either profile is identically zero
  ///   at its own numerical zero threshold, the judgement under which
  ///   `profile` counts no anchoring face.
  [[nodiscard]] static double projectiveDistance(const DressedAnchorRead &a,
                                                 const DressedAnchorRead &b);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_DRESSEDANCHOR_H
