// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SURFACE_SEED_H
#define TESSERA_COBORDISM_SURFACE_SEED_H

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// The tube of `tubedCollars`: a prism over one attachment face of each
/// far surface.
struct TubeSpec {
  /// Number of prism layers along the tube.
  int layers{2};
  /// Spacing between consecutive rings.
  double length{1.0};
  /// Scale applied to the interior rings about their centroid.
  double waist{1.0};
  /// Attachment face in the first far surface's own vertex ids.
  std::vector<std::uint64_t> faceA;
  /// Attachment face in the second far surface's own vertex ids.
  std::vector<std::uint64_t> faceB;
  /// Match the last two vertices of `faceB` in reversed order, giving the
  /// orientation-consistent connected sum in which both far surfaces keep the
  /// bulk's induced orientation. False reverses the second.
  bool reflect{true};
};

/// # SurfaceSeed
///
/// A host seeded from boundary surfaces: one spacetime holding the surfaces
/// on disjoint vertex id ranges (`fromSurfaces`), the collar between two of
/// them (`collar`), two collars joined along a removed cell
/// (`joinedCollars`), or two collars joined by a tube (`tubedCollars`).
/// The cobordism a `MultiCobordism` node grows starts from one of these.
struct SurfaceSeed {
    /// One \f$ d \f$-dimensional `Spacetime` (\f$ d \f$ is the surfaces'
    /// dimension plus 1; 3 for tori) holding every surface as its own
    /// simplices — triangles, edges and vertices with the surface's lengths and
    /// zero phases — on disjoint vertex id ranges. Either the bare surfaces
    /// with no \f$ d \f$-cell (`fromSurfaces`) or the collar between two
    /// surfaces (`collar`).
    std::shared_ptr<Spacetime> host;
    /// `vertexIds[s]` maps surface \f$ s \f$'s vertex ids to the host's, so a
    /// marking or fiber stated on the surface carries over. The region form of
    /// `seedInputs` takes those id sets as the blocks.
    std::vector<std::map<std::uint64_t, std::uint64_t>> vertexIds;
    /// The vertex rings of a tube (`tubedCollars`), ring 0 the attachment
    /// face on the first collar's far surface, the last ring the attachment
    /// face on the second collar's, in the order the two faces were matched;
    /// empty on every other seed.
    std::vector<std::vector<std::uint64_t>> tubeRings;

  /// Build the `SurfaceSeed` of the bare \p surfaces (no \f$ d \f$-cell), each
  /// a closed \f$ (d-1) \f$-dimensional `Spacetime`, for example
  /// `observables::SimplicialQubit::flatTorus(tau, n, n).spacetime()`. The
  /// metric signature (Lorentzian, dimension \f$ d \f$), the CDT type and the
  /// preferred foliation are those of `SolidSimplex::spacetime`; the surfaces themselves
  /// are read, never changed.
  /// @throws std::invalid_argument on no surfaces, a null surface, surfaces of
  ///   differing dimension, or a surface without top cells.
  [[nodiscard]] static SurfaceSeed fromSurfaces(
      const std::vector<std::shared_ptr<Spacetime>> &surfaces);

  /// Two collars joined along a removed tetrahedron: four boundary surfaces on
  /// one connected manifold.
  ///
  /// A two-torus collar's harmonic space cannot carry a 4-dimensional target,
  /// and that is forced: for a compact oriented 3-manifold
  /// \f$\operatorname{rank}(H^1(W)\to H^1(\partial W)) = b_1(\partial W)/2\f$,
  /// so two tori leave 2 and four leave 4.
  ///
  /// Each surface is collared with its partner as `collar` does, and the
  /// two collars are joined by removing one all-interior cell from each and
  /// identifying the two boundary spheres. Gluing along a sphere is a connected
  /// sum, which adds no first homology, so \f$b_1 = 2+2 = 4\f$ with nothing
  /// dying on the boundary.
  ///
  /// \p layers must be at least three: a prism cell spans two adjacent layers,
  /// so an all-interior cell exists only once there are two interior layers.
  /// `vertexIds` comes back in the order the surfaces were given.
  /// @throws std::invalid_argument on a null surface, surfaces differing in
  ///   dimension or combinatorics, fewer than three layers, or a join that is
  ///   not a manifold-with-boundary.
  [[nodiscard]] static SurfaceSeed joinedCollars(
      const std::vector<std::shared_ptr<Spacetime>> &surfaces, int layers = 3,
      const std::vector<std::uint64_t> &twist = {});
  /// The collar between two surfaces: the minimal manifold connecting the two
  /// boundaries, \f$ T^2 \times I \f$ over their shared triangulation
  /// (`Spacetime::prismCells` over the base faces with \p layers product
  /// layers), created as one gated whole.
  ///
  /// The surfaces must have identical combinatorics: their vertices in
  /// ascending id order are the base indices, and the two face sets must
  /// coincide under those indices. Layer 0 carries surface A's vertices (host
  /// ids \f$ 0 \ldots n-1 \f$), the last layer surface B's (host ids
  /// \f$ n\cdot\mathrm{layers} \ldots \f$), and intermediate layers are fresh
  /// interior vertices. A's and B's edges carry their surfaces' lengths
  /// verbatim; every other edge carries `Spacetime::autoWiredLength`; every
  /// phase is zero. The whole host is gated once by
  /// `ChainComplex::dualComplexIsValid`, so `bridgePhaseComplete()` holds by
  /// construction on the node that seeds both surfaces as its input blocks.
  ///
  /// The collar must be seeded rather than grown because a per-cell drawing of
  /// the bulk cannot reach \f$ \partial W = T_A \sqcup T_B \f$: a cell the
  /// manifold gate accepts meets the complex along a disk, so a drawing from
  /// one cell stays a ball.
  ///
  /// \p twist relabels surface B's base indices before the two surfaces are
  /// identified: `twist[k]` is the base index of B that meets base index \p k
  /// of A. Empty is the identity, giving the product collar, whose two boundary
  /// surfaces are homologous and whose monodromy is therefore the identity. A
  /// twist makes the collar an \f$I\f$-bundle that is not a product, and the
  /// monodromy becomes the twist's class; it must be a simplicial automorphism
  /// of the shared triangulation. The standard grid torus admits the swap
  /// \f$[[0,1],[1,0]]\f$ (orientation-reversing, propagator
  /// \f$\tau\mapsto1/\tau\f$) but not a Dehn twist, which sends the diagonal to
  /// a step that is no edge. Either way the monodromy is an integer matrix: the
  /// whole-complex harmonic reading realizes \f$GL(2,\mathbb{Z})\f$ and no
  /// more.
  /// @throws std::invalid_argument on a null surface, `layers < 1`, surfaces of
  ///   differing dimension or combinatorics, a surface without top cells, or a
  ///   collar the manifold gate refuses.
  [[nodiscard]] static SurfaceSeed collar(const std::shared_ptr<Spacetime> &surfaceA,
                                              const std::shared_ptr<Spacetime> &surfaceB,
                                              int layers = 1,
                                              const std::vector<std::uint64_t> &twist = {});
  /// Two collars joined by a tube: the far surfaces of two `collar` collars
  /// (surfaces 1 and 3 of the four given, in pairs) are connected by the prism
  /// \f$ t \times [0, \text{layers}] \f$ over one attachment face \f$ t \f$ of
  /// each (`Spacetime::prismCells`), forming a 1-handle between the two
  /// components. The far boundary becomes one surface of genus two — the
  /// connected sum of the two far tori through the tube — while the near tori
  /// are untouched, so
  /// \f$ \partial W = T^2 \sqcup T^2 \sqcup \Sigma_2 \f$,
  /// \f$ b_1(W) = 2 + 2 = 4 \f$ (the tube joins two components and adds no
  /// loop), and \f$ \operatorname{rank}(H^1(W) \to H^1(\partial W)) = 4 =
  /// b_1(\partial W)/2 \f$ with no class invisible to the boundary. No sphere
  /// join is made; the tube is the connection.
  ///
  /// The tube's geometry is Euclidean: the two attachment faces are laid out in
  /// the plane from their own lengths, and ring \f$ \ell \f$ of the tube is the
  /// affine interpolation of the two layouts at
  /// \f$ s = \ell/\text{layers} \f$, scaled about its centroid by `tube.waist`
  /// (the end rings by 1) at height \f$ \ell \cdot \f$ `tube.length`. Every
  /// tube edge gets the Euclidean distance of its endpoints, so the end rings
  /// carry the surfaces' own lengths verbatim. Shrinking the waist or
  /// lengthening the tube pinches the connected sum, which is the separating
  /// degeneration of the genus-two surface.
  ///
  /// Gated once as a whole by `ChainComplex::dualComplexIsValid`. `vertexIds`
  /// comes back in the order the surfaces were given; `tubeRings` lists the
  /// tube's vertex rings.
  /// @throws std::invalid_argument on other than four surfaces, a null one,
  ///   an attachment face that is no face of its surface, a non-positive
  ///   waist or length, fewer than one tube layer, or a join that is not a
  ///   manifold-with-boundary.
  [[nodiscard]] static SurfaceSeed tubedCollars(
      const std::vector<std::shared_ptr<Spacetime>> &surfaces, int layers,
      const std::vector<std::uint64_t> &twist, const TubeSpec &tube);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SURFACE_SEED_H
