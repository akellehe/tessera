// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_MESH_RIEMANNSHEET_H
#define TESSERA_MESH_RIEMANNSHEET_H

#include <complex>
#include <cstdint>

namespace tessera::mesh {

/// \f$ \arg z \in (-\pi, \pi] \f$, with a negative-zero imaginary part normalised
/// away first so that a real radicand never lands on the far side of the cut.
[[nodiscard]] double principalArgument(std::complex<double> z) noexcept;

/// The principal square root, with the same negative-zero normalisation.
[[nodiscard]] std::complex<double> principalSquareRoot(std::complex<double> z) noexcept;

/// The principal inverse cosine, with a real argument pinned to the \f$ +0 \f$
/// side of the cuts \f$ (-\infty, -1] \f$ and \f$ [1, \infty) \f$ — the same
/// convention `Simplex::dihedralAngle` states for its dihedral cosine, so that
/// a sheeted angle started at the principal label reproduces it exactly.
[[nodiscard]] std::complex<double> principalArcCosine(std::complex<double> r) noexcept;

/// # SheetedSqrt
///
/// A square root \f$ \sqrt{z} \f$ carried along a path of radicands together
/// with the Riemann sheet it sits on.
///
/// The square root has a two-sheeted Riemann surface branched over \f$ z = 0 \f$.
/// A value is therefore not fixed by \f$ z \f$ alone: it is fixed by \f$ z \f$
/// and a declared sheet, which this class holds as an integer winding number
/// \f$ w \f$, the number of times the radicand has encircled the origin since
/// the root was declared. The declared argument is
/// \f[ \theta(z) = \operatorname{Arg} z + 2\pi w, \f]
/// and the value is \f$ |z|^{1/2} e^{i\theta/2} = (-1)^w \sqrt{z}_{\text{princ}} \f$,
/// so that the sheet label \f$ w \bmod 2 \f$ is exactly the monodromy: one turn
/// of the radicand around the branch point returns the root with the opposite
/// sign, on the other sheet.
///
/// The value is formed as \f$ (-1)^w \f$ times the principal root rather than
/// from the accumulated argument. The two agree analytically, but the
/// accumulated argument carries the drift of every step taken to reach the
/// current radicand, which on real positive data shows up as a spurious
/// imaginary part; the sign-times-principal form is exact, because \f$ w \f$
/// is an integer and rounding cannot move it except at the cut, where the
/// principal root flips sign at the same moment \f$ w \f$ does.
///
/// `advance` continues the root to a new radicand along the straight step
/// between the two. The step must turn the radicand by less than half a turn —
/// the continuation cannot tell a rotation by \f$ \pi + \delta \f$ from one by
/// \f$ \delta - \pi \f$ — so a caller walking a loop around a branch point must
/// sample it finely enough that consecutive radicands subtend less than
/// \f$ \pi \f$ at the origin. `lastStep` reports the turn each step made, so the
/// caller can check that rather than assume it.
class SheetedSqrt {
 public:
  /// Declare the root at \a radicand on the principal sheet (\f$ w = 0 \f$).
  explicit SheetedSqrt(std::complex<double> radicand = {1.0, 0.0}) noexcept;
  /// Declare the root at \a radicand on the sheet \a winding turns from the
  /// principal one.
  SheetedSqrt(std::complex<double> radicand, int winding) noexcept;

  /// Continue the root to \a radicand, updating the winding by the turn the
  /// step makes about the origin.
  void advance(std::complex<double> radicand) noexcept;

  [[nodiscard]] std::complex<double> radicand() const noexcept { return radicand_; }
  /// \f$ (-1)^w \sqrt{z} \f$ — the root on the declared sheet.
  [[nodiscard]] std::complex<double> value() const noexcept;
  /// \f$ \operatorname{Arg} z + 2\pi w \f$, the argument of the radicand on the
  /// declared sheet. Unbounded: it is the continued quantity, not a principal value.
  [[nodiscard]] double declaredArgument() const noexcept;
  /// The accumulated monodromy \f$ w \f$: signed turns of the radicand about the
  /// branch point since the root was declared.
  [[nodiscard]] int winding() const noexcept { return winding_; }
  /// \f$ w \bmod 2 \in \{0, 1\} \f$: which of the two sheets of \f$ \sqrt{\cdot} \f$
  /// the value sits on. Sheet 0 is the principal one.
  [[nodiscard]] int sheet() const noexcept;
  /// True when the declared sheet is the principal one.
  [[nodiscard]] bool isPrincipal() const noexcept { return sheet() == 0; }
  /// The turn about the origin, in radians, that the last `advance` made. A
  /// magnitude approaching \f$ \pi \f$ means the path was sampled too coarsely
  /// for the continuation to be trusted.
  [[nodiscard]] double lastStep() const noexcept { return lastStep_; }
  /// True when some step put the radicand exactly on the branch point, where no
  /// argument exists and the sheet label is held rather than continued.
  [[nodiscard]] bool touchedBranchPoint() const noexcept { return touchedBranchPoint_; }

 private:
  std::complex<double> radicand_{1.0, 0.0};
  int winding_{0};
  double lastStep_{0.0};
  bool touchedBranchPoint_{false};
};

/// # SheetedAcos
///
/// An inverse cosine \f$ \arccos r \f$ carried along a path of cosines together
/// with the Riemann sheet it sits on.
///
/// The inverse cosine is branched over \f$ r = \pm 1 \f$ and has a logarithmic
/// branch point at infinity, so its Riemann surface has countably many sheets.
/// A sheet is labelled by a pair \f$ (k, \varepsilon) \f$ with
/// \f[ \theta = 2\pi k + \varepsilon \operatorname{Arccos} r, \qquad
///     k \in \mathbb{Z}, \ \varepsilon \in \{+1, -1\}, \f]
/// \f$ \operatorname{Arccos} \f$ the principal value. A loop around \f$ r = +1 \f$
/// flips \f$ \varepsilon \f$ and leaves \f$ k \f$; a loop around \f$ r = -1 \f$
/// flips \f$ \varepsilon \f$ and shifts \f$ k \f$ — the monodromy of the pair is
/// the infinite dihedral group, and both components are carried.
///
/// `advance` continues the angle by taking, among the candidate values on the
/// sheets neighbouring the current one, the one nearest the current value. That
/// is the continuation exactly when the path is sampled finely enough for the
/// true continued value to be the nearest candidate; `lastStep` reports the
/// distance moved so a caller can check it.
class SheetedAcos {
 public:
  /// Declare the angle at \a cosine on the principal sheet
  /// (\f$ k = 0 \f$, \f$ \varepsilon = +1 \f$).
  explicit SheetedAcos(std::complex<double> cosine = {1.0, 0.0}) noexcept;
  /// Declare the angle at \a cosine on the sheet \f$ (k, \varepsilon) \f$.
  SheetedAcos(std::complex<double> cosine, int branchIndex, int orientation) noexcept;

  /// Continue the angle to \a cosine.
  void advance(std::complex<double> cosine) noexcept;

  [[nodiscard]] std::complex<double> cosine() const noexcept { return cosine_; }
  /// \f$ 2\pi k + \varepsilon \operatorname{Arccos} r \f$ — the angle on the
  /// declared sheet.
  [[nodiscard]] std::complex<double> value() const noexcept;
  /// The integer \f$ k \f$ of the declared sheet: the logarithmic winding about
  /// \f$ r = \infty \f$.
  [[nodiscard]] int branchIndex() const noexcept { return branchIndex_; }
  /// The sign \f$ \varepsilon \f$ of the declared sheet: which of the two
  /// arccosine values \f$ \pm \operatorname{Arccos} r \f$ the angle continues.
  [[nodiscard]] int orientation() const noexcept { return orientation_; }
  /// True when the declared sheet is \f$ (0, +1) \f$, the principal one.
  [[nodiscard]] bool isPrincipal() const noexcept {
    return branchIndex_ == 0 && orientation_ == 1;
  }
  /// The distance in the complex plane the angle moved on the last `advance`.
  [[nodiscard]] double lastStep() const noexcept { return lastStep_; }

 private:
  std::complex<double> cosine_{1.0, 0.0};
  int branchIndex_{0};
  int orientation_{1};
  double lastStep_{0.0};
};

}  // namespace tessera::mesh

#endif  // TESSERA_MESH_RIEMANNSHEET_H
