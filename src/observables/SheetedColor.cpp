// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SheetedColor.h"

#include "quantum/GradedFock.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tessera::observables {
namespace {

using ::tessera::cobordism::Certificate;
using ::tessera::cobordism::CertificateDomain;
using ::tessera::cobordism::CertificateRegime;

using cd = std::complex<double>;

/// The Kronecker product kron(left, right) with the left factor slow and the
/// right factor fast — the single index convention of this file.
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& left,
                      const Eigen::MatrixXcd& right) {
  Eigen::MatrixXcd out(left.rows() * right.rows(), left.cols() * right.cols());
  for (Eigen::Index i = 0; i < left.rows(); ++i) {
    for (Eigen::Index j = 0; j < left.cols(); ++j) {
      out.block(i * right.rows(), j * right.cols(), right.rows(),
                right.cols()) = left(i, j) * right;
    }
  }
  return out;
}

/// The identity of the given size as a dense complex matrix.
Eigen::MatrixXcd identity(std::size_t size) {
  return Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(size),
                                    static_cast<Eigen::Index>(size));
}

void requirePositive(double tolerance, const char* what) {
  if (!(tolerance > 0.0)) {
    std::ostringstream message;
    message << what << " must be positive; received " << tolerance << ".";
    throw std::invalid_argument(message.str());
  }
}

}  // namespace

// ─── SheetedSupport ────────────────────────────────────────────────────────

SheetedSupport::SheetedSupport(std::size_t sheetCount,
                               std::size_t baseCellCount)
    : sheetCount_(sheetCount), baseCellCount_(baseCellCount) {
  if (sheetCount_ == 0) {
    throw std::invalid_argument(
        "SheetedSupport: the sheet count must be at least one; a support with "
        "no sheets carries no modes.");
  }
}

std::size_t SheetedSupport::modeIndex(std::size_t baseCell,
                                      std::size_t sheet) const {
  if (baseCell >= baseCellCount_) {
    std::ostringstream message;
    message << "SheetedSupport::modeIndex: base cell " << baseCell
            << " is outside the base complex of " << baseCellCount_
            << " cells.";
    throw std::invalid_argument(message.str());
  }
  if (sheet >= sheetCount_) {
    std::ostringstream message;
    message << "SheetedSupport::modeIndex: sheet " << sheet
            << " is outside the support of " << sheetCount_ << " sheets.";
    throw std::invalid_argument(message.str());
  }
  return baseCell * sheetCount_ + sheet;
}

SheetIsomorphismRead SheetedSupport::certifyIsomorphism(
    const std::vector<Eigen::VectorXcd>& sheetSquaredLengths,
    const std::vector<Eigen::VectorXcd>& sheetConnections,
    double tolerance) const {
  requirePositive(tolerance, "SheetedSupport::certifyIsomorphism tolerance");
  if (sheetSquaredLengths.size() != sheetCount_ ||
      sheetConnections.size() != sheetCount_) {
    std::ostringstream message;
    message << "SheetedSupport::certifyIsomorphism: expected " << sheetCount_
            << " squared-length vectors and " << sheetCount_
            << " connection vectors, one per sheet; received "
            << sheetSquaredLengths.size() << " and " << sheetConnections.size()
            << ".";
    throw std::invalid_argument(message.str());
  }
  const auto baseRows = static_cast<Eigen::Index>(baseCellCount_);
  bool connectionsSupplied = false;
  for (std::size_t s = 0; s < sheetCount_; ++s) {
    if (sheetSquaredLengths[s].size() != baseRows) {
      std::ostringstream message;
      message << "SheetedSupport::certifyIsomorphism: the squared-length "
                 "vector of sheet "
              << s << " has " << sheetSquaredLengths[s].size()
              << " entries; the base complex has " << baseCellCount_
              << " cells.";
      throw std::invalid_argument(message.str());
    }
    if (sheetConnections[s].size() == 0) continue;
    connectionsSupplied = true;
    if (sheetConnections[s].size() != baseRows) {
      std::ostringstream message;
      message << "SheetedSupport::certifyIsomorphism: the connection vector "
                 "of sheet "
              << s << " has " << sheetConnections[s].size()
              << " entries; the base complex has " << baseCellCount_
              << " cells.";
      throw std::invalid_argument(message.str());
    }
  }
  if (connectionsSupplied) {
    for (std::size_t s = 0; s < sheetCount_; ++s) {
      if (sheetConnections[s].size() != baseRows) {
        throw std::invalid_argument(
            "SheetedSupport::certifyIsomorphism: connection values were "
            "supplied for some sheets but not for all of them; supply the "
            "connection on every sheet or on none.");
      }
    }
  }

  SheetIsomorphismRead read;
  read.sheetCount = sheetCount_;
  read.baseCellCount = baseCellCount_;
  const auto worstGap = [](const Eigen::VectorXcd& left,
                           const Eigen::VectorXcd& right) {
    if (left.size() == 0) return 0.0;
    return (left - right).cwiseAbs().maxCoeff();
  };
  for (std::size_t s = 0; s + 1 < sheetCount_; ++s) {
    for (std::size_t t = s + 1; t < sheetCount_; ++t) {
      read.squaredLengthResidual =
          std::max(read.squaredLengthResidual,
                   worstGap(sheetSquaredLengths[s], sheetSquaredLengths[t]));
      if (!connectionsSupplied) continue;
      read.connectionResidual =
          std::max(read.connectionResidual,
                   worstGap(sheetConnections[s], sheetConnections[t]));
    }
  }
  const double residual =
      std::max(read.squaredLengthResidual, read.connectionResidual);
  read.isomorphic = residual <= tolerance;
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      tolerance);
  return read;
}

Eigen::MatrixXcd SheetedSupport::freeOperator(
    const Eigen::MatrixXcd& baseOperator) const {
  const auto baseRows = static_cast<Eigen::Index>(baseCellCount_);
  if (baseOperator.rows() != baseRows || baseOperator.cols() != baseRows) {
    std::ostringstream message;
    message << "SheetedSupport::freeOperator: the base operator is "
            << baseOperator.rows() << "x" << baseOperator.cols()
            << "; the base complex has " << baseCellCount_ << " cells.";
    throw std::invalid_argument(message.str());
  }
  return kron(baseOperator, identity(sheetCount_));
}

Eigen::MatrixXcd SheetedSupport::liftBand(
    const Eigen::MatrixXcd& baseBand) const {
  const auto baseRows = static_cast<Eigen::Index>(baseCellCount_);
  if (baseBand.rows() != baseRows) {
    std::ostringstream message;
    message << "SheetedSupport::liftBand: the base band has "
            << baseBand.rows() << " rows; the base complex has "
            << baseCellCount_ << " cells.";
    throw std::invalid_argument(message.str());
  }
  return kron(baseBand, identity(sheetCount_));
}

Eigen::MatrixXcd SheetedSupport::sheetFrameOperator(
    const Eigen::MatrixXcd& g) const {
  const auto sheets = static_cast<Eigen::Index>(sheetCount_);
  if (g.rows() != sheets || g.cols() != sheets) {
    std::ostringstream message;
    message << "SheetedSupport::sheetFrameOperator: the sheet frame is "
            << g.rows() << "x" << g.cols() << "; the support has "
            << sheetCount_ << " sheets.";
    throw std::invalid_argument(message.str());
  }
  return kron(identity(baseCellCount_), g);
}

Eigen::MatrixXcd SheetedSupport::baseSymmetryOperator(
    const Eigen::MatrixXcd& baseAction) const {
  const auto baseRows = static_cast<Eigen::Index>(baseCellCount_);
  if (baseAction.rows() != baseRows || baseAction.cols() != baseRows) {
    std::ostringstream message;
    message << "SheetedSupport::baseSymmetryOperator: the base action is "
            << baseAction.rows() << "x" << baseAction.cols()
            << "; the base complex has " << baseCellCount_ << " cells.";
    throw std::invalid_argument(message.str());
  }
  return kron(baseAction, identity(sheetCount_));
}

double SheetedSupport::sheetCommutatorResidual(
    const Eigen::MatrixXcd& baseOperator, const Eigen::MatrixXcd& g) const {
  const Eigen::MatrixXcd lifted = baseSymmetryOperator(baseOperator);
  const Eigen::MatrixXcd frame = sheetFrameOperator(g);
  const Eigen::MatrixXcd commutator = lifted * frame - frame * lifted;
  if (commutator.size() == 0) return 0.0;
  return commutator.cwiseAbs().maxCoeff();
}

// ─── SheetFock ─────────────────────────────────────────────────────────────

SheetFock::SheetFock(std::size_t sheetCount) : sheetCount_(sheetCount) {
  if (sheetCount_ == 0) {
    throw std::invalid_argument(
        "SheetFock: the sheet count must be at least one; the exterior "
        "algebra of no sheets carries no sectors.");
  }
  if (sheetCount_ > ::tessera::quantum::ExteriorAlgebra::kMaxMatrixModes) {
    std::ostringstream message;
    message << "SheetFock: the sheet count " << sheetCount_
            << " exceeds the matrix layer's limit of "
            << ::tessera::quantum::ExteriorAlgebra::kMaxMatrixModes
            << " modes.";
    throw std::invalid_argument(message.str());
  }
  dimension_ = std::size_t{1} << sheetCount_;
}

void SheetFock::validateSheet(std::size_t sheet) const {
  if (sheet >= sheetCount_) {
    std::ostringstream message;
    message << "SheetFock: sheet " << sheet << " is outside the "
            << sheetCount_ << "-sheet space.";
    throw std::invalid_argument(message.str());
  }
}

std::vector<SheetSector> SheetFock::sectors() const {
  static const char* kRankThreeNames[4] = {
      "scalar vacuum", "fundamental E", "det E (x) E-dual",
      "determinant line"};
  std::vector<SheetSector> out;
  out.reserve(sheetCount_ + 1);
  std::size_t binomial = 1;
  for (std::size_t n = 0; n <= sheetCount_; ++n) {
    SheetSector sector;
    sector.occupation = n;
    sector.dimension = binomial;
    sector.fermionParity = (n % 2 == 0) ? +1 : -1;
    if (sheetCount_ == 3) {
      sector.representation = kRankThreeNames[n];
    } else {
      std::ostringstream name;
      name << "exterior power " << n;
      sector.representation = name.str();
    }
    out.push_back(sector);
    // binomial(k, n + 1) from binomial(k, n), exactly in integer arithmetic.
    if (n < sheetCount_) binomial = binomial * (sheetCount_ - n) / (n + 1);
  }
  return out;
}

Eigen::MatrixXcd SheetFock::sectorProjector(std::size_t occupation) const {
  const ::tessera::quantum::ExteriorAlgebra algebra(sheetCount_);
  return Eigen::MatrixXcd(algebra.sectorProjector(occupation));
}

Eigen::MatrixXcd SheetFock::exteriorCreation(std::size_t sheet) const {
  validateSheet(sheet);
  const ::tessera::quantum::ExteriorAlgebra algebra(sheetCount_);
  return Eigen::MatrixXcd(algebra.creationMatrix(sheet));
}

Eigen::MatrixXcd SheetFock::contraction(std::size_t sheet) const {
  validateSheet(sheet);
  const ::tessera::quantum::ExteriorAlgebra algebra(sheetCount_);
  return Eigen::MatrixXcd(algebra.annihilationMatrix(sheet));
}

Eigen::MatrixXcd SheetFock::sheetBilinear(std::size_t i, std::size_t j) const {
  validateSheet(i);
  validateSheet(j);
  const ::tessera::quantum::ExteriorAlgebra algebra(sheetCount_);
  return Eigen::MatrixXcd(algebra.creationMatrix(i) *
                          algebra.annihilationMatrix(j));
}

double SheetFock::commutatorResidual() const {
  double worst = 0.0;
  for (std::size_t i = 0; i < sheetCount_; ++i) {
    for (std::size_t j = 0; j < sheetCount_; ++j) {
      const Eigen::MatrixXcd eij = sheetBilinear(i, j);
      for (std::size_t k = 0; k < sheetCount_; ++k) {
        for (std::size_t l = 0; l < sheetCount_; ++l) {
          const Eigen::MatrixXcd ekl = sheetBilinear(k, l);
          Eigen::MatrixXcd expected =
              Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(dimension_),
                                     static_cast<Eigen::Index>(dimension_));
          if (j == k) expected += sheetBilinear(i, l);
          if (i == l) expected -= sheetBilinear(k, j);
          const Eigen::MatrixXcd defect = eij * ekl - ekl * eij - expected;
          worst = std::max(worst, defect.cwiseAbs().maxCoeff());
        }
      }
    }
  }
  return worst;
}

double SheetFock::sectorAgreementResidual() const {
  if (sheetCount_ != 3) {
    throw std::logic_error(
        "SheetFock::sectorAgreementResidual: the ColorFiber sectors describe "
        "the exterior algebra of three sheets only; this space has a "
        "different sheet count.");
  }
  double worst = 0.0;
  for (std::size_t n = 0; n <= 3; ++n) {
    const Eigen::MatrixXcd mine = sectorProjector(n);
    const Eigen::MatrixXcd theirs = ColorFiber::sectorProjector(n);
    worst = std::max(worst, (mine - theirs).cwiseAbs().maxCoeff());
  }
  return worst;
}

// ─── SheetAttachment ───────────────────────────────────────────────────────

void SheetAttachment::validateSquare(const Eigen::MatrixXcd& m,
                                     const char* what) {
  if (m.rows() != m.cols() || m.rows() == 0) {
    std::ostringstream message;
    message << "SheetAttachment: " << what << " is " << m.rows() << "x"
            << m.cols() << "; a colour transport is a nonempty square matrix.";
    throw std::invalid_argument(message.str());
  }
}

AttachmentRead SheetAttachment::attachmentMatrix(
    std::size_t sheetCount, const std::vector<ConnectingSimplex>& simplices,
    double fullRankTolerance) {
  requirePositive(fullRankTolerance,
                  "SheetAttachment::attachmentMatrix fullRankTolerance");
  if (sheetCount == 0) {
    throw std::invalid_argument(
        "SheetAttachment::attachmentMatrix: the sheet count must be at least "
        "one.");
  }
  const auto k = static_cast<Eigen::Index>(sheetCount);
  AttachmentRead read;
  read.matrix = Eigen::MatrixXcd::Zero(k, k);
  read.simplexCount = simplices.size();
  for (const ConnectingSimplex& simplex : simplices) {
    if (simplex.sheetA >= sheetCount || simplex.sheetB >= sheetCount) {
      std::ostringstream message;
      message << "SheetAttachment::attachmentMatrix: a connecting simplex "
                 "names sheet "
              << simplex.sheetA << " of A and sheet " << simplex.sheetB
              << " of B; the supports carry " << sheetCount << " sheets.";
      throw std::invalid_argument(message.str());
    }
    if (simplex.sheetA != simplex.sheetB) read.sheetDiagonal = false;
    read.matrix(static_cast<Eigen::Index>(simplex.sheetA),
                static_cast<Eigen::Index>(simplex.sheetB)) += simplex.weight;
  }
  read.determinant = read.matrix.determinant();

  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(read.matrix);
  const Eigen::VectorXd singular = svd.singularValues();
  read.minSingularValue = singular.size() == 0 ? 0.0 : singular.minCoeff();
  const double maxSingular = singular.size() == 0 ? 0.0 : singular.maxCoeff();
  read.conditioning = read.minSingularValue > 0.0
                          ? maxSingular / read.minSingularValue
                          : std::numeric_limits<double>::infinity();
  // The accumulation is exact given the verified premise that every
  // connecting simplex named a sheet at each end, so a full-rank attachment
  // is graded StructureExact at residual zero. A rank-dropping one certifies
  // nothing: the frame law still applies to it, but it is not an element of
  // GL(k, C) and no transport it composes into is invertible.
  read.certificate =
      read.minSingularValue > fullRankTolerance
          ? Certificate::structureExact(CertificateDomain::Static,
                                        CertificateRegime::NonNormal, 0.0,
                                        read.conditioning, fullRankTolerance)
          : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                            CertificateRegime::NonNormal);
  return read;
}

Eigen::MatrixXcd SheetAttachment::frameChanged(
    const Eigen::MatrixXcd& attachment, const Eigen::MatrixXcd& frameA,
    const Eigen::MatrixXcd& frameB) {
  validateSquare(attachment, "the attachment matrix");
  validateSquare(frameA, "the frame at A");
  validateSquare(frameB, "the frame at B");
  if (frameA.rows() != attachment.rows() ||
      frameB.rows() != attachment.cols()) {
    std::ostringstream message;
    message << "SheetAttachment::frameChanged: the attachment matrix is "
            << attachment.rows() << "x" << attachment.cols()
            << " but the frames are " << frameA.rows() << "x" << frameA.cols()
            << " at A and " << frameB.rows() << "x" << frameB.cols()
            << " at B.";
    throw std::invalid_argument(message.str());
  }
  const Eigen::FullPivLU<Eigen::MatrixXcd> lu(frameA);
  if (!lu.isInvertible()) {
    throw std::invalid_argument(
        "SheetAttachment::frameChanged: the frame at A is singular, so it is "
        "not a sheet relabeling in GL(k, C).");
  }
  return lu.inverse() * attachment * frameB;
}

Eigen::MatrixXcd SheetAttachment::couplingBlock(
    const Eigen::MatrixXcd& baseCoupling,
    const Eigen::MatrixXcd& attachment) {
  if (baseCoupling.size() == 0 || attachment.size() == 0) {
    throw std::invalid_argument(
        "SheetAttachment::couplingBlock: both the base coupling block and the "
        "attachment matrix must be nonempty.");
  }
  return kron(baseCoupling, attachment);
}

Eigen::MatrixXcd SheetAttachment::compose(
    const std::vector<Eigen::MatrixXcd>& path, std::size_t sheetCount) {
  if (path.empty()) {
    if (sheetCount == 0) {
      throw std::invalid_argument(
          "SheetAttachment::compose: an empty path has no shape of its own, "
          "so the sheet count of the identity it returns must be supplied.");
    }
    return identity(sheetCount);
  }
  validateSquare(path.front(), "a path factor");
  Eigen::MatrixXcd accumulated = path.front();
  for (std::size_t i = 1; i < path.size(); ++i) {
    validateSquare(path[i], "a path factor");
    if (path[i].cols() != accumulated.rows()) {
      std::ostringstream message;
      message << "SheetAttachment::compose: path factor " << i << " is "
              << path[i].rows() << "x" << path[i].cols()
              << " and cannot be applied after a " << accumulated.rows() << "x"
              << accumulated.cols() << " partial transport.";
      throw std::invalid_argument(message.str());
    }
    accumulated = path[i] * accumulated;
  }
  return accumulated;
}

HolonomyInvariants SheetAttachment::holonomy(
    const std::vector<Eigen::MatrixXcd>& links) {
  if (links.empty()) {
    throw std::invalid_argument(
        "SheetAttachment::holonomy: a closed sequence needs at least one "
        "link.");
  }
  HolonomyInvariants out;
  out.holonomy = compose(links);
  out.linkCount = links.size();
  out.determinant = out.holonomy.determinant();

  const auto k = static_cast<std::size_t>(out.holonomy.rows());
  Eigen::MatrixXcd power = Eigen::MatrixXcd::Identity(out.holonomy.rows(),
                                                      out.holonomy.cols());
  out.powerTraces.reserve(k);
  for (std::size_t j = 1; j <= k; ++j) {
    power = power * out.holonomy;
    out.powerTraces.push_back(power.trace());
  }

  // Newton's identities: with e_0 = 1 and p_j the power traces,
  //   j e_j = sum_{i=1}^{j} (-1)^{i-1} e_{j-i} p_i,
  // and det(lambda I - H) = sum_j (-1)^j e_j lambda^{k-j}. The stored
  // coefficients are in ascending powers of lambda, so coefficient m is
  // (-1)^{k-m} e_{k-m}.
  std::vector<cd> elementary(k + 1, cd(0.0, 0.0));
  elementary[0] = cd(1.0, 0.0);
  for (std::size_t j = 1; j <= k; ++j) {
    cd sum(0.0, 0.0);
    for (std::size_t i = 1; i <= j; ++i) {
      const double sign = (i % 2 == 1) ? 1.0 : -1.0;
      sum += sign * elementary[j - i] * out.powerTraces[i - 1];
    }
    elementary[j] = sum / static_cast<double>(j);
  }
  out.characteristicPolynomial.assign(k + 1, cd(0.0, 0.0));
  for (std::size_t m = 0; m <= k; ++m) {
    const std::size_t j = k - m;
    const double sign = (j % 2 == 0) ? 1.0 : -1.0;
    out.characteristicPolynomial[m] = sign * elementary[j];
  }
  return out;
}

// ─── ColorSinglet ──────────────────────────────────────────────────────────

namespace {

/// The shared core of `amplitude` and `frameCovarianceResidual`: validate the
/// three transports and three colour vectors, then form the transported
/// column matrix c-hat_X = S_pX c_X.
Eigen::MatrixXcd transportedColumns(
    const std::vector<Eigen::MatrixXcd>& transports,
    const std::vector<Eigen::VectorXcd>& colors) {
  if (transports.size() != 3 || colors.size() != 3) {
    std::ostringstream message;
    message << "ColorSinglet: the common-frame colour amplitude is the wedge "
               "of exactly three quark colour vectors; received "
            << colors.size() << " colour vectors and " << transports.size()
            << " transports.";
    throw std::invalid_argument(message.str());
  }
  Eigen::MatrixXcd columns(3, 3);
  for (std::size_t x = 0; x < 3; ++x) {
    if (transports[x].rows() != 3 || transports[x].cols() != 3) {
      std::ostringstream message;
      message << "ColorSinglet: colour transport " << x << " is "
              << transports[x].rows() << "x" << transports[x].cols()
              << "; the adopted quark support has three sheets, so every "
                 "colour transport is 3x3.";
      throw std::invalid_argument(message.str());
    }
    if (colors[x].size() != 3) {
      std::ostringstream message;
      message << "ColorSinglet: colour vector " << x << " has "
              << colors[x].size()
              << " entries; the adopted quark support has three sheets.";
      throw std::invalid_argument(message.str());
    }
    columns.col(static_cast<Eigen::Index>(x)) = transports[x] * colors[x];
  }
  return columns;
}

}  // namespace

ColorSingletRead ColorSinglet::amplitude(
    Complex trivialization, const std::vector<Eigen::MatrixXcd>& transports,
    const std::vector<Eigen::VectorXcd>& colors, double tolerance) {
  requirePositive(tolerance, "ColorSinglet::amplitude tolerance");
  ColorSingletRead read;
  read.transportedColumns = transportedColumns(transports, colors);
  read.transportedWedge = read.transportedColumns.determinant();
  read.amplitude = trivialization * read.transportedWedge;
  read.magnitude = std::abs(read.amplitude);
  read.nonvanishing = read.magnitude > tolerance;

  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(read.transportedColumns);
  const Eigen::VectorXd singular = svd.singularValues();
  read.minSingularValue = singular.size() == 0 ? 0.0 : singular.minCoeff();
  // The amplitude is a determinant of supplied numbers, so it is exact to
  // rounding; what the certificate grades is the nonvanishing premise. A
  // wedge that vanished at the declared tolerance certifies nothing, because
  // the singlet condition the whitepaper states is exactly that it does not
  // vanish.
  read.certificate =
      read.nonvanishing
          ? Certificate::algebraicallyExact(CertificateDomain::Static,
                                            CertificateRegime::NonNormal, 0.0,
                                            tolerance)
          : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                            CertificateRegime::NonNormal);
  return read;
}

double ColorSinglet::frameCovarianceResidual(
    Complex trivialization, const std::vector<Eigen::MatrixXcd>& transports,
    const std::vector<Eigen::VectorXcd>& colors,
    const Eigen::MatrixXcd& baseFrame,
    const std::vector<Eigen::MatrixXcd>& frames) {
  if (frames.size() != 3) {
    std::ostringstream message;
    message << "ColorSinglet::frameCovarianceResidual: expected three sheet "
               "frames, one per quark; received "
            << frames.size() << ".";
    throw std::invalid_argument(message.str());
  }
  // Validate the transports and colour vectors before any of them is used,
  // through the same shared core the amplitude uses.
  (void)transportedColumns(transports, colors);
  if (baseFrame.rows() != 3 || baseFrame.cols() != 3) {
    std::ostringstream message;
    message << "ColorSinglet::frameCovarianceResidual: the base-cluster frame "
               "is "
            << baseFrame.rows() << "x" << baseFrame.cols()
            << "; the adopted quark support has three sheets.";
    throw std::invalid_argument(message.str());
  }
  const Eigen::FullPivLU<Eigen::MatrixXcd> baseLu(baseFrame);
  if (!baseLu.isInvertible()) {
    throw std::invalid_argument(
        "ColorSinglet::frameCovarianceResidual: the base-cluster frame is "
        "singular, so it is not a sheet relabeling in GL(3, C).");
  }
  const Eigen::MatrixXcd baseInverse = baseLu.inverse();

  std::vector<Eigen::MatrixXcd> changedTransports;
  std::vector<Eigen::VectorXcd> changedColors;
  changedTransports.reserve(3);
  changedColors.reserve(3);
  for (std::size_t x = 0; x < 3; ++x) {
    if (frames[x].rows() != 3 || frames[x].cols() != 3) {
      std::ostringstream message;
      message << "ColorSinglet::frameCovarianceResidual: sheet frame " << x
              << " is " << frames[x].rows() << "x" << frames[x].cols()
              << "; the adopted quark support has three sheets.";
      throw std::invalid_argument(message.str());
    }
    const Eigen::FullPivLU<Eigen::MatrixXcd> lu(frames[x]);
    if (!lu.isInvertible()) {
      std::ostringstream message;
      message << "ColorSinglet::frameCovarianceResidual: sheet frame " << x
              << " is singular, so it is not a sheet relabeling in GL(3, C).";
      throw std::invalid_argument(message.str());
    }
    changedTransports.push_back(baseInverse * transports[x] * frames[x]);
    changedColors.push_back(Eigen::VectorXcd(lu.inverse() * colors[x]));
  }
  const Complex changedTrivialization =
      trivialization * baseFrame.determinant();

  const ColorSingletRead original =
      amplitude(trivialization, transports, colors);
  const ColorSingletRead changed =
      amplitude(changedTrivialization, changedTransports, changedColors);
  return std::abs(changed.amplitude - original.amplitude);
}

}  // namespace tessera::observables
