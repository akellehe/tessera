// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Implementation of the complex fibre transport declared in
/// include/observables/ComplexTransport.h: the general-linear fibre map, the
/// leakage measured before restriction, the reversal by transposition, the
/// Kato parallel transport of an isolated band, and the interferometric
/// exchange character of two general-linear holonomies.

#include "observables/ComplexTransport.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

namespace tessera::observables {
namespace {

using ::tessera::cobordism::Certificate;
using ::tessera::cobordism::CertificateDomain;
using ::tessera::cobordism::CertificateRegime;

using cd = std::complex<double>;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The spectral norm ||A||_2 of a small dense matrix: its largest singular
/// value, and zero for an empty matrix.
double spectralNorm(const Eigen::MatrixXcd& a) {
  if (a.size() == 0) return 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(a);
  return svd.singularValues().size() > 0 ? svd.singularValues()[0] : 0.0;
}

/// The identity of the given size as a dense complex matrix.
Eigen::MatrixXcd identity(Eigen::Index size) {
  return Eigen::MatrixXcd::Identity(size, size);
}

/// The band's isolation: its separation in the complex plane from the nearest
/// eigenvalue outside it. When that separation was not measured the read falls
/// back to the smaller of the two sort-order gaps, propagating an unknown side
/// as a NaN rather than reporting an unmeasured band as infinitely isolated.
double isolationGap(const SpectralBandCertificate& c) {
  if (!std::isnan(c.nearestDiscardedSeparation))
    return c.nearestDiscardedSeparation;
  if (std::isnan(c.lowerGap) || std::isnan(c.upperGap)) return kNaN;
  return std::min(c.lowerGap, c.upperGap);
}

/// The paired regime of two endpoint bands: NonNormal dominates, then the
/// bilinear pencil, then the Hermitian indefinite regime, else the shared
/// positive one.
CertificateRegime pairedRegime(CertificateRegime a, CertificateRegime b) {
  if (a == CertificateRegime::NonNormal || b == CertificateRegime::NonNormal)
    return CertificateRegime::NonNormal;
  if (a == CertificateRegime::ComplexSymmetricPencil ||
      b == CertificateRegime::ComplexSymmetricPencil)
    return CertificateRegime::ComplexSymmetricPencil;
  if (a == CertificateRegime::HermitianIndefinite ||
      b == CertificateRegime::HermitianIndefinite)
    return CertificateRegime::HermitianIndefinite;
  return CertificateRegime::PositiveSemidefinite;
}

/// Refuse a matrix that is not square and non-empty, naming it.
void requireSquare(const Eigen::MatrixXcd& m, const char* what) {
  if (m.rows() == 0 || m.rows() != m.cols()) {
    std::ostringstream message;
    message << what << " must be a non-empty square matrix; received " << m.rows()
            << " x " << m.cols() << ".";
    throw std::invalid_argument(message.str());
  }
}

/// The largest of two doubles, treating a NaN as "not measured" so that a
/// measured value always wins over an unmeasured one.
double maxMeasured(double a, double b) {
  if (std::isnan(a)) return b;
  if (std::isnan(b)) return a;
  return std::max(a, b);
}

}  // namespace

// ─── the fibre map ─────────────────────────────────────────────────────────

Eigen::MatrixXcd ComplexTransport::fiberMap(
    const Eigen::MatrixXcd& dualFrameTo, const Eigen::MatrixXcd& transfer,
    const Eigen::MatrixXcd& rightFrameFrom) {
  if (dualFrameTo.rows() != transfer.rows()) {
    std::ostringstream message;
    message << "ComplexTransport::fiberMap: the destination dual frame has "
            << dualFrameTo.rows() << " rows and the transfer has "
            << transfer.rows()
            << "; both count the destination band's cells.";
    throw std::invalid_argument(message.str());
  }
  if (transfer.cols() != rightFrameFrom.rows()) {
    std::ostringstream message;
    message << "ComplexTransport::fiberMap: the transfer has " << transfer.cols()
            << " columns and the source right frame has "
            << rightFrameFrom.rows()
            << " rows; both count the source band's cells.";
    throw std::invalid_argument(message.str());
  }
  return dualFrameTo.transpose() * transfer * rightFrameFrom;
}

double ComplexTransport::leakage(const Eigen::MatrixXcd& projectorTo,
                                 const Eigen::MatrixXcd& transfer,
                                 const Eigen::MatrixXcd& projectorFrom) {
  requireSquare(projectorTo, "ComplexTransport::leakage destination projector");
  requireSquare(projectorFrom, "ComplexTransport::leakage source projector");
  if (projectorTo.rows() != transfer.rows() ||
      projectorFrom.rows() != transfer.cols()) {
    std::ostringstream message;
    message << "ComplexTransport::leakage: the transfer is " << transfer.rows()
            << " x " << transfer.cols() << " and the projectors are "
            << projectorTo.rows() << " x " << projectorTo.cols() << " and "
            << projectorFrom.rows() << " x " << projectorFrom.cols()
            << "; the destination projector must match the transfer's rows and "
               "the source projector its columns.";
    throw std::invalid_argument(message.str());
  }
  const Eigen::MatrixXcd escaping =
      (identity(transfer.rows()) - projectorTo) * transfer * projectorFrom;
  return spectralNorm(escaping);
}

GeneralLinearTransportRead ComplexTransport::transport(
    const SpectralFiber& to, const SpectralFiber& from,
    const Eigen::MatrixXcd& transfer, const ComplexTransportConfig& cfg) {
  const SpectralBandCertificate& certTo = to.certificate();
  const SpectralBandCertificate& certFrom = from.certificate();
  if (certTo.degree != certFrom.degree) {
    std::ostringstream message;
    message << "ComplexTransport::transport: the destination band sits at "
               "degree "
            << certTo.degree << " and the source band at " << certFrom.degree
            << "; a transport joins two bands of one degree.";
    throw std::invalid_argument(message.str());
  }
  const auto rowsNeeded = static_cast<Eigen::Index>(to.cellVertices().size());
  const auto colsNeeded = static_cast<Eigen::Index>(from.cellVertices().size());
  if (transfer.rows() != rowsNeeded || transfer.cols() != colsNeeded) {
    std::ostringstream message;
    message << "ComplexTransport::transport: the transfer is " << transfer.rows()
            << " x " << transfer.cols() << "; the destination band has "
            << rowsNeeded << " cells (the rows) and the source band has "
            << colsNeeded << " cells (the columns).";
    throw std::invalid_argument(message.str());
  }

  GeneralLinearTransportRead read;
  read.degree = certTo.degree;
  read.rank = static_cast<int>(to.rank());
  read.regime = pairedRegime(certTo.certificate.regime(),
                             certFrom.certificate.regime());
  read.toIsolation = isolationGap(certTo);
  read.fromIsolation = isolationGap(certFrom);
  read.toProjectorNorm = certTo.projectorNorm;
  read.fromProjectorNorm = certFrom.projectorNorm;
  read.toRightFrameResidual = certTo.eigenResidual;
  read.toLeftFrameResidual = certTo.leftResidual;
  read.fromRightFrameResidual = certFrom.eigenResidual;
  read.fromLeftFrameResidual = certFrom.leftResidual;
  read.toResolventBound =
      read.toIsolation > 0.0 ? certTo.projectorNorm / read.toIsolation : kInf;
  read.fromResolventBound = read.fromIsolation > 0.0
                                ? certFrom.projectorNorm / read.fromIsolation
                                : kInf;

  // The fibre map itself: the bilinear pairing of the destination's transpose
  // dual frame with the source's right frame. No polar factor, no compact real
  // form and no determinant root is applied to it anywhere below.
  const Eigen::MatrixXcd m =
      fiberMap(to.dualFrame(), transfer, from.rightFrame());
  read.map = m;

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(m);
  const Eigen::VectorXd& sigma = svd.singularValues();
  read.singularValues.assign(sigma.data(), sigma.data() + sigma.size());
  const double sigmaMax = sigma.size() > 0 ? sigma[0] : 0.0;
  const double sigmaMin = sigma.size() > 0 ? sigma[sigma.size() - 1] : 0.0;
  read.minSingularValue = sigma.size() > 0 ? sigmaMin : kNaN;
  int numericalRank = 0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i)
    if (sigma[i] > cfg.rankTolerance * sigmaMax) ++numericalRank;
  if (sigmaMax == 0.0) numericalRank = 0;
  read.numericalRank = numericalRank;
  read.conditionNumber =
      sigma.size() == 0 ? kNaN : (sigmaMin > 0.0 ? sigmaMax / sigmaMin : kInf);
  read.invertible =
      m.rows() == m.cols() && m.size() > 0 && numericalRank == m.rows();
  if (read.invertible) read.determinant = m.determinant();

  // The leakage is the whitepaper's quantity, measured on the transfer before
  // the restriction to the bands: the part that starts inside the source band
  // and lands outside the destination band.
  const Eigen::MatrixXcd projectorTo = to.projector();
  const Eigen::MatrixXcd projectorFrom = from.projector();
  read.leakage = leakage(projectorTo, transfer, projectorFrom);
  const double restrictedScale = spectralNorm(transfer * projectorFrom);
  read.relativeLeakage =
      restrictedScale > 0.0 ? read.leakage / restrictedScale : 0.0;

  const auto reject = [&read](const std::string& reason) {
    read.accepted = false;
    read.rejectionReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, read.regime);
    return read;
  };

  if (static_cast<int>(to.rank()) != static_cast<int>(from.rank()))
    return reject(
        "the two bands have different ranks, and a fibre map is defined only "
        "between matched frames of one common rank");
  if (cfg.requireCertifiedFibers && !(certTo.accepted && certFrom.accepted))
    return reject(
        "an endpoint band is uncertified, so the isolated bundle the transport "
        "lives in does not exist");
  if (cfg.isolationFloor > 0.0 &&
      !(read.toIsolation >= cfg.isolationFloor &&
        read.fromIsolation >= cfg.isolationFloor))
    return reject("an endpoint band's isolation is below the declared floor");
  if (!(certTo.projectorNorm <= cfg.conditionNumberCap) ||
      !(certFrom.projectorNorm <= cfg.conditionNumberCap))
    return reject(
        "an endpoint band's projector norm is above the conditioning cap");
  if (read.numericalRank < read.rank)
    return reject(
        "the fibre map is rank deficient, so it is not an element of "
        "GL(r, C)");
  if (!(read.conditionNumber <= cfg.conditionNumberCap))
    return reject("the fibre map's condition number is above the cap");
  if (!(read.relativeLeakage <= cfg.leakageTolerance))
    return reject(
        "the transfer leaks out of the destination band by more than the "
        "declared tolerance");

  read.accepted = true;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, read.regime, read.relativeLeakage,
      read.conditionNumber, cfg.leakageTolerance);
  return read;
}

std::string GeneralLinearTransportRead::describe() const {
  std::ostringstream out;
  out << "GL(" << rank << ", C) transport at degree " << degree
      << ": det = " << determinant.real()
      << (determinant.imag() < 0.0 ? " - " : " + ") << std::abs(determinant.imag())
      << "i, condition number " << conditionNumber << ", relative leakage "
      << relativeLeakage;
  if (accepted) {
    out << ", accepted";
  } else {
    out << ", rejected (" << rejectionReason << ")";
  }
  return out.str();
}

Eigen::MatrixXcd ComplexTransport::frameChanged(
    const Eigen::MatrixXcd& map, const Eigen::MatrixXcd& frameTo,
    const Eigen::MatrixXcd& frameFrom) {
  return SheetAttachment::frameChanged(map, frameTo, frameFrom);
}

Eigen::MatrixXcd ComplexTransport::reversedTransfer(
    const Eigen::MatrixXcd& transfer) {
  return transfer.transpose();
}

Eigen::MatrixXcd ComplexTransport::dualTransport(const Eigen::MatrixXcd& map) {
  requireSquare(map, "ComplexTransport::dualTransport map");
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(map.transpose());
  if (!lu.isInvertible()) {
    throw std::invalid_argument(
        "ComplexTransport::dualTransport: the map is singular, so it has no "
        "dual transport; a singular transport is the one analytic failure mode "
        "of the construction and is reported rather than inverted.");
  }
  return lu.inverse();
}

Eigen::MatrixXcd ComplexTransport::compose(
    const std::vector<Eigen::MatrixXcd>& path, std::size_t rank) {
  return SheetAttachment::compose(path, rank);
}

HolonomyInvariants ComplexTransport::holonomy(
    const std::vector<Eigen::MatrixXcd>& links) {
  return SheetAttachment::holonomy(links);
}

// ─── Kato transport ────────────────────────────────────────────────────────

Eigen::MatrixXcd ComplexTransport::katoGenerator(
    const Eigen::MatrixXcd& projectorRate, const Eigen::MatrixXcd& projector) {
  requireSquare(projectorRate, "ComplexTransport::katoGenerator projector rate");
  requireSquare(projector, "ComplexTransport::katoGenerator projector");
  if (projectorRate.rows() != projector.rows()) {
    std::ostringstream message;
    message << "ComplexTransport::katoGenerator: the projector rate is "
            << projectorRate.rows() << " x " << projectorRate.cols()
            << " and the projector is " << projector.rows() << " x "
            << projector.cols() << "; the two live on one space.";
    throw std::invalid_argument(message.str());
  }
  return projectorRate * projector - projector * projectorRate;
}

Eigen::MatrixXcd ComplexTransport::katoStep(const Eigen::MatrixXcd& from,
                                            const Eigen::MatrixXcd& to,
                                            KatoScheme scheme) {
  requireSquare(from, "ComplexTransport::katoStep source projector");
  requireSquare(to, "ComplexTransport::katoStep destination projector");
  if (from.rows() != to.rows()) {
    std::ostringstream message;
    message << "ComplexTransport::katoStep: the source projector is "
            << from.rows() << " x " << from.cols()
            << " and the destination projector is " << to.rows() << " x "
            << to.cols() << "; the two live on one space.";
    throw std::invalid_argument(message.str());
  }
  const Eigen::MatrixXcd id = identity(from.rows());
  // The unnormalized intertwiner. R P0 = P1 R holds exactly for any pair of
  // idempotents: R P0 = P1 P0 P0 + (I - P1)(I - P0) P0 = P1 P0, and
  // P1 R = P1 P1 P0 + P1 (I - P1)(I - P0) = P1 P0.
  const Eigen::MatrixXcd intertwiner = to * from + (id - to) * (id - from);
  switch (scheme) {
    case KatoScheme::Intertwiner:
      return intertwiner;
    case KatoScheme::ExponentialGenerator: {
      const Eigen::MatrixXcd generator = to * from - from * to;
      return generator.exp();
    }
    case KatoScheme::DirectRotation:
    default: {
      const Eigen::MatrixXcd difference = to - from;
      const double separation = spectralNorm(difference);
      if (!(separation < 1.0)) {
        std::ostringstream message;
        message << "ComplexTransport::katoStep: the two projectors are at "
                   "operator-norm distance "
                << separation
                << ", which is not below one, so the geodesic joining them in "
                   "the Grassmannian is not unique and no direct rotation "
                   "exists; sample the path more finely, or name the "
                   "Intertwiner scheme, which is exact for any pair of "
                   "idempotents.";
        throw std::invalid_argument(message.str());
      }
      const Eigen::MatrixXcd gram = id - difference * difference;
      const Eigen::MatrixXcd root = gram.sqrt();
      if (!root.allFinite() ||
          spectralNorm(root * root - gram) > 1e-8 * std::max(1.0, spectralNorm(gram))) {
        throw std::invalid_argument(
            "ComplexTransport::katoStep: the principal square root of "
            "I - (P1 - P0)^2 did not reproduce its argument, so the direct "
            "rotation is not defined for this pair of projectors; name the "
            "Intertwiner scheme, which is exact for any pair of idempotents.");
      }
      Eigen::FullPivLU<Eigen::MatrixXcd> lu(root);
      if (!lu.isInvertible()) {
        throw std::invalid_argument(
            "ComplexTransport::katoStep: the direct rotation's normalizing "
            "factor I - (P1 - P0)^2 is singular, so the two projectors are at "
            "operator-norm distance one and no direct rotation joins them.");
      }
      return lu.inverse() * intertwiner;
    }
  }
}

KatoTransportRead ComplexTransport::katoTransport(
    const std::vector<Eigen::MatrixXcd>& projectors, KatoScheme scheme,
    const ComplexTransportConfig& cfg) {
  if (projectors.size() < 2) {
    std::ostringstream message;
    message << "ComplexTransport::katoTransport: a transport needs at least "
               "two sampled projectors; received "
            << projectors.size() << ".";
    throw std::invalid_argument(message.str());
  }
  const Eigen::Index dimension = projectors.front().rows();
  for (std::size_t t = 0; t < projectors.size(); ++t) {
    if (projectors[t].rows() != dimension ||
        projectors[t].cols() != dimension || dimension == 0) {
      std::ostringstream message;
      message << "ComplexTransport::katoTransport: projector " << t << " is "
              << projectors[t].rows() << " x " << projectors[t].cols()
              << "; every projector on the path must be a non-empty square "
                 "matrix of the first one's size, "
              << dimension << " x " << dimension << ".";
      throw std::invalid_argument(message.str());
    }
  }

  KatoTransportRead read;
  read.scheme = scheme;
  read.steps = projectors.size() - 1;
  read.dimension = static_cast<std::size_t>(dimension);

  const cd traceFirst = projectors.front().trace();
  read.rank = static_cast<std::size_t>(std::llround(traceFirst.real()));
  double rankDefect = 0.0;
  double idempotency = 0.0;
  for (const Eigen::MatrixXcd& p : projectors) {
    rankDefect = std::max(
        rankDefect, std::abs(p.trace() - cd(static_cast<double>(read.rank), 0.0)));
    idempotency = std::max(idempotency, spectralNorm(p * p - p));
  }
  read.rankDefect = rankDefect;
  read.idempotencyResidual = idempotency;

  double maxStep = 0.0;
  for (std::size_t t = 0; t + 1 < projectors.size(); ++t)
    maxStep = std::max(maxStep, spectralNorm(projectors[t + 1] - projectors[t]));
  read.maxProjectorStep = maxStep;

  const auto invalidate = [&read](const std::string& reason) {
    read.transport = Eigen::MatrixXcd();
    read.stepTransports.clear();
    read.intertwiningResidual = kNaN;
    read.composedIntertwiningResidual = kNaN;
    read.complete = false;
    read.invalidReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::NonNormal);
    return read;
  };

  Eigen::MatrixXcd composed = identity(dimension);
  double intertwining = 0.0;
  read.stepTransports.reserve(read.steps);
  for (std::size_t t = 0; t + 1 < projectors.size(); ++t) {
    Eigen::MatrixXcd step;
    try {
      step = katoStep(projectors[t], projectors[t + 1], scheme);
    } catch (const std::invalid_argument& failure) {
      return invalidate(failure.what());
    }
    intertwining = std::max(
        intertwining,
        spectralNorm(step * projectors[t] - projectors[t + 1] * step));
    read.stepTransports.push_back(step);
    composed = step * composed;
  }
  read.transport = composed;
  read.intertwiningResidual = intertwining;
  read.composedIntertwiningResidual = spectralNorm(
      composed * projectors.front() - projectors.back() * composed);
  read.complete = true;
  // The regime of a Kato transport is the regime of the projectors it carries:
  // a Hermitian projector path is a positive-metric one, and an oblique Riesz
  // path is non-normal. The idempotency and self-adjointness of the supplied
  // projectors decide it, and nothing is inferred from the operator behind
  // them.
  bool selfAdjoint = true;
  for (const Eigen::MatrixXcd& p : projectors)
    selfAdjoint =
        selfAdjoint && spectralNorm(p - p.adjoint()) <= cfg.certificateTolerance;
  const CertificateRegime regime = selfAdjoint
                                       ? CertificateRegime::PositiveSemidefinite
                                       : CertificateRegime::NonNormal;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime,
      maxMeasured(read.intertwiningResidual, read.idempotencyResidual),
      read.maxProjectorStep, cfg.certificateTolerance);
  return read;
}

KatoTransportRead ComplexTransport::katoTransportOnFibers(
    const std::vector<SpectralFiber>& loop, KatoScheme scheme,
    const ComplexTransportConfig& cfg) {
  if (loop.size() < 2) {
    std::ostringstream message;
    message << "ComplexTransport::katoTransportOnFibers: a transport needs at "
               "least two bands on the path; received "
            << loop.size() << ".";
    throw std::invalid_argument(message.str());
  }
  const std::vector<std::vector<std::uint64_t>>& cells =
      loop.front().cellVertices();
  std::vector<Eigen::MatrixXcd> projectors;
  projectors.reserve(loop.size());
  for (std::size_t t = 0; t < loop.size(); ++t) {
    if (loop[t].cellVertices() != cells) {
      std::ostringstream message;
      message << "ComplexTransport::katoTransportOnFibers: band " << t
              << " carries a different cell list from band 0; one transport "
                 "matrix exists only on one cell space, so a path over "
                 "changing supports is transported link by link with "
                 "ComplexTransport::transport instead.";
      throw std::invalid_argument(message.str());
    }
    projectors.push_back(loop[t].projector());
  }
  return katoTransport(projectors, scheme, cfg);
}

Eigen::MatrixXcd ComplexTransport::bandTransport(
    const Eigen::MatrixXcd& transport, const Eigen::MatrixXcd& dualFrameEnd,
    const Eigen::MatrixXcd& rightFrameStart) {
  return fiberMap(dualFrameEnd, transport, rightFrameStart);
}

// ─── the exchange character ────────────────────────────────────────────────

ExchangeCharacterRead ComplexTransport::exchangeCharacter(
    const Eigen::MatrixXcd& exchangeHolonomy,
    const Eigen::MatrixXcd& referenceHolonomy, double pathLeakage,
    const ComplexTransportConfig& cfg) {
  requireSquare(exchangeHolonomy,
                "ComplexTransport::exchangeCharacter exchange holonomy");
  requireSquare(referenceHolonomy,
                "ComplexTransport::exchangeCharacter reference holonomy");
  if (exchangeHolonomy.rows() != referenceHolonomy.rows()) {
    std::ostringstream message;
    message << "ComplexTransport::exchangeCharacter: the exchange holonomy is "
            << exchangeHolonomy.rows() << " x " << exchangeHolonomy.cols()
            << " and the reference holonomy is " << referenceHolonomy.rows()
            << " x " << referenceHolonomy.cols()
            << "; a matched reference path carries the same rank as the "
               "exchange path it cancels.";
    throw std::invalid_argument(message.str());
  }

  ExchangeCharacterRead read;
  read.rank = static_cast<int>(exchangeHolonomy.rows());
  read.pathLeakage = pathLeakage;
  read.exchangeDeterminant = exchangeHolonomy.determinant();
  read.referenceDeterminant = referenceHolonomy.determinant();

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(referenceHolonomy);
  const Eigen::VectorXd& sigma = svd.singularValues();
  const double sigmaMax = sigma.size() > 0 ? sigma[0] : 0.0;
  const double sigmaMin = sigma.size() > 0 ? sigma[sigma.size() - 1] : 0.0;
  read.referenceConditionNumber = sigmaMin > 0.0 ? sigmaMax / sigmaMin : kInf;

  Eigen::FullPivLU<Eigen::MatrixXcd> lu(referenceHolonomy);
  read.referenceInvertible = lu.isInvertible();
  if (!read.referenceInvertible) {
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::NonNormal);
    return read;
  }

  // chi_F as written: the determinant of the composed matrix. The ratio of the
  // two determinants is the same number by multiplicativity and is computed
  // independently, so that the agreement of the two routes is a measured
  // number rather than an assumption.
  const Eigen::MatrixXcd composed = exchangeHolonomy * lu.inverse();
  read.character = composed.determinant();
  read.determinantRatio = read.exchangeDeterminant / read.referenceDeterminant;
  read.routeAgreementResidual =
      std::abs(read.character - read.determinantRatio);
  read.modulus = std::abs(read.character);
  read.distanceToMinusOne = std::abs(read.character + cd(1.0, 0.0));
  read.distanceToPlusOne = std::abs(read.character - cd(1.0, 0.0));

  // What the certificate grades is the premise of the experiment: that the
  // realized motion is a Kato transport, measured by the path leakage, and
  // that the composition is consistent. The modulus of the character is
  // reported above and enters no conjunct here: a character is not projected
  // to the unit circle, and a modulus away from one is a fact about the
  // motion.
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, CertificateRegime::NonNormal,
      maxMeasured(pathLeakage, read.routeAgreementResidual),
      read.referenceConditionNumber, cfg.leakageTolerance);
  return read;
}

ExchangeCharacterRead ComplexTransport::exchangeCharacterOfPaths(
    const std::vector<Eigen::MatrixXcd>& exchangePath,
    const std::vector<Eigen::MatrixXcd>& referencePath, double pathLeakage,
    const ComplexTransportConfig& cfg) {
  if (exchangePath.empty() || referencePath.empty()) {
    throw std::invalid_argument(
        "ComplexTransport::exchangeCharacterOfPaths: both the exchange path "
        "and the reference path must carry at least one link.");
  }
  const Eigen::MatrixXcd exchange = compose(exchangePath);
  const Eigen::MatrixXcd reference = compose(referencePath);
  return exchangeCharacter(exchange, reference, pathLeakage, cfg);
}

}  // namespace tessera::observables
