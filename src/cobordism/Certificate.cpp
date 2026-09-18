// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Certificate.h"

#include <stdexcept>

#include <sstream>

namespace tessera::cobordism {

const char *gradeName(CertificateGrade grade) noexcept {
  switch (grade) {
  case CertificateGrade::AlgebraicallyExact:
    return "algebraically-exact";
  case CertificateGrade::StructureExact:
    return "structure-exact";
  case CertificateGrade::CertifiedNumerical:
    return "certified-numerical";
  case CertificateGrade::HeuristicDiscovery:
    return "heuristic-discovery";
  }
  return "unknown";
}

const char *domainName(CertificateDomain domain) noexcept {
  switch (domain) {
  case CertificateDomain::Static:
    return "static";
  case CertificateDomain::BandWindow:
    return "band-window";
  }
  return "unknown";
}

const char *regimeName(CertificateRegime regime) noexcept {
  switch (regime) {
  case CertificateRegime::PositiveSemidefinite:
    return "positive-semidefinite";
  case CertificateRegime::HermitianIndefinite:
    return "hermitian-indefinite";
  case CertificateRegime::ComplexSymmetricPencil:
    return "complex-symmetric-pencil";
  case CertificateRegime::NonNormal:
    return "non-normal";
  }
  return "unknown";
}

CertificateGrade gradeFromName(const std::string &name) {
  if (name == "algebraically-exact") return CertificateGrade::AlgebraicallyExact;
  if (name == "structure-exact") return CertificateGrade::StructureExact;
  if (name == "certified-numerical") return CertificateGrade::CertifiedNumerical;
  if (name == "heuristic-discovery") return CertificateGrade::HeuristicDiscovery;
  throw std::invalid_argument("Certificate: unknown grade '" + name + "'");
}

CertificateDomain domainFromName(const std::string &name) {
  if (name == "static") return CertificateDomain::Static;
  if (name == "band-window") return CertificateDomain::BandWindow;
  throw std::invalid_argument("Certificate: unknown domain '" + name + "'");
}

CertificateRegime regimeFromName(const std::string &name) {
  if (name == "positive-semidefinite")
    return CertificateRegime::PositiveSemidefinite;
  if (name == "hermitian-indefinite")
    return CertificateRegime::HermitianIndefinite;
  if (name == "non-normal") return CertificateRegime::NonNormal;
  if (name == "complex-symmetric-pencil")
    return CertificateRegime::ComplexSymmetricPencil;
  throw std::invalid_argument("Certificate: unknown regime '" + name + "'");
}

Certificate Certificate::algebraicallyExact(CertificateDomain domain,
                                            CertificateRegime regime,
                                            double residual, double tolerance) {
  return {CertificateGrade::AlgebraicallyExact, domain,     regime,
          residual,                             kUnmeasured, tolerance};
}

Certificate Certificate::structureExact(CertificateDomain domain,
                                        CertificateRegime regime,
                                        double residual, double conditioning,
                                        double tolerance) {
  return {CertificateGrade::StructureExact, domain,       regime,
          residual,                         conditioning, tolerance};
}

Certificate Certificate::certifiedNumerical(CertificateDomain domain,
                                            CertificateRegime regime,
                                            double residual, double conditioning,
                                            double tolerance) {
  return {CertificateGrade::CertifiedNumerical, domain,       regime,
          residual,                             conditioning, tolerance};
}

Certificate Certificate::heuristicDiscovery(CertificateDomain domain,
                                            CertificateRegime regime) {
  return {CertificateGrade::HeuristicDiscovery, domain,      regime,
          kUnmeasured,                          kUnmeasured, 0.0};
}

std::string Certificate::describe() const {
  std::ostringstream out;
  out << gradeName(grade_) << " (" << domainName(domain_) << ", "
      << regimeName(regime_) << "): residual=" << residual_
      << " conditioning=" << conditioning_
      << " denseReferenceError=" << denseReferenceError_
      << " tolerance=" << tolerance_ << " holds=" << (holds() ? "yes" : "no");
  return out.str();
}

} // namespace tessera::cobordism
