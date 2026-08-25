// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// #776 — the unforced recursive-analysis integration of `MultiCobordism`.
//
// This translation unit holds the simulation modes, the one permitted
// carried-state energy coupling, the particle-independent refinement rule, the
// post-hoc analysis overlay, and the versioned checkpoint/replay path. The
// scalar objective itself stays in `MultiCobordism.cpp`; nothing here is
// reachable from it (see the firewall note in `MultiCobordism.h`).

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "cobordism/AnalyticCache.h"
#include "cobordism/Certificate.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/RecursiveQuotient.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "observables/ColorFiber.h"
#include "observables/FiberConnection.h"
#include "observables/ParticleClusters.h"
#include "observables/PersistentModularity.h"
#include "observables/SpectralFiber.h"
#include "quantum/CovarianceState.h"
#include "quantum/LazyFock.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::observables::ColorAnchor;
using ::tessera::observables::ComponentRead;
using ::tessera::observables::PersistentModularity;
using ::tessera::observables::PersistentModularityConfig;
using ::tessera::observables::ScanReport;
using ::tessera::observables::ComponentBandRead;
using ::tessera::observables::FiberConnection;
using ::tessera::observables::FiberTransportRead;
using ::tessera::observables::OrientedTriangle;
using ::tessera::observables::ParticleClusters;
using ::tessera::observables::QuarkCandidateEvidence;
using ::tessera::observables::QuarkRead;
using ::tessera::observables::SpectralFiber;
using ::tessera::observables::SpectralFiberConfig;
using ::tessera::observables::SpectralFiberTracker;
using ::tessera::quantum::CovarianceState;
using complexd = std::complex<double>;

namespace {

/// # Json
///
/// The minimal writer/reader the checkpoint schema needs
/// (scalars, strings, arrays, objects, `null`). The schema uses no other
/// construct, so a small self-contained implementation beats a third-party
/// dependency — the same call the holography schema made.
///
/// The `number` writer is where the "unknown is null, never zero" rule is
/// enforced: a non-finite double is a quantity that was NOT measured, and it
/// serializes as `null`.
class Json {
  public:
    /// A JSON string literal with the mandatory escapes.
    static std::string str(const std::string &value) {
      std::string out = "\"";
      for (const char character : value) {
        switch (character) {
          case '"': out += "\\\""; break;
          case '\\': out += "\\\\"; break;
          case '\n': out += "\\n"; break;
          case '\r': out += "\\r"; break;
          case '\t': out += "\\t"; break;
          default:
            if (static_cast<unsigned char>(character) < 0x20) {
              char buffer[7];
              std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                            static_cast<unsigned>(character));
              out += buffer;
            } else {
              out += character;
            }
        }
      }
      return out + "\"";
    }

    /// A double: round-trip exact, or `null` when the value was never
    /// measured (NaN / infinite). Unknown is NEVER zero.
    static std::string number(double value) {
      if (!std::isfinite(value)) return "null";
      std::ostringstream stream;
      stream << std::setprecision(17) << value;
      std::string text = stream.str();
      // A measured double stays a JSON float even at a whole value, so a
      // downstream reader never sees `1` where the schema promises a real.
      if (text.find_first_of(".eE") == std::string::npos) text += ".0";
      return text;
    }

    /// A complex number as the pair `[re, im]` (each `null` when unmeasured).
    static std::string complexPair(const complexd &value) {
      return "[" + number(value.real()) + ", " + number(value.imag()) + "]";
    }

    static std::string boolean(bool value) { return value ? "true" : "false"; }

    static std::string integer(long long value) {
      return std::to_string(value);
    }

    /// An optional double: `null` when absent, otherwise the number.
    static std::string optional(const std::optional<double> &value) {
      return value.has_value() ? number(*value) : std::string("null");
    }
    static std::string optional(const std::optional<int> &value) {
      return value.has_value() ? integer(*value) : std::string("null");
    }
    /// An optional complex: `null` when the read never certified one — an
    /// ABSENT character, distinct from a measured `[null, null]`.
    static std::string optional(const std::optional<complexd> &value) {
      return value.has_value() ? complexPair(*value) : std::string("null");
    }

    template <typename T, typename Fn>
    static std::string array(const std::vector<T> &items, Fn &&render) {
      std::string out = "[";
      for (std::size_t index = 0; index < items.size(); ++index) {
        if (index) out += ", ";
        out += render(items[index]);
      }
      return out + "]";
    }

    static std::string idArray(const std::vector<std::uint64_t> &ids) {
      return array(ids, [](std::uint64_t id) {
        return std::to_string(id);
      });
    }

    static std::string stringArray(const std::vector<std::string> &values) {
      return array(values, [](const std::string &value) { return str(value); });
    }

    /// Assemble `{"k": v, ...}` from ordered (key, rendered-value) pairs.
    static std::string object(
        const std::vector<std::pair<std::string, std::string>> &fields) {
      std::string out = "{";
      for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index) out += ", ";
        out += str(fields[index].first) + ": " + fields[index].second;
      }
      return out + "}";
    }

    /// The certificate record every analytic-first kernel result travels
    /// with (#764), rendered with unmeasured fields as `null`.
    static std::string certificate(const Certificate &record) {
      return object({
          {"grade", str(gradeName(record.grade()))},
          {"domain", str(record.domain() == CertificateDomain::Static
                             ? "static" : "band_window")},
          {"regime", str(regimeName(record.regime()))},
          {"residual", number(record.residual())},
          {"conditioning", number(record.conditioning())},
          {"dense_reference_error", number(record.denseReferenceError())},
          {"tolerance", number(record.tolerance())},
          {"holds", boolean(record.holds())},
      });
    }

    static std::string gradeName(CertificateGrade grade) {
      switch (grade) {
        case CertificateGrade::AlgebraicallyExact: return "algebraically_exact";
        case CertificateGrade::StructureExact: return "structure_exact";
        case CertificateGrade::CertifiedNumerical: return "certified_numerical";
        case CertificateGrade::HeuristicDiscovery: return "heuristic_discovery";
      }
      return "heuristic_discovery";
    }

    static std::string regimeName(CertificateRegime regime) {
      switch (regime) {
        case CertificateRegime::PositiveSemidefinite:
          return "positive_semidefinite";
        case CertificateRegime::HermitianIndefinite:
          return "hermitian_indefinite";
        case CertificateRegime::NonNormal: return "non_normal";
        case CertificateRegime::ComplexSymmetricPencil:
          return "complex_symmetric_pencil";
      }
      return "non_normal";
    }

    // ── the reader (replay) ───────────────────────────────────────────────

    /// The value of a TOP-LEVEL key of a JSON object document, as raw text
    /// (`""` when absent). Nesting, strings, and escapes are tracked, so a
    /// key that also appears inside a nested object is not mistaken for the
    /// top-level one.
    static std::string topLevelValue(const std::string &document,
                                     const std::string &key) {
      std::size_t position = skipSpace(document, 0);
      if (position >= document.size() || document[position] != '{')
        throw std::invalid_argument(
            "MultiCobordism checkpoint: document is not a JSON object");
      ++position;
      while (true) {
        position = skipSpace(document, position);
        if (position >= document.size()) return {};
        if (document[position] == '}') return {};
        if (document[position] != '"')
          throw std::invalid_argument(
              "MultiCobordism checkpoint: expected a key");
        const std::size_t keyEnd = endOfString(document, position);
        const std::string found =
            unescape(document.substr(position + 1, keyEnd - position - 2));
        position = skipSpace(document, keyEnd);
        if (position >= document.size() || document[position] != ':')
          throw std::invalid_argument(
              "MultiCobordism checkpoint: expected ':' after a key");
        position = skipSpace(document, position + 1);
        const std::size_t valueEnd = endOfValue(document, position);
        if (found == key)
          return document.substr(position, valueEnd - position);
        position = skipSpace(document, valueEnd);
        if (position < document.size() && document[position] == ',') ++position;
      }
    }

    /// Split a JSON array's raw text into its element texts.
    static std::vector<std::string> elements(const std::string &arrayText) {
      std::vector<std::string> out;
      std::size_t position = skipSpace(arrayText, 0);
      if (position >= arrayText.size() || arrayText[position] != '[')
        throw std::invalid_argument(
            "MultiCobordism checkpoint: expected a JSON array");
      ++position;
      while (true) {
        position = skipSpace(arrayText, position);
        if (position >= arrayText.size()) return out;
        if (arrayText[position] == ']') return out;
        const std::size_t valueEnd = endOfValue(arrayText, position);
        out.push_back(arrayText.substr(position, valueEnd - position));
        position = skipSpace(arrayText, valueEnd);
        if (position < arrayText.size() && arrayText[position] == ',')
          ++position;
      }
    }

    static double asNumber(const std::string &text) {
      const std::string trimmed = text.substr(skipSpace(text, 0));
      if (trimmed.rfind("null", 0) == 0)
        return std::numeric_limits<double>::quiet_NaN();
      return std::stod(trimmed);
    }

    static std::string asString(const std::string &text) {
      const std::size_t start = skipSpace(text, 0);
      if (start >= text.size() || text[start] != '"')
        throw std::invalid_argument(
            "MultiCobordism checkpoint: expected a JSON string");
      const std::size_t end = endOfString(text, start);
      return unescape(text.substr(start + 1, end - start - 2));
    }

  private:
    static std::size_t skipSpace(const std::string &text,
                                 std::size_t position) {
      while (position < text.size() &&
             (text[position] == ' ' || text[position] == '\n' ||
              text[position] == '\r' || text[position] == '\t'))
        ++position;
      return position;
    }

    /// One PAST the closing quote of the string starting at `position`.
    static std::size_t endOfString(const std::string &text,
                                   std::size_t position) {
      ++position;
      while (position < text.size()) {
        if (text[position] == '\\') {
          position += 2;
          continue;
        }
        if (text[position] == '"') return position + 1;
        ++position;
      }
      throw std::invalid_argument(
          "MultiCobordism checkpoint: unterminated string");
    }

    /// One PAST the end of the value starting at `position`.
    static std::size_t endOfValue(const std::string &text,
                                  std::size_t position) {
      if (position >= text.size())
        throw std::invalid_argument(
            "MultiCobordism checkpoint: truncated document");
      if (text[position] == '"') return endOfString(text, position);
      if (text[position] == '{' || text[position] == '[') {
        int depth = 0;
        while (position < text.size()) {
          const char character = text[position];
          if (character == '"') {
            position = endOfString(text, position);
            continue;
          }
          if (character == '{' || character == '[') ++depth;
          if (character == '}' || character == ']') {
            --depth;
            if (depth == 0) return position + 1;
          }
          ++position;
        }
        throw std::invalid_argument(
            "MultiCobordism checkpoint: unbalanced brackets");
      }
      while (position < text.size() && text[position] != ',' &&
             text[position] != '}' && text[position] != ']' &&
             text[position] != ' ' && text[position] != '\n' &&
             text[position] != '\r' && text[position] != '\t')
        ++position;
      return position;
    }

    static std::string unescape(const std::string &text) {
      std::string out;
      for (std::size_t index = 0; index < text.size(); ++index) {
        if (text[index] != '\\') {
          out += text[index];
          continue;
        }
        ++index;
        if (index >= text.size()) break;
        switch (text[index]) {
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          default: out += text[index];
        }
      }
      return out;
    }
};

/// # BandAnchor
///
/// The #767 calibrated oriented-triangle anchor of one accepted degree-one
/// rank-three band, built from the complex the band was read on: the atlas is
/// every 2-cell all three of whose boundary edges are cells of the band, with
/// the standard incidence signs (+, −, +) of \f$ \partial[a,b,c] \f$. The
/// weighting is the UNIFORM convex one and is declared BEFORE any evaluation,
/// so no post-hoc weight selection is possible.
///
/// An empty atlas (a band whose cells carry no complete triangle) returns the
/// default profile — MISSING anchor evidence, which the classifier reports as
/// a named failed certificate rather than a presumed pass.
class BandAnchor {
  public:
    static ::tessera::observables::AnchorProfile of(
        const std::shared_ptr<::tessera::spacetime::Spacetime> &spacetime,
        const SpectralFiber &fiber) {
      const auto &cells = fiber.cellVertices();
      std::map<std::vector<std::uint64_t>, Eigen::Index> rowOfEdge;
      for (std::size_t index = 0; index < cells.size(); ++index) {
        auto key = cells[index];
        std::sort(key.begin(), key.end());
        rowOfEdge[key] = static_cast<Eigen::Index>(index);
      }
      std::vector<OrientedTriangle> atlas;
      const auto chainComplex =
          ChainComplex::fromSpacetime(*spacetime);
      for (const auto &triangle : chainComplex.kSimplexVertices(2)) {
        if (triangle.size() != 3) continue;
        auto sorted = triangle;
        std::sort(sorted.begin(), sorted.end());
        // ∂[a,b,c] = [b,c] − [a,c] + [a,b]
        const std::vector<std::vector<std::uint64_t>> boundary = {
            {sorted[1], sorted[2]}, {sorted[0], sorted[2]},
            {sorted[0], sorted[1]}};
        OrientedTriangle oriented;
        oriented.signs = {+1, -1, +1};
        bool complete = true;
        for (std::size_t side = 0; side < 3; ++side) {
          const auto found = rowOfEdge.find(boundary[side]);
          if (found == rowOfEdge.end()) {
            complete = false;
            break;
          }
          oriented.edges[side] = found->second;
        }
        if (complete) atlas.push_back(oriented);
      }
      if (atlas.empty()) return {};
      const Eigen::VectorXcd weights = fiber.weightDiagonal();
      Eigen::VectorXd edgeWeights(weights.size());
      for (Eigen::Index index = 0; index < weights.size(); ++index)
        edgeWeights(index) = weights(index).real();
      try {
        const Eigen::MatrixXcd frame =
            ColorAnchor::orthonormalizeFrame(fiber.rightFrame(), edgeWeights);
        ColorAnchor anchor(std::move(atlas));
        return anchor.evaluate(frame, edgeWeights);
      } catch (const std::exception &) {
        return {};  // an unanchorable frame is MISSING evidence, not a pass
      }
    }
};

}  // namespace

// ======================================================================
// modes
// ======================================================================

void MultiCobordism::setSimulationMode(SimulationMode mode,
                                       EmergenceSubmode submode) {
  simulationMode_ = mode;
  emergenceSubmode_ = submode;
  // Leaving the backreaction sub-mode SELECTS the strict one, which is the
  // acceptance statement of #776: "disabling the state-energy coupling selects
  // the different strict sub-mode". The weight is the coupling, so it is what
  // gets zeroed.
  if (mode != SimulationMode::Emergence ||
      submode != EmergenceSubmode::CertificatesBlindMeanField)
    carriedStateEnergyWeight_ = 0.0;
}

std::string MultiCobordism::modeName(SimulationMode mode) {
  switch (mode) {
    case SimulationMode::Emergence: return "emergence";
    case SimulationMode::Synthesis: return "synthesis";
    case SimulationMode::Replay: return "replay";
  }
  return "emergence";
}

std::string MultiCobordism::submodeName(EmergenceSubmode submode) {
  switch (submode) {
    case EmergenceSubmode::Strict: return "strict";
    case EmergenceSubmode::CertificatesBlindMeanField:
      return "certificates_blind_mean_field";
  }
  return "strict";
}

// ======================================================================
// the carried quasi-free state — the ONE permitted coupling
// ======================================================================

void MultiCobordism::setCarriedState(
    const std::vector<std::vector<std::uint64_t>> &modeCells, int degree,
    const std::vector<complexd> &covariance) {
  // Degree ONE and above. At degree zero the Hodge operator is the graph
  // Laplacian over vertices in sorted-id order rather than the canonical
  // ChainComplex cell order, so a mode cell would be located against the
  // wrong index; and `laplacianGradient` — the coupling's exact derivative —
  // is defined only for k >= 1 anyway. Refusing loudly beats mis-mapping.
  // A DECLARED domain for the carried state, not a capability limit: since
  // #805 L_0 and its exact gradient exist too (a degree-zero carried state
  // would put the modes on VERTICES, which is a separate design decision).
  if (degree < 1)
    throw std::invalid_argument(
        "MultiCobordism::setCarriedState: the carried state is declared over "
        "degrees >= 1; got degree " + std::to_string(degree));
  const std::size_t modeCount = modeCells.size();
  if (covariance.size() != modeCount * modeCount)
    throw std::invalid_argument(
        "MultiCobordism::setCarriedState: covariance must be a square matrix "
        "over the declared mode cells");
  carriedModeCells_ = modeCells;
  carriedCovariance_ = covariance;
  carriedStateDegree_ = degree;
}

void MultiCobordism::clearCarriedState() {
  carriedModeCells_.clear();
  carriedCovariance_.clear();
}

void MultiCobordism::setCarriedStateEnergyWeight(double weight) {
  if (!std::isfinite(weight))
    throw std::invalid_argument(
        "MultiCobordism: carried-state energy weight must be finite");
  if (weight != 0.0 &&
      (simulationMode_ != SimulationMode::Emergence ||
       emergenceSubmode_ != EmergenceSubmode::CertificatesBlindMeanField))
    throw std::invalid_argument(
        "MultiCobordism: a nonzero carried-state energy weight is the "
        "certificates_blind_mean_field coupling; select that emergence "
        "sub-mode first");
  carriedStateEnergyWeight_ = weight;
}

std::vector<complexd> MultiCobordism::carriedStateGenerator(
    const std::shared_ptr<Spacetime> &spacetime) const {
  const std::size_t modeCount = carriedModeCells_.size();
  std::vector<complexd> generator(modeCount * modeCount, complexd{0.0, 0.0});
  if (!spacetime || modeCount == 0) return generator;

  // The carried modes' cells, located in the canonical ChainComplex order by
  // vertex SET (no vertex order is ever imposed).
  const auto chainComplex = ChainComplex::fromSpacetime(*spacetime);
  const auto cells = chainComplex.kSimplexVertices(carriedStateDegree_);
  std::map<std::vector<std::uint64_t>, std::size_t> indexOfCell;
  for (std::size_t cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
    auto key = cells[cellIndex];
    std::sort(key.begin(), key.end());
    indexOfCell[key] = cellIndex;
  }
  std::vector<long long> row(modeCount, -1);
  bool anyPresent = false;
  for (std::size_t modeIndex = 0; modeIndex < modeCount; ++modeIndex) {
    auto key = carriedModeCells_[modeIndex];
    std::sort(key.begin(), key.end());
    const auto found = indexOfCell.find(key);
    if (found == indexOfCell.end()) continue;  // the move removed this cell
    row[modeIndex] = static_cast<long long>(found->second);
    anyPresent = true;
  }
  if (!anyPresent) return generator;

  const auto laplacian =
      HodgeLaplacian(spacetime).laplacian(carriedStateDegree_, /*metric=*/true);
  const std::size_t cellCount = cells.size();
  if (laplacian.size() != cellCount * cellCount) return generator;
  // h_S = the HERMITIAN PART of L_k restricted to the carried cells. The
  // signed-weight operator is generally non-normal, and a mean-field generator
  // must be Hermitian for the covariance evolution to stay unitary (and hence
  // Gaussian-closed); (L + L†)/2 is that part exactly, never a repair.
  for (std::size_t i = 0; i < modeCount; ++i) {
    if (row[i] < 0) continue;
    for (std::size_t j = 0; j < modeCount; ++j) {
      if (row[j] < 0) continue;
      const auto a = static_cast<std::size_t>(row[i]);
      const auto b = static_cast<std::size_t>(row[j]);
      generator[i * modeCount + j] =
          0.5 * (laplacian[a * cellCount + b] +
                 std::conj(laplacian[b * cellCount + a]));
    }
  }
  return generator;
}

double MultiCobordism::carriedStateEnergy(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // Cheapest gate first: this runs inside the stage-1 scoring loop.
  if (carriedModeCells_.empty()) return 0.0;
  if (simulationMode_ != SimulationMode::Emergence ||
      emergenceSubmode_ != EmergenceSubmode::CertificatesBlindMeanField)
    return 0.0;
  if (!spacetime) return 0.0;
  const std::size_t modeCount = carriedModeCells_.size();
  const auto generator = carriedStateGenerator(spacetime);
  // E = Re tr(Γ h) = Re Σ_ij Γ_ij h_ji.
  complexd trace{0.0, 0.0};
  for (std::size_t i = 0; i < modeCount; ++i)
    for (std::size_t j = 0; j < modeCount; ++j)
      trace += carriedCovariance_[i * modeCount + j] *
               generator[j * modeCount + i];
  return trace.real();
}

std::vector<complexd> MultiCobordism::carriedStateEnergyGradient(
    const std::shared_ptr<Spacetime> &spacetime) const {
  const auto edges = spacetime && spacetime->getEdgeList()
                         ? spacetime->getEdgeList()->toVector()
                         : std::vector<::tessera::mesh::Edge *>{};
  std::vector<complexd> gradient(edges.size(), complexd{0.0, 0.0});
  if (carriedModeCells_.empty()) return gradient;
  if (simulationMode_ != SimulationMode::Emergence ||
      emergenceSubmode_ != EmergenceSubmode::CertificatesBlindMeanField)
    return gradient;
  // No degree guard: setCarriedState already declares the carried degree's
  // domain, and laplacianGradient is exact at every degree it admits --
  // including zero since #805, where L_0 = d_1 W_1^-1 d_1^T is holomorphic in z.

  const auto chainComplex = ChainComplex::fromSpacetime(*spacetime);
  const auto cells = chainComplex.kSimplexVertices(carriedStateDegree_);
  const std::size_t cellCount = cells.size();
  if (cellCount == 0) return gradient;
  std::map<std::vector<std::uint64_t>, std::size_t> indexOfCell;
  for (std::size_t cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
    auto key = cells[cellIndex];
    std::sort(key.begin(), key.end());
    indexOfCell[key] = cellIndex;
  }
  const std::size_t modeCount = carriedModeCells_.size();
  std::vector<long long> row(modeCount, -1);
  for (std::size_t modeIndex = 0; modeIndex < modeCount; ++modeIndex) {
    auto key = carriedModeCells_[modeIndex];
    std::sort(key.begin(), key.end());
    const auto found = indexOfCell.find(key);
    if (found != indexOfCell.end())
      row[modeIndex] = static_cast<long long>(found->second);
  }

  const HodgeLaplacian hodge(spacetime);
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto *edge = edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto derivative = hodge.laplacianGradient(
        carriedStateDegree_, edge->getSource()->getId(),
        edge->getTarget()->getId());
    if (derivative.size() != cellCount * cellCount) continue;
    // ∂E/∂z_e = Re tr(Γ_S ∂h_S/∂z_e), with ∂h = (∂L + ∂L†)/2 exactly as the
    // value uses. Real-analytic in z: the operator's dependence on ℓ² enters
    // only through the real inner-product weights, so ∂E/∂(Im z) = 0 and the
    // real-plane ascent displacement is the real derivative alone.
    complexd trace{0.0, 0.0};
    for (std::size_t i = 0; i < modeCount; ++i) {
      if (row[i] < 0) continue;
      for (std::size_t j = 0; j < modeCount; ++j) {
        if (row[j] < 0) continue;
        const auto a = static_cast<std::size_t>(row[j]);
        const auto b = static_cast<std::size_t>(row[i]);
        const complexd derivativeEntry =
            0.5 * (derivative[a * cellCount + b] +
                   std::conj(derivative[b * cellCount + a]));
        trace += carriedCovariance_[i * modeCount + j] * derivativeEntry;
      }
    }
    gradient[edgeIndex] = complexd{trace.real(), 0.0};
  }
  return gradient;
}

double MultiCobordism::carriedStatePurityDefect() const {
  if (carriedModeCells_.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const auto modeCount = static_cast<Eigen::Index>(carriedModeCells_.size());
  Eigen::MatrixXcd gamma(modeCount, modeCount);
  for (Eigen::Index i = 0; i < modeCount; ++i)
    for (Eigen::Index j = 0; j < modeCount; ++j)
      gamma(i, j) = carriedCovariance_[static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(modeCount) +
                                       static_cast<std::size_t>(j)];
  return CovarianceState(std::move(gamma)).purityDefect();
}

bool MultiCobordism::carriedStatePurityHolds(double tolerance) const {
  if (carriedModeCells_.empty()) return false;
  const auto modeCount = static_cast<Eigen::Index>(carriedModeCells_.size());
  Eigen::MatrixXcd gamma(modeCount, modeCount);
  for (Eigen::Index i = 0; i < modeCount; ++i)
    for (Eigen::Index j = 0; j < modeCount; ++j)
      gamma(i, j) = carriedCovariance_[static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(modeCount) +
                                       static_cast<std::size_t>(j)];
  return CovarianceState(std::move(gamma)).purityCertificate(tolerance).holds();
}

void MultiCobordism::setMeanFieldSchedule(double dt, int steps) {
  if (!std::isfinite(dt))
    throw std::invalid_argument(
        "MultiCobordism::setMeanFieldSchedule: dt must be finite");
  if (steps < 0)
    throw std::invalid_argument(
        "MultiCobordism::setMeanFieldSchedule: steps must be non-negative");
  meanFieldStepSize_ = dt;
  meanFieldSteps_ = steps;
}

double MultiCobordism::advanceCarriedState() {
  if (carriedModeCells_.empty() || meanFieldSteps_ <= 0)
    return std::numeric_limits<double>::quiet_NaN();
  const auto modeCount = static_cast<Eigen::Index>(carriedModeCells_.size());
  Eigen::MatrixXcd gamma(modeCount, modeCount);
  for (Eigen::Index i = 0; i < modeCount; ++i)
    for (Eigen::Index j = 0; j < modeCount; ++j)
      gamma(i, j) = carriedCovariance_[static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(modeCount) +
                                       static_cast<std::size_t>(j)];
  CovarianceState state(std::move(gamma));

  // h(Γ, g): the classical geometry is closed over by THIS caller, exactly as
  // #780's `meanFieldEvolve` contract states. The generator is the same
  // Hermitian one-particle operator the energy term measures, so the objective
  // coupling and the state's propagation are one functional, not two.
  const auto generatorFlat = carriedStateGenerator(spacetime_);
  Eigen::MatrixXcd generator(modeCount, modeCount);
  for (Eigen::Index i = 0; i < modeCount; ++i)
    for (Eigen::Index j = 0; j < modeCount; ++j)
      generator(i, j) = generatorFlat[static_cast<std::size_t>(i) *
                                          static_cast<std::size_t>(modeCount) +
                                      static_cast<std::size_t>(j)];
  const auto steps = state.meanFieldEvolve(
      [&generator](const Eigen::MatrixXcd &) { return generator; },
      meanFieldStepSize_, static_cast<std::size_t>(meanFieldSteps_));

  const Eigen::MatrixXcd &evolved = state.gamma();
  for (Eigen::Index i = 0; i < modeCount; ++i)
    for (Eigen::Index j = 0; j < modeCount; ++j)
      carriedCovariance_[static_cast<std::size_t>(i) *
                             static_cast<std::size_t>(modeCount) +
                         static_cast<std::size_t>(j)] = evolved(i, j);

  double worstPurityDefect = 0.0;
  for (const auto &step : steps)
    worstPurityDefect = std::max(worstPurityDefect, step.purityDefect);
  return worstPurityDefect;
}

// ======================================================================
// particle-independent refinement
// ======================================================================

std::vector<std::string> MultiCobordism::refinementIndicatorNames() {
  return {"regge_stationarity_residual", "hodge_stationarity_residual",
          "curvature_concentration", "mesh_quality", "solver_error"};
}

MultiCobordism::RefinementIndicators MultiCobordism::refinementIndicators()
    const {
  RefinementIndicators indicators;
  indicators.solverError = lastStage2Improvement_;
  if (!spacetime_) return indicators;
  indicators.reggeStationarityResidual = reggeActionGradient(spacetime_);
  indicators.hodgeStationarityResidual = hodgeEntropyStationarity();

  const int dimension = spacetime_->getDimensions();
  double maximumDeficit = 0.0;
  double totalDeficit = 0.0;
  std::size_t hingeCount = 0;
  double minimumVolume = std::numeric_limits<double>::infinity();
  double maximumVolume = 0.0;
  for (const auto &simplex : spacetime_->getSimplices()) {
    if (!simplex) continue;
    const int simplexDimension =
        static_cast<int>(simplex->getVertices().size()) - 1;
    if (simplexDimension == dimension - 2) {
      const double magnitude = std::abs(simplex->deficitAngle());
      if (!std::isfinite(magnitude)) continue;
      maximumDeficit = std::max(maximumDeficit, magnitude);
      totalDeficit += magnitude;
      ++hingeCount;
    } else if (simplexDimension == dimension) {
      const double volume = std::abs(simplex->volume());
      if (!std::isfinite(volume)) continue;
      minimumVolume = std::min(minimumVolume, volume);
      maximumVolume = std::max(maximumVolume, volume);
    }
  }
  if (hingeCount > 0 && totalDeficit > 0.0)
    indicators.curvatureConcentration =
        maximumDeficit / (totalDeficit / static_cast<double>(hingeCount));
  indicators.meshQuality =
      maximumVolume > 0.0 && std::isfinite(minimumVolume)
          ? minimumVolume / maximumVolume
          : (maximumVolume > 0.0 ? 0.0 : 1.0);
  return indicators;
}

void MultiCobordism::setRefinementThresholds(
    const RefinementIndicators &thresholds) {
  refinementThresholds_ = thresholds;
}

MultiCobordism::RefinementDecision MultiCobordism::refinementDecisionOf(
    const RefinementIndicators &indicators,
    const RefinementIndicators &thresholds) {
  // STATIC over two indicator records. There is no third argument: no coarse
  // response residual, band gap, modularity, transport leakage, Wilson/center
  // read, exchange read, anchor score, amplitude Gram defect, or particle
  // score can reach this decision, because none of them is in scope.
  RefinementDecision decision;
  decision.indicators = indicators;
  const auto exceeded = [](double value, double threshold) {
    return threshold > 0.0 && std::isfinite(threshold) && value > threshold;
  };
  if (exceeded(indicators.reggeStationarityResidual,
               thresholds.reggeStationarityResidual)) {
    decision.refine = true;
    decision.trigger = "regge_stationarity_residual";
  } else if (exceeded(indicators.hodgeStationarityResidual,
                      thresholds.hodgeStationarityResidual)) {
    decision.refine = true;
    decision.trigger = "hodge_stationarity_residual";
  } else if (exceeded(indicators.curvatureConcentration,
                      thresholds.curvatureConcentration)) {
    decision.refine = true;
    decision.trigger = "curvature_concentration";
  } else if (thresholds.meshQuality > 0.0 &&
             indicators.meshQuality < thresholds.meshQuality) {
    decision.refine = true;
    decision.trigger = "mesh_quality";
  } else if (exceeded(indicators.solverError, thresholds.solverError)) {
    decision.refine = true;
    decision.trigger = "solver_error";
  }
  return decision;
}

MultiCobordism::RefinementDecision MultiCobordism::refinementDecision() const {
  return refinementDecisionOf(refinementIndicators(), refinementThresholds_);
}

int MultiCobordism::refineGeometry(int maxCells) {
  if (maxCells <= 0) return 0;
  if (!refinementDecision().refine) return 0;
  // The refinement move is the EXISTING gated cone-in — `preconeCells` drives
  // exactly the same `applyMoveSpecification` / `dualComplexValid` primitive
  // stage 1 uses. Nothing is reimplemented and nothing bypasses the manifold
  // and orientation gates.
  const std::size_t cellsBefore = spacetime_ ? spacetime_->getTopSimplices().size() : 0;
  preconeCells(maxCells, /*timelike=*/false, /*alternate=*/false);
  const std::size_t cellsAfter = spacetime_ ? spacetime_->getTopSimplices().size() : 0;
  const int committed =
      cellsAfter > cellsBefore ? static_cast<int>(cellsAfter - cellsBefore) : 0;
  // Each committed refinement cell is an accepted move, so the cadence counts
  // it as one — the same bookkeeping a stage-1 commit gets.
  for (int cell = 0; cell < committed; ++cell) noteAcceptedMove();
  return committed;
}

// ======================================================================
// the post-hoc analysis overlay
// ======================================================================

void MultiCobordism::setAnalysisConfig(const AnalysisConfig &config) {
  if (config.cadence < 1)
    throw std::invalid_argument(
        "MultiCobordism::setAnalysisConfig: cadence must be at least one");
  analysisConfig_ = config;
}

void MultiCobordism::setProvenance(const std::string &configHash,
                                   const std::string &commit) {
  provenanceConfigHash_ = configHash;
  provenanceCommit_ = commit;
}

void MultiCobordism::noteAcceptedMove() {
  ++acceptedMoveCount_;
  if (!analysisConfig_.enabled) return;  // the disabled path costs one branch
  if (acceptedMoveCount_ %
          static_cast<std::uint64_t>(std::max(1, analysisConfig_.cadence)) !=
      0)
    return;
  runRecursiveAnalysisOn(spacetime_);
}

void MultiCobordism::runRecursiveAnalysis() {
  runRecursiveAnalysisOn(spacetime_);
}

std::string MultiCobordism::rawComplexJson(
    const std::shared_ptr<Spacetime> &spacetime) const {
  if (!spacetime) return Json::object({});
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &topSimplex : spacetime->getTopSimplices())
    cells.push_back(topSimplex->topTuple());
  std::string cellText = "[";
  for (std::size_t index = 0; index < cells.size(); ++index) {
    if (index) cellText += ", ";
    cellText += Json::idArray(cells[index]);
  }
  cellText += "]";
  // Serialized in CANONICAL endpoint order, not in the live list order: the
  // raw complex a checkpoint records is then a pure function of the geometry,
  // so two runs that reached the same complex write the same bytes even when
  // their internal edge lists were built in different orders.
  // Both edge fields are recorded. The length is orientation-free, but the
  // connection phase is NOT: the link on the reverse orientation is the
  // inverse, so a phase written against the canonical min->max direction must
  // be negated whenever the live edge stores target->source. Recording the
  // canonical-direction phase (`edgeGeometryOf`, the stage-1 snapshot's own
  // record) keeps the document a pure function of the geometry, exactly as
  // the endpoint sort does for the cell order.
  std::map<std::pair<std::uint64_t, std::uint64_t>, EdgeGeometry> edgesByEndpoints;
  for (const auto *edge : spacetime->getEdgeList()->toVector()) {
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto a = edge->getSource()->getId();
    const auto b = edge->getTarget()->getId();
    edgesByEndpoints[{std::min(a, b), std::max(a, b)}] = edgeGeometryOf(*edge);
  }
  std::string edgeText = "[";
  bool firstEdge = true;
  for (const auto &entry : edgesByEndpoints) {
    if (!firstEdge) edgeText += ", ";
    firstEdge = false;
    edgeText += Json::object({
        {"a", Json::integer(static_cast<long long>(entry.first.first))},
        {"b", Json::integer(static_cast<long long>(entry.first.second))},
        {"length", Json::complexPair(entry.second.length)},
        {"phase", Json::complexPair(entry.second.phase)},
    });
  }
  edgeText += "]";
  return Json::object({
      {"dimensions", Json::integer(spacetime->getDimensions())},
      {"cells", cellText},
      {"edges", edgeText},
  });
}

void MultiCobordism::runRecursiveAnalysisOn(
    const std::shared_ptr<Spacetime> &spacetime) {
  if (!spacetime) return;
  ++analysisPassCount_;

  // ── §17.0 record the objective of the state about to be analysed ─────
  //
  // FIRST, before any spectral read. Measured (#776): the engine's stage-2
  // trajectory shifts by ~1e-11 relative when ANY read-only Hodge observable
  // — `HodgeLaplacian::spectrum`, `spectralEntropy`, `MultiCobordism::betti`,
  // `hodgeEntropy` — is evaluated before `ReggeSolver::actionGradientExact`
  // rather than after it. That sensitivity is pre-existing and has nothing to
  // do with this ticket (a bare `HodgeLaplacian(st).spectralEntropy(1)` with
  // no analysis at all reproduces it exactly, value for value), but the
  // overlay would inherit it. Evaluating the objective's own terms first —
  // which is the order `objective()` itself uses — keeps an analysed run
  // BIT-IDENTICAL to an unanalysed one, and is the natural order anyway:
  // a checkpoint records the objective of the state it describes.
  const auto terms = objectiveTermsFor(spacetime);
  const auto indicators = refinementIndicators();
  const auto decision = refinementDecisionOf(indicators, refinementThresholds_);

  // ── §17.1 publish the accepted move and update the analytic caches ──
  //
  // The touched star is the exact support of what changed since the last pass:
  // the vertices of every created/deleted top cell, plus the endpoints of every
  // edge whose complex length moved. Entries whose component misses the star
  // survive — the invalidation is local, not global.
  std::set<std::vector<std::uint64_t>> cellSet;
  for (const auto &topSimplex : spacetime->getTopSimplices())
    cellSet.insert(topSimplex->topTuple());
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> edgeLengths;
  for (const auto *edge : spacetime->getEdgeList()->toVector()) {
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto a = edge->getSource()->getId();
    const auto b = edge->getTarget()->getId();
    edgeLengths[{std::min(a, b), std::max(a, b)}] = edge->getLength();
  }
  TouchedStar star;
  std::vector<std::uint64_t> touchedCells;
  if (analysisCellSetValid_) {
    for (const auto &cell : cellSet)
      if (!analysisCellSet_.count(cell)) star.addCreatedCell(cell);
    for (const auto &cell : analysisCellSet_)
      if (!cellSet.count(cell)) star.addDeletedCell(cell);
    for (const auto &entry : edgeLengths) {
      const auto previous = analysisEdgeLengths_.find(entry.first);
      if (previous == analysisEdgeLengths_.end() ||
          previous->second != entry.second)
        star.addChangedEdge(entry.first.first, entry.first.second);
    }
    for (const auto vertexId : star.vertices()) touchedCells.push_back(vertexId);
  }
  analysisCellSet_ = cellSet;
  analysisEdgeLengths_ = edgeLengths;
  analysisCellSetValid_ = true;

  // The cache survives ACROSS passes while the complex object does, so a
  // published star drops only the entries whose component it meets. A
  // committed combinatorial move rebuilds the complex (the engine's existing
  // `build(snapshot)` behaviour), which necessarily rebinds the cache.
  const auto boundSpacetime = analysisCacheBinding_.lock();
  if (!analysisCache_ || boundSpacetime != spacetime) {
    analysisCache_ = std::make_shared<AnalyticCache>(spacetime);
    analysisCacheBinding_ = spacetime;
  }
  auto cache = std::static_pointer_cast<AnalyticCache>(analysisCache_);
  cache->setEnabled(!analysisConfig_.coldCaches);
  // The checkpoint reports THIS pass's cache activity, not the cache's
  // lifetime totals, so a reader can see what one incremental update cost.
  const std::uint64_t hitsBefore = cache->hits();
  const std::uint64_t missesBefore = cache->misses();
  const std::uint64_t invalidationsBefore = cache->invalidations();
  if (!star.empty()) cache->publish(star);

  // ── §17.2 the local component hierarchy ──────────────────────────────
  PersistentModularityConfig modularityConfig;
  modularityConfig.resolutions = analysisConfig_.resolutions.empty()
                                     ? std::vector<double>{1.0}
                                     : analysisConfig_.resolutions;
  modularityConfig.baseSeed = seed_;
  const auto modularity = PersistentModularity::fromSpacetime(*spacetime);
  const ScanReport report = modularity.scanResolutions(modularityConfig);
  const auto invalidated =
      PersistentModularity::invalidatedAncestry(report, touchedCells);
  // Every component at every hierarchy level of every slice — the denominator
  // the invalidated count is local WITH RESPECT TO.
  std::size_t totalComponents = 0;
  for (const auto &slice : report.slices)
    for (const auto &level : slice.hierarchy) totalComponents += level.size();

  std::vector<ComponentRead> components;
  std::vector<ComponentRead> nextLevelComponents;
  if (!report.slices.empty()) {
    components = report.slices.front().components;
    // The NEXT modular level of the same slice: the bound-supercomponent
    // search reads it, and it is where a three-quark binding would live.
    if (report.slices.front().hierarchy.size() > 1)
      nextLevelComponents = report.slices.front().hierarchy[1];
  }

  // ── §17.3 spectral projectors, the labeled fiber sum, and transports ──
  SpectralFiberConfig fiberConfig;
  fiberConfig.degrees = analysisConfig_.degrees.empty()
                            ? std::vector<int>{1}
                            : analysisConfig_.degrees;
  const SpectralFiberTracker tracker(spacetime, fiberConfig);
  std::vector<ComponentBandRead> bandReads;
  // `ComponentBandRead` carries the component's SUPPORT, not its label-free
  // #765 identity, so the owning component index travels beside each read.
  std::vector<std::size_t> bandComponent;
  for (std::size_t componentIndex = 0; componentIndex < components.size();
       ++componentIndex)
    for (const int degree : fiberConfig.degrees) {
      bandReads.push_back(tracker.enumerateBandsCached(
          *cache, components[componentIndex].support, degree));
      bandComponent.push_back(componentIndex);
    }

  std::vector<std::pair<std::size_t, RecursiveQuotient::LabeledFiberSumRead>>
      labeledSums;
  std::vector<std::pair<std::size_t, Certificate>> staticCertificates;
  if (!components.empty()) {
    std::vector<std::vector<std::uint64_t>> supports;
    supports.reserve(components.size());
    for (const auto &component : components) supports.push_back(component.support);
    for (const int degree : fiberConfig.degrees) {
      try {
        const auto quotient = RecursiveQuotient::overVertexSupports(
            spacetime, degree, supports, RecursiveQuotient::Options(), cache);
        staticCertificates.emplace_back(
            static_cast<std::size_t>(degree),
            quotient.staticReduction().certificate);
        labeledSums.emplace_back(static_cast<std::size_t>(degree),
                                 quotient.labeledFiberSum());
      } catch (const std::exception &) {
        // A degree the reduction refuses (an uncovered or empty skeleton) is
        // an UNKNOWN, recorded by its absence from the checkpoint — never a
        // fabricated zero and never a reason to disturb the geometry.
      }
    }
  }

  // The candidate band of each component read: the FIRST accepted band, the
  // one every downstream read is assembled around.
  std::vector<int> candidateBand(bandReads.size(), -1);
  for (std::size_t index = 0; index < bandReads.size(); ++index)
    for (std::size_t band = 0; band < bandReads[index].fibers.size(); ++band)
      if (bandReads[index].fibers[band].accepted()) {
        candidateBand[index] = static_cast<int>(band);
        break;
      }

  // MUTUAL transports: one derived link per ordered pair of candidate bands
  // at the same degree — the cross-component family the bound-supercomponent
  // search reads. Deliberately NOT every band against every band: a
  // component's bands are alternative carriers, not links, and pairing them
  // all costs O(bands²) derived transports per component pair (measured 663
  // at 62 cells, 30x the optimizer step) for links no certificate consumes.
  //
  // The LIFETIME family — the world-tube transports of ONE candidate across
  // frames — is a different object, and one analysis pass sees exactly one
  // frame, so it stays empty here and the corresponding certificate is
  // NAMED as missing rather than filled with cross-component links that do
  // not mean that.
  const FiberConnection connection;
  struct TransportRecord {
    std::size_t fromRead = 0;
    std::size_t toRead = 0;
    FiberTransportRead read;
  };
  std::vector<TransportRecord> transports;
  for (std::size_t a = 0; a < bandReads.size(); ++a) {
    if (candidateBand[a] < 0) continue;
    for (std::size_t b = 0; b < bandReads.size(); ++b) {
      if (a == b || candidateBand[b] < 0) continue;
      if (bandReads[a].degree != bandReads[b].degree) continue;
      const auto &toFiber =
          bandReads[a].fibers[static_cast<std::size_t>(candidateBand[a])];
      const auto &fromFiber =
          bandReads[b].fibers[static_cast<std::size_t>(candidateBand[b])];
      if (fromFiber.rank() != toFiber.rank()) continue;
      try {
        transports.push_back(
            {b, a,
             connection.transportOnSpacetimeCached(*cache, spacetime, toFiber,
                                                   fromFiber)});
      } catch (const std::exception &) {
        // A shape the transfer refuses is an unknown link, not a fault.
      }
    }
  }

  // ── §17.4 the quasi-free covariance and its Wick reads ───────────────
  //
  // The carried state when the run declares one; otherwise the exact pure
  // Slater covariance of each ACCEPTED band projector (#780's documented
  // `fromBandProjector` entry point) — a READ of the relaxed geometry, never a
  // fabricated occupancy.
  std::vector<std::optional<CovarianceState>> bandStates(bandReads.size());
  double worstPurityDefect = std::numeric_limits<double>::quiet_NaN();
  std::size_t activeModes = 0;
  for (std::size_t index = 0; index < bandReads.size(); ++index) {
    for (const auto &fiber : bandReads[index].fibers) {
      if (!fiber.accepted()) continue;
      try {
        auto state = CovarianceState::fromBandProjector(fiber.projector());
        activeModes = std::max(activeModes, state.modeCount());
        const double defect = state.purityDefect();
        worstPurityDefect = std::isfinite(worstPurityDefect)
                                ? std::max(worstPurityDefect, defect)
                                : defect;
        bandStates[index] = std::move(state);
      } catch (const std::exception &) {
        // A rank-deficient or malformed projector leaves the state unknown.
      }
      break;  // one carried state per component read
    }
  }
  const bool carriedStatePresent = hasCarriedState();
  const double carriedPurityDefect = carriedStatePurityDefect();

  // ── §17.5 the lazy Fock expression — ORACLE / non-Gaussian only ──────
  //
  // Built ONLY when the oracle is selected, and built for real: the #771
  // lazy Slater DAG of the first accepted band's projector, reporting its own
  // node count, discarded norm, and exactness. Never the quasi-free
  // production representation — the covariance above is that. A band wider
  // than the DAG's support cap is reported ABSENT with its reason, not
  // silently claimed.
  std::size_t oracleModes = 0;
  std::size_t oracleNodes = 0;
  bool oracleExact = false;
  double oracleDiscardedNorm = std::numeric_limits<double>::quiet_NaN();
  std::string oracleReason;
  bool oraclePresent = false;
  if (analysisConfig_.fockOracle) {
    oracleReason = "no accepted band";
    for (std::size_t index = 0; index < bandReads.size(); ++index) {
      if (candidateBand[index] < 0) continue;
      const auto &fiber =
          bandReads[index].fibers[static_cast<std::size_t>(candidateBand[index])];
      const Eigen::MatrixXcd projector = fiber.projector();
      const auto modeCount = static_cast<std::size_t>(projector.rows());
      if (modeCount > ::tessera::quantum::LazyFockEngine::kMaxSupportModes) {
        oracleReason = "band support exceeds the lazy DAG mode cap";
        continue;
      }
      std::vector<std::size_t> modes(modeCount);
      for (std::size_t mode = 0; mode < modeCount; ++mode) modes[mode] = mode;
      try {
        const ::tessera::quantum::LazyFockEngine engine(modeCount);
        const auto slater =
            engine.slaterFromProjector(modes, projector, 1e-9);
        oraclePresent = slater.state.valid();
        oracleModes = modeCount;
        oracleNodes = oraclePresent ? slater.state.nodeCount() : 0;
        oracleDiscardedNorm =
            oraclePresent ? slater.state.discardedNorm()
                          : std::numeric_limits<double>::quiet_NaN();
        oracleExact = engine.exactMode();
        oracleReason.clear();
      } catch (const std::exception &error) {
        oracleReason = error.what();
      }
      break;
    }
  }

  // ── §17.6 particle reads ─────────────────────────────────────────────
  const ParticleClusters classifier;
  std::vector<QuarkRead> quarkReads;
  std::vector<std::size_t> quarkBandRead;
  std::vector<::tessera::observables::BoundSupercomponentRead> bindings;
  std::vector<::tessera::observables::BaryonRead> baryonReads;
  for (std::size_t index = 0; index < bandReads.size(); ++index) {
    if (candidateBand[index] < 0) continue;
    const auto &bandRead = bandReads[index];
    const auto &fiber =
        bandRead.fibers[static_cast<std::size_t>(candidateBand[index])];
    QuarkCandidateEvidence evidence;
    const auto &componentId = components[bandComponent[index]].id;
    evidence.component = componentId;
    evidence.colorBand = fiber;
    // The calibrated oriented-triangle anchor, when the band's cells carry a
    // triangle atlas (a rank-three degree-one band). Missing evidence is a
    // NAMED failed certificate downstream — never a presumed pass, and the
    // LIFETIME transport family is deliberately left unsupplied: one pass is
    // one frame, and a world tube needs more than one.
    if (bandRead.degree == 1 && fiber.rank() == 3)
      evidence.anchor = BandAnchor::of(spacetime, fiber);
    if (bandStates[index].has_value()) {
      evidence.parityRead = bandStates[index]->wickParity();
      evidence.occupationRead = bandStates[index]->wickTotalNumber();
    }
    for (const auto &track : report.tracks)
      for (const auto &member : track.members)
        if (member == componentId) {
          evidence.persistenceLifetime =
              static_cast<double>(track.lastSlice - track.firstSlice + 1);
          evidence.persistenceMinOverlap = track.minAdjacentOverlap;
        }
    // This driver reads ONE cobordism frame, so the frame lifetime is one
    // and its adjacent-frame overlap is vacuously one (there is no adjacent
    // pair).  Both are MEASURED facts about this read, not placeholders:
    // the persistence certificate fails because the candidate was seen in a
    // single frame, which is a physical statement about the evidence, not
    // an artifact of reading a single modularity resolution.
    evidence.frameLifetime = 1.0;
    evidence.frameMinOverlap = 1.0;
    quarkReads.push_back(classifier.classifyQuarkCached(*cache, evidence));
    quarkBandRead.push_back(index);
  }
  if (!nextLevelComponents.empty() && !quarkReads.empty()) {
    std::vector<::tessera::observables::BoundCandidateEvidence> candidates;
    candidates.reserve(quarkReads.size());
    for (std::size_t index = 0; index < quarkReads.size(); ++index) {
      ::tessera::observables::BoundCandidateEvidence candidate;
      candidate.quark = quarkReads[index];
      const std::size_t readIndex = quarkBandRead[index];
      candidate.support = bandReads[readIndex].support;
      // The cross-component links ARE this candidate's mutual transports.
      for (const auto &record : transports)
        if (record.toRead == readIndex || record.fromRead == readIndex)
          candidate.mutualTransports.push_back(record.read);
      candidates.push_back(std::move(candidate));
    }
    bindings = classifier.boundSupercomponentSearch(nextLevelComponents,
                                                    candidates);
    // The three-cluster verdict itself: every binding
    // that grouped EXACTLY three CERTIFIED constituents is classified. The
    // evidence this pass HAS is the binding, those three #773 verdicts, and
    // the bound component's #765 lifetime; the color columns, the octet
    // flux, the #772 2π character, the #780 spin reads, the swept
    // covariance-only class and the refinement window are not read here, and
    // the classifier NAMES each of them as missing. That is the honest
    // report of one analysis pass on one frame — an unsupplied quantity is
    // unknown, never a fabricated pass.
    std::vector<double> boundLifetimes;
    boundLifetimes.reserve(bindings.size());
    for (const auto &binding : bindings) {
      double lifetime = std::numeric_limits<double>::quiet_NaN();
      for (const auto &track : report.tracks)
        for (const auto &member : track.members)
          if (member == binding.boundComponent)
            lifetime =
                static_cast<double>(track.lastSlice - track.firstSlice + 1);
      boundLifetimes.push_back(lifetime);
    }
    baryonReads = classifier.classifyBoundSupercomponents(
        bindings, quarkReads, boundLifetimes);
  }

  // ── the checkpoint (schema version 4) ────────────────────────────────
  std::string hierarchyText = "[";
  for (std::size_t sliceIndex = 0; sliceIndex < report.slices.size();
       ++sliceIndex) {
    const auto &slice = report.slices[sliceIndex];
    if (sliceIndex) hierarchyText += ", ";
    std::string componentText = "[";
    for (std::size_t index = 0; index < slice.components.size(); ++index) {
      if (index) componentText += ", ";
      const auto &component = slice.components[index];
      componentText += Json::object({
          {"id", Json::str(component.id.canonicalHash())},
          {"level", Json::integer(
                        static_cast<long long>(component.id.level()))},
          {"support", Json::idArray(component.support)},
          {"strength", Json::complexPair(component.strength)},
          {"conductance", Json::number(component.conductance)},
          {"modularity_contribution",
           Json::complexPair(component.modularityContribution)},
      });
    }
    componentText += "]";
    hierarchyText += Json::object({
        {"gamma", Json::number(slice.gamma)},
        {"q", Json::complexPair(slice.q)},
        {"objective_value", Json::number(slice.objectiveValue)},
        {"levels", Json::integer(static_cast<long long>(slice.levels))},
        {"restart_spread", Json::number(slice.restartSpread)},
        {"components", componentText},
    });
  }
  hierarchyText += "]";

  std::string fiberText = "[";
  for (std::size_t index = 0; index < bandReads.size(); ++index) {
    const auto &bandRead = bandReads[index];
    for (const auto &fiber : bandRead.fibers) {
      if (fiberText.size() > 1) fiberText += ", ";
      const auto &certificate = fiber.certificate();
      fiberText += Json::object({
          {"component",
           Json::str(components[bandComponent[index]].id.canonicalHash())},
          {"degree", Json::integer(fiber.degree())},
          {"rank", Json::integer(static_cast<long long>(fiber.rank()))},
          {"accepted", Json::boolean(fiber.accepted())},
          {"self_adjoint", Json::boolean(certificate.selfAdjoint)},
          {"lower_gap", Json::number(certificate.lowerGap)},
          {"upper_gap", Json::number(certificate.upperGap)},
          {"nearest_discarded_separation",
           Json::number(certificate.nearestDiscardedSeparation)},
          {"localization", Json::number(certificate.localization)},
          {"localization_support_fraction",
           Json::number(certificate.localizationSupportFraction)},
          {"localization_excess",
           Json::number(certificate.localizationExcess)},
          {"projector_residual", Json::number(certificate.projectorResidual)},
          {"eigen_residual", Json::number(certificate.eigenResidual)},
          {"gram_defect", Json::number(certificate.gramDefect)},
          {"projector_norm", Json::number(certificate.projectorNorm)},
          {"frame_condition_number",
           Json::number(certificate.frameConditionNumber)},
          {"band_center", Json::complexPair(fiber.bandCenter())},
      });
    }
  }
  fiberText += "]";

  std::string labeledSumText = "[";
  for (std::size_t index = 0; index < labeledSums.size(); ++index) {
    if (index) labeledSumText += ", ";
    const auto &sum = labeledSums[index].second;
    labeledSumText += Json::object({
        {"degree", Json::integer(
                       static_cast<long long>(labeledSums[index].first))},
        {"nominal_rank", Json::integer(
                             static_cast<long long>(sum.nominalRank))},
        {"effective_rank", Json::integer(
                               static_cast<long long>(sum.effectiveRank))},
        {"gram_defect", Json::number(sum.gramDefect)},
        {"quotient_nullity",
         Json::integer(static_cast<long long>(sum.quotientNullity))},
        {"certificate", Json::certificate(sum.certificate)},
    });
  }
  labeledSumText += "]";

  std::string transportText = "[";
  for (std::size_t index = 0; index < transports.size(); ++index) {
    if (index) transportText += ", ";
    const auto &read = transports[index].read;
    transportText += Json::object({
        {"degree", Json::integer(read.degree)},
        {"rank", Json::integer(read.rank)},
        {"numerical_rank", Json::integer(read.numericalRank)},
        {"accepted", Json::boolean(read.accepted)},
        {"leakage", Json::number(read.leakage)},
        {"overlap_condition_number",
         Json::number(read.overlapConditionNumber)},
        {"frame_condition_number", Json::number(read.frameConditionNumber)},
        {"to_gap", Json::number(read.toGap)},
        {"from_gap", Json::number(read.fromGap)},
        {"determinant_phase", Json::complexPair(read.determinantPhase)},
    });
  }
  transportText += "]";

  std::string quarkText = "[";
  for (std::size_t index = 0; index < quarkReads.size(); ++index) {
    if (index) quarkText += ", ";
    const auto &read = quarkReads[index];
    quarkText += Json::object({
        {"component", Json::str(read.component.canonicalHash())},
        {"classification", Json::str(read.classification)},
        {"confidence", Json::number(read.confidence)},
        {"color_rank", Json::integer(read.colorRank)},
        {"exterior_parity", Json::integer(read.exteriorParity)},
        {"triangle_anchor_score", Json::number(read.triangleAnchorScore)},
        {"anchor_weighting_id", Json::str(read.anchorWeightingId)},
        {"determinant_winding", Json::optional(read.determinantWinding)},
        {"winding_closure", Json::str(read.windingClosure)},
        {"baryon_flux", Json::optional(read.baryonFlux)},
        {"isospin", Json::optional(read.isospin)},
        {"electric_flux", Json::optional(read.electricFlux)},
        {"occupation_total", Json::number(read.occupationTotal)},
        {"transport_count",
         Json::integer(static_cast<long long>(read.transportCount))},
        {"transport_leakage_max", Json::number(read.transportLeakageMax)},
        {"persistence_lifetime", Json::number(read.persistenceLifetime)},
        {"frame_lifetime", Json::number(read.frameLifetime)},
        {"localization", Json::number(read.localization)},
        {"localization_support_fraction",
         Json::number(read.localizationSupportFraction)},
        {"failed_certificates", Json::stringArray(read.failedCertificates)},
    });
  }
  quarkText += "]";

  std::string bindingText = "[";
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    if (index) bindingText += ", ";
    const auto &binding = bindings[index];
    bindingText += Json::object({
        {"bound_component", Json::str(binding.boundComponent.canonicalHash())},
        {"found", Json::boolean(binding.found)},
        {"constituents",
         Json::integer(static_cast<long long>(binding.quarks.size()))},
        {"lifetime_overlap", Json::number(binding.lifetimeOverlap)},
        {"min_containment", Json::number(binding.minContainment)},
        {"transport_leakage_max", Json::number(binding.transportLeakageMax)},
        {"failed_certificates",
         Json::stringArray(binding.failedCertificates)},
    });
  }
  bindingText += "]";

  // `particles.baryons` carries the §16.4 THREE-CLUSTER VERDICT — one
  // `BaryonRead` per binding that grouped exactly three certified
  // constituents — in the `BaryonRead::toRecord` field vocabulary. Unknown
  // and uncertified values are `null`, never zero, and every gap is NAMED in
  // `failed_certificates`.
  std::string baryonText = "[";
  for (std::size_t index = 0; index < baryonReads.size(); ++index) {
    if (index) baryonText += ", ";
    const auto &read = baryonReads[index];
    std::string quarkIdText = "[";
    for (std::size_t leg = 0; leg < read.quarks.size(); ++leg) {
      if (leg) quarkIdText += ", ";
      quarkIdText += Json::str(read.quarks[leg].canonicalHash());
    }
    quarkIdText += "]";
    baryonText += Json::object({
        {"bound_component", Json::str(read.boundComponent.canonicalHash())},
        {"quarks", quarkIdText},
        {"classification", Json::str(read.classification)},
        {"confidence", Json::number(read.confidence)},
        {"color_gram_determinant", Json::number(read.colorGramDeterminant)},
        {"color_wedge", Json::complexPair(read.colorWedge)},
        {"color_flux", Json::number(read.colorFlux)},
        {"baryon_flux", Json::optional(read.baryonFlux)},
        {"electric_flux", Json::optional(read.electricFlux)},
        {"total_winding", Json::optional(read.totalWinding)},
        {"exterior_parity", Json::integer(read.exteriorParity)},
        {"flavor_pattern", Json::str(read.flavorPattern)},
        {"total_isospin", Json::optional(read.totalIsospin)},
        {"total_j2", Json::optional(read.totalJ2)},
        {"total_j2_variance", Json::optional(read.totalJ2Variance)},
        {"sharp_spin", Json::boolean(read.sharpSpin)},
        {"quasi_free_class_swept",
         Json::boolean(read.quasiFreeClassSwept)},
        {"class_variance_floor", Json::number(read.classVarianceFloor)},
        {"rotation_character", Json::optional(read.rotationCharacter)},
        {"rotation_character_sign",
         Json::integer(read.rotationCharacterSign)},
        {"exchange_character", Json::optional(read.exchangeCharacter)},
        {"spin_statistics_ratio", Json::optional(read.spinStatisticsRatio)},
        {"spin_lift_applicable", Json::boolean(read.spinLiftApplicable)},
        {"spin_lift_accepted", Json::boolean(read.spinLiftAccepted)},
        {"radius", Json::number(read.radius)},
        {"radius_finite", Json::boolean(read.radiusFinite)},
        {"spectral_mass", Json::number(read.spectralMass)},
        {"radius_ratio", Json::number(read.radiusRatio)},
        {"profile_max_deviation", Json::number(read.profileMaxDeviation)},
        {"profile_stable", Json::boolean(read.profileStable)},
        {"physical_mass", Json::optional(read.physicalMass)},
        {"persistence", Json::number(read.persistence)},
        {"lifetime_overlap", Json::number(read.lifetimeOverlap)},
        {"transport_count",
         Json::integer(static_cast<long long>(read.transportCount))},
        {"transport_leakage_max", Json::number(read.transportLeakageMax)},
        {"failed_certificates", Json::stringArray(read.failedCertificates)},
        {"certificate", Json::certificate(read.certificate)},
    });
  }
  baryonText += "]";

  std::string staticCertificateText = "[";
  for (std::size_t index = 0; index < staticCertificates.size(); ++index) {
    if (index) staticCertificateText += ", ";
    staticCertificateText += Json::object({
        {"degree", Json::integer(
                       static_cast<long long>(staticCertificates[index].first))},
        {"certificate", Json::certificate(staticCertificates[index].second)},
    });
  }
  staticCertificateText += "]";

  std::vector<double> resolutionList = modularityConfig.resolutions;
  std::vector<int> degreeList = fiberConfig.degrees;

  checkpointJson_ = Json::object({
      {"schema_version", Json::integer(checkpointSchemaVersion())},
      {"mode", Json::str(modeName(simulationMode_))},
      {"emergence_submode", Json::str(submodeName(emergenceSubmode_))},
      {"geometry_revision",
       Json::integer(static_cast<long long>(spacetime->metricRevisionKey()))},
      {"raw_complex", rawComplexJson(spacetime)},
      {"edge_quantum_data",
       Json::object({
           {"carried_state_present", Json::boolean(carriedStatePresent)},
           {"carried_state_degree", Json::integer(carriedStateDegree_)},
           {"carried_mode_count",
            Json::integer(static_cast<long long>(carriedModeCells_.size()))},
       })},
      {"objective",
       Json::object({
           {"total", Json::number(objectiveOf(terms))},
           {"regge_stationarity", Json::number(terms.reggeStationarity)},
           {"hodge_stationarity", Json::number(terms.hodgeStationarity)},
           {"connection_stationarity",
            Json::number(terms.connectionStationarity)},
           {"register_residual", Json::number(terms.registerResidual)},
           {"action_magnitude", Json::number(terms.actionMagnitude)},
           {"carried_state_energy", Json::number(terms.carriedStateEnergy)},
           {"regge_weight", Json::number(reggeWeight_)},
           {"hodge_entropy_weight", Json::number(hodgeEntropyWeight_)},
           {"connection_entropy_weight",
            Json::number(connectionEntropyWeight_)},
           {"carried_state_energy_weight",
            Json::number(carriedStateEnergyWeight_)},
           {"mean_field_dt", Json::number(meanFieldStepSize_)},
           {"mean_field_steps", Json::integer(meanFieldSteps_)},
       })},
      {"refinement",
       Json::object({
           {"refine", Json::boolean(decision.refine)},
           {"trigger", Json::str(decision.trigger)},
           {"regge_stationarity_residual",
            Json::number(indicators.reggeStationarityResidual)},
           {"hodge_stationarity_residual",
            Json::number(indicators.hodgeStationarityResidual)},
           {"curvature_concentration",
            Json::number(indicators.curvatureConcentration)},
           {"mesh_quality", Json::number(indicators.meshQuality)},
           {"solver_error", Json::number(indicators.solverError)},
       })},
      {"hierarchy", hierarchyText},
      {"invalidated_ancestry",
       Json::object({
           {"components",
            Json::integer(
                static_cast<long long>(invalidated.components.size()))},
           {"component_ids",
            Json::array(invalidated.components,
                        [](const ::tessera::observables::ComponentId &id) {
                          return Json::str(id.canonicalHash());
                        })},
           {"total_components", Json::integer(
                                    static_cast<long long>(totalComponents))},
           {"touched_vertices",
            Json::integer(static_cast<long long>(touchedCells.size()))},
           {"tracks",
            Json::integer(static_cast<long long>(invalidated.tracks.size()))},
       })},
      {"fibers", fiberText},
      {"labeled_fiber_sums", labeledSumText},
      {"transports", transportText},
      {"covariance",
       Json::object({
           {"active_modes", Json::integer(
                                static_cast<long long>(activeModes))},
           {"number_conserving", Json::boolean(true)},
           {"purity_defect", Json::number(worstPurityDefect)},
           {"carried_purity_defect", Json::number(carriedPurityDefect)},
           {"matrix_sidecar", Json::str("")},
       })},
      {"fock_oracle",
       Json::object({
           {"present", Json::boolean(oraclePresent)},
           {"active_modes", Json::integer(
                                static_cast<long long>(oracleModes))},
           {"nodes", Json::integer(static_cast<long long>(oracleNodes))},
           {"exact", Json::boolean(oracleExact)},
           {"discarded_norm", Json::number(oracleDiscardedNorm)},
           {"absent_reason",
            oracleReason.empty() ? std::string("null")
                                 : Json::str(oracleReason)},
       })},
      {"particles",
       Json::object({
           {"quarks", quarkText},
           {"gluons", "[]"},
           {"bound_supercomponents", bindingText},
           {"baryons", baryonText},
       })},
      {"certificates",
       Json::object({
           {"static_reduction", staticCertificateText},
           {"covariance_purity_holds",
            carriedStatePresent ? Json::boolean(carriedStatePurityHolds())
                                : std::string("null")},
       })},
      {"analysis",
       Json::object({
           {"pass", Json::integer(
                        static_cast<long long>(analysisPassCount_))},
           {"accepted_moves",
            Json::integer(static_cast<long long>(acceptedMoveCount_))},
           {"cadence", Json::integer(analysisConfig_.cadence)},
           {"cold_caches", Json::boolean(analysisConfig_.coldCaches)},
           {"fock_oracle", Json::boolean(analysisConfig_.fockOracle)},
           {"degrees", Json::array(degreeList,
                                   [](int degree) {
                                     return Json::integer(degree);
                                   })},
           {"resolutions", Json::array(resolutionList,
                                       [](double gamma) {
                                         return Json::number(gamma);
                                       })},
           {"cache_hits",
            Json::integer(static_cast<long long>(cache->hits() - hitsBefore))},
           {"cache_misses", Json::integer(static_cast<long long>(
                                cache->misses() - missesBefore))},
           {"cache_invalidations",
            Json::integer(static_cast<long long>(cache->invalidations() -
                                                 invalidationsBefore))},
           {"cache_entries",
            Json::integer(static_cast<long long>(cache->size()))},
       })},
      {"provenance",
       Json::object({
           {"seed", Json::integer(static_cast<long long>(seed_))},
           {"config_hash", Json::str(provenanceConfigHash_)},
           {"commit", Json::str(provenanceCommit_)},
       })},
  });
}

int MultiCobordism::checkpointVersionOf(const std::string &checkpoint) {
  const std::string version = Json::topLevelValue(checkpoint, "schema_version");
  if (version.empty())
    throw std::invalid_argument(
        "MultiCobordism checkpoint: no schema_version");
  return static_cast<int>(Json::asNumber(version));
}

std::string MultiCobordism::replayCheckpoint(const std::string &checkpoint) {
  const int version = checkpointVersionOf(checkpoint);
  if (version != checkpointSchemaVersion())
    throw std::invalid_argument(
        "MultiCobordism checkpoint: unknown schema_version " +
        std::to_string(version) + " (this build reads " +
        std::to_string(checkpointSchemaVersion()) + ")");

  const std::string rawComplex = Json::topLevelValue(checkpoint, "raw_complex");
  if (rawComplex.empty())
    throw std::invalid_argument(
        "MultiCobordism checkpoint: no raw_complex to replay");
  const int dimensions = static_cast<int>(
      Json::asNumber(Json::topLevelValue(rawComplex, "dimensions")));
  std::vector<std::vector<std::uint64_t>> cells;
  for (const auto &cellText :
       Json::elements(Json::topLevelValue(rawComplex, "cells"))) {
    std::vector<std::uint64_t> cell;
    for (const auto &idText : Json::elements(cellText))
      cell.push_back(static_cast<std::uint64_t>(Json::asNumber(idText)));
    cells.push_back(std::move(cell));
  }
  // The document IS a snapshot: the cells and, per edge on the canonical
  // min->max direction, the length and the phase. It is rebuilt through the
  // same `rebuild` a committed stage-1 move goes through, so replay and the
  // incremental run restore an edge identically. An edge the document
  // records without a phase takes `replayPhaseDefault`.
  Snapshot recorded;
  recorded.first = std::move(cells);
  for (const auto &edgeText :
       Json::elements(Json::topLevelValue(rawComplex, "edges"))) {
    const auto a = static_cast<std::uint64_t>(
        Json::asNumber(Json::topLevelValue(edgeText, "a")));
    const auto b = static_cast<std::uint64_t>(
        Json::asNumber(Json::topLevelValue(edgeText, "b")));
    const auto parts = Json::elements(Json::topLevelValue(edgeText, "length"));
    if (parts.size() != 2) continue;
    EdgeGeometry geometry{
        complexd{Json::asNumber(parts[0]), Json::asNumber(parts[1])},
        replayPhaseDefault()};
    const auto phaseParts =
        Json::elements(Json::topLevelValue(edgeText, "phase"));
    if (phaseParts.size() == 2)
      geometry.phase = complexd{Json::asNumber(phaseParts[0]),
                                Json::asNumber(phaseParts[1])};
    recorded.second[{std::min(a, b), std::max(a, b)}] = geometry;
  }
  auto spacetime = rebuild(dimensions, recorded);

  // Rebuild the node the checkpoint describes and recompute EVERYTHING cold.
  const std::string provenance = Json::topLevelValue(checkpoint, "provenance");
  const std::uint64_t seed =
      provenance.empty()
          ? 0
          : static_cast<std::uint64_t>(
                Json::asNumber(Json::topLevelValue(provenance, "seed")));
  const std::string analysis = Json::topLevelValue(checkpoint, "analysis");
  std::vector<int> degrees;
  std::vector<double> resolutions;
  bool fockOracle = false;
  if (!analysis.empty()) {
    for (const auto &text :
         Json::elements(Json::topLevelValue(analysis, "degrees")))
      degrees.push_back(static_cast<int>(Json::asNumber(text)));
    for (const auto &text :
         Json::elements(Json::topLevelValue(analysis, "resolutions")))
      resolutions.push_back(Json::asNumber(text));
    const std::string oracle = Json::topLevelValue(analysis, "fock_oracle");
    fockOracle = oracle == "true";
  }
  if (degrees.empty()) degrees.push_back(1);
  if (resolutions.empty()) resolutions.push_back(1.0);

  MultiCobordism replayed(spacetime, {}, {}, {degrees.back()}, 1.0, seed);
  replayed.setObjective(std::make_shared<JointStationarityObjective>());
  replayed.setSimulationMode(SimulationMode::Replay);
  AnalysisConfig config;
  config.enabled = true;
  config.degrees = degrees;
  config.resolutions = resolutions;
  config.fockOracle = fockOracle;
  config.coldCaches = true;  // serve NOTHING: the cold path, by construction
  replayed.setAnalysisConfig(config);
  if (!provenance.empty()) {
    const std::string configHash =
        Json::topLevelValue(provenance, "config_hash");
    const std::string commit = Json::topLevelValue(provenance, "commit");
    replayed.setProvenance(configHash.empty() ? "" : Json::asString(configHash),
                           commit.empty() ? "" : Json::asString(commit));
  }
  replayed.runRecursiveAnalysis();
  return replayed.checkpointJson();
}

}  // namespace tessera::cobordism
