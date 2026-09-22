// Lazy graded Fock oracle and boundary carrier — implementation of
// include/quantum/LazyFock.h.
//
// Exactness spine: every amplitude is evaluated through the Koszul sign
// ε(A,B) = (−1)^{#inversions between disjoint occupied sets}, computed
// against the single global compilation order. The bit-level canonical
// anticommutation relation (CAR) signs are OccupationBitset's — nothing
// fermionic is re-derived here. Subset sums
// delegate to cobordism::OccupationSpectra; certificates are
// cobordism::Certificate; content hashes chain mesh::Fingerprint::mix64
// (order-sensitive chaining — fingerprintOf's XOR set hash is not used,
// because expression content is order-sensitive).

#include "quantum/LazyFock.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "cobordism/OccupationSpectra.h"
#include "mesh/Fingerprint.h"

namespace tessera::quantum {

namespace {

using Complex = std::complex<double>;
using Term = LazyFockNode::Term;

std::atomic<std::uint64_t> g_nodeCounter{1};

/// File-local bit/sign utilities (static-only helper class per the
/// no-free-function convention).
struct Bits {
    /// Strict total order on equal-universe bitsets (highest chunk first).
    static bool keyLess(const OccupationBitset& x, const OccupationBitset& y) {
        const auto& cx = x.chunks();
        const auto& cy = y.chunks();
        for (std::size_t i = cx.size(); i-- > 0;)
            if (cx[i] != cy[i]) return cx[i] < cy[i];
        return false;
    }

    static std::uint64_t keyHash(const OccupationBitset& x) {
        std::uint64_t h = 0x243f6a8885a308d3ull;
        for (std::uint64_t c : x.chunks())
            h = mesh::Fingerprint::mix64(h ^ mesh::Fingerprint::mix64(c));
        return h;
    }

    /// (−1)^{#{(i,j) ∈ a×b : i > j}} for sorted disjoint mode lists — the
    /// Koszul sign of merging two ascending wedge words.
    static int koszulSign(const std::vector<std::size_t>& a,
                          const std::vector<std::size_t>& b) {
        std::size_t inversions = 0;
        std::size_t k = 0;
        for (std::size_t j : b) {
            while (k < a.size() && a[k] < j) ++k;
            inversions += a.size() - k;
        }
        return (inversions % 2 == 0) ? +1 : -1;
    }

    /// Whether sorted `sub` is a subset of sorted `super`.
    static bool isSubset(const std::vector<std::size_t>& sub,
                         const std::vector<std::size_t>& super) {
        std::size_t k = 0;
        for (std::size_t m : sub) {
            while (k < super.size() && super[k] < m) ++k;
            if (k == super.size() || super[k] != m) return false;
            ++k;
        }
        return true;
    }

    static bool contains(const std::vector<std::size_t>& sorted,
                         std::size_t mode) {
        return std::binary_search(sorted.begin(), sorted.end(), mode);
    }

    /// Occupied modes of `key` restricted to the sorted list `modes`.
    static std::vector<std::size_t> occupiedIn(
        const OccupationBitset& key, const std::vector<std::size_t>& modes) {
        std::vector<std::size_t> out;
        for (std::size_t m : key.occupiedModes())
            if (contains(modes, m)) out.push_back(m);
        return out;
    }

    /// Support-local Fock index of `key` (bit k = occupation of
    /// supportModes[k]) plus the occupied-support list.
    static std::size_t supportIndex(const OccupationBitset& key,
                                    const std::vector<std::size_t>& support) {
        std::size_t idx = 0;
        for (std::size_t k = 0; k < support.size(); ++k)
            if (key.test(support[k])) idx |= (std::size_t{1} << k);
        return idx;
    }

    /// The global occupied-mode list of a support-local Fock index.
    static std::vector<std::size_t> supportOccupied(
        std::size_t index, const std::vector<std::size_t>& support) {
        std::vector<std::size_t> out;
        for (std::size_t k = 0; k < support.size(); ++k)
            if (index & (std::size_t{1} << k)) out.push_back(support[k]);
        return out;
    }
};

struct KeyHasher {
    std::size_t operator()(const OccupationBitset& k) const noexcept {
        return static_cast<std::size_t>(Bits::keyHash(k));
    }
};
struct KeyEq {
    bool operator()(const OccupationBitset& a,
                    const OccupationBitset& b) const noexcept {
        return a == b;
    }
};
using TermMap = std::unordered_map<OccupationBitset, Complex, KeyHasher, KeyEq>;

/// Order-sensitive content-hash chain over mix64 (see header rationale).
struct HashChain {
    std::uint64_t h;
    explicit HashChain(std::uint64_t seed)
        : h(mesh::Fingerprint::mix64(seed)) {}
    void mix(std::uint64_t x) {
        h = mesh::Fingerprint::mix64(h ^ mesh::Fingerprint::mix64(x));
    }
    void mixDouble(double x) { mix(std::bit_cast<std::uint64_t>(x)); }
    void mixComplex(const Complex& z) {
        mixDouble(z.real());
        mixDouble(z.imag());
    }
};

void requireFinite(const Complex& z, const char* what) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument(std::string(what) +
                                    ": non-finite amplitude");
}

/// Spectral-norm upper bound of a small operator: exact ‖·‖₂ (SVD) when
/// the dense form is affordable, else the Frobenius bound ‖·‖_F ≥ ‖·‖₂.
double operatorNormBound(const LazyFockNode::SparseColOp& op) {
    if (op.rows() <= 512) {
        Eigen::MatrixXcd dense(op);
        if (dense.size() == 0) return 0.0;
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(dense);
        return svd.singularValues().size() > 0 ? svd.singularValues()(0) : 0.0;
    }
    double frob2 = 0.0;
    for (int k = 0; k < op.outerSize(); ++k)
        for (LazyFockNode::SparseColOp::InnerIterator it(op, k); it; ++it)
            frob2 += std::norm(it.value());
    return std::sqrt(frob2);
}

double matrixSpectralBound(const Eigen::MatrixXcd& m) {
    if (m.size() == 0) return 0.0;
    if (m.rows() <= 512) {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(m);
        return svd.singularValues().size() > 0 ? svd.singularValues()(0) : 0.0;
    }
    return m.norm();  // Frobenius upper bound
}

// ── Minimal strict-JSON reader for the serialization schema ─────────────
// (objects, arrays, strings with basic escapes, doubles, ints, bools).
// File-local by design; the writer below emits exactly this subset.
class MiniJson {
  public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type{Type::Null};
    bool boolean{false};
    double number{0.0};
    std::string str{};
    std::vector<MiniJson> arr{};
    std::vector<std::pair<std::string, MiniJson>> obj{};

    static MiniJson parse(const std::string& text) {
        std::size_t pos = 0;
        MiniJson v = parseValue(text, pos);
        skipWs(text, pos);
        if (pos != text.size())
            throw std::invalid_argument("LazyFock deserialize: trailing data");
        return v;
    }

    const MiniJson& at(const std::string& key) const {
        for (const auto& [k, v] : obj)
            if (k == key) return v;
        throw std::invalid_argument("LazyFock deserialize: missing field '" +
                                    key + "'");
    }
    bool has(const std::string& key) const {
        for (const auto& [k, v] : obj)
            if (k == key) return true;
        return false;
    }
    double asNumber() const {
        if (type != Type::Number)
            throw std::invalid_argument("LazyFock deserialize: expected number");
        return number;
    }
    std::size_t asIndex() const {
        double n = asNumber();
        if (n < 0 || n != std::floor(n))
            throw std::invalid_argument(
                "LazyFock deserialize: expected non-negative integer");
        return static_cast<std::size_t>(n);
    }
    const std::string& asString() const {
        if (type != Type::String)
            throw std::invalid_argument("LazyFock deserialize: expected string");
        return str;
    }
    const std::vector<MiniJson>& asArray() const {
        if (type != Type::Array)
            throw std::invalid_argument("LazyFock deserialize: expected array");
        return arr;
    }

  private:
    static void skipWs(const std::string& t, std::size_t& p) {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\n' || t[p] == '\t' ||
                                t[p] == '\r'))
            ++p;
    }
    static MiniJson parseValue(const std::string& t, std::size_t& p) {
        skipWs(t, p);
        if (p >= t.size())
            throw std::invalid_argument("LazyFock deserialize: truncated");
        char c = t[p];
        if (c == '{') return parseObject(t, p);
        if (c == '[') return parseArray(t, p);
        if (c == '"') return parseString(t, p);
        if (c == 't' || c == 'f') return parseBool(t, p);
        if (c == 'n') {
            expect(t, p, "null");
            return MiniJson{};
        }
        return parseNumber(t, p);
    }
    static void expect(const std::string& t, std::size_t& p,
                       const char* word) {
        for (const char* w = word; *w; ++w, ++p)
            if (p >= t.size() || t[p] != *w)
                throw std::invalid_argument(
                    "LazyFock deserialize: malformed literal");
    }
    static MiniJson parseBool(const std::string& t, std::size_t& p) {
        MiniJson v;
        v.type = Type::Bool;
        if (t[p] == 't') {
            expect(t, p, "true");
            v.boolean = true;
        } else {
            expect(t, p, "false");
            v.boolean = false;
        }
        return v;
    }
    static MiniJson parseNumber(const std::string& t, std::size_t& p) {
        const char* start = t.c_str() + p;
        char* end = nullptr;
        double d = std::strtod(start, &end);
        if (end == start)
            throw std::invalid_argument("LazyFock deserialize: bad number");
        p += static_cast<std::size_t>(end - start);
        MiniJson v;
        v.type = Type::Number;
        v.number = d;
        return v;
    }
    static MiniJson parseString(const std::string& t, std::size_t& p) {
        ++p;  // opening quote
        std::string out;
        while (p < t.size() && t[p] != '"') {
            char c = t[p++];
            if (c == '\\') {
                if (p >= t.size())
                    throw std::invalid_argument(
                        "LazyFock deserialize: truncated escape");
                char e = t[p++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': {
                        if (p + 4 > t.size())
                            throw std::invalid_argument(
                                "LazyFock deserialize: truncated \\u");
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = t[p++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f')
                                code |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F')
                                code |= unsigned(h - 'A' + 10);
                            else
                                throw std::invalid_argument(
                                    "LazyFock deserialize: bad \\u digit");
                        }
                        if (code > 0x7f)
                            throw std::invalid_argument(
                                "LazyFock deserialize: non-ASCII \\u escape "
                                "unsupported");
                        out.push_back(static_cast<char>(code));
                        break;
                    }
                    default:
                        throw std::invalid_argument(
                            "LazyFock deserialize: bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        if (p >= t.size())
            throw std::invalid_argument("LazyFock deserialize: unterminated string");
        ++p;  // closing quote
        MiniJson v;
        v.type = Type::String;
        v.str = std::move(out);
        return v;
    }
    static MiniJson parseArray(const std::string& t, std::size_t& p) {
        ++p;
        MiniJson v;
        v.type = Type::Array;
        skipWs(t, p);
        if (p < t.size() && t[p] == ']') {
            ++p;
            return v;
        }
        while (true) {
            v.arr.push_back(parseValue(t, p));
            skipWs(t, p);
            if (p >= t.size())
                throw std::invalid_argument(
                    "LazyFock deserialize: unterminated array");
            if (t[p] == ',') {
                ++p;
                continue;
            }
            if (t[p] == ']') {
                ++p;
                return v;
            }
            throw std::invalid_argument("LazyFock deserialize: bad array");
        }
    }
    static MiniJson parseObject(const std::string& t, std::size_t& p) {
        ++p;
        MiniJson v;
        v.type = Type::Object;
        skipWs(t, p);
        if (p < t.size() && t[p] == '}') {
            ++p;
            return v;
        }
        while (true) {
            skipWs(t, p);
            if (p >= t.size() || t[p] != '"')
                throw std::invalid_argument("LazyFock deserialize: bad key");
            MiniJson key = parseString(t, p);
            skipWs(t, p);
            if (p >= t.size() || t[p] != ':')
                throw std::invalid_argument("LazyFock deserialize: missing ':'");
            ++p;
            v.obj.emplace_back(key.str, parseValue(t, p));
            skipWs(t, p);
            if (p >= t.size())
                throw std::invalid_argument(
                    "LazyFock deserialize: unterminated object");
            if (t[p] == ',') {
                ++p;
                continue;
            }
            if (t[p] == '}') {
                ++p;
                return v;
            }
            throw std::invalid_argument("LazyFock deserialize: bad object");
        }
    }
};

/// JSON writer helpers (17 significant digits — exact double round-trip).
struct JsonOut {
    static void number(std::ostringstream& os, double x) {
        os << std::setprecision(17) << x;
    }
    static void string(std::ostringstream& os, const std::string& s) {
        os << '"';
        for (char c : s) {
            switch (c) {
                case '"': os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\n': os << "\\n"; break;
                case '\t': os << "\\t"; break;
                case '\r': os << "\\r"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        os << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << int(c) << std::dec
                           << std::setfill(' ');
                    } else {
                        os << c;
                    }
            }
        }
        os << '"';
    }
    static std::string hex64(std::uint64_t v) {
        std::ostringstream os;
        os << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
        return os.str();
    }
    static std::uint64_t parseHex64(const std::string& s) {
        if (s.size() != 18 || s[0] != '0' || s[1] != 'x')
            throw std::invalid_argument("LazyFock deserialize: bad hash literal");
        std::uint64_t v = 0;
        for (std::size_t i = 2; i < s.size(); ++i) {
            char h = s[i];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= std::uint64_t(h - '0');
            else if (h >= 'a' && h <= 'f') v |= std::uint64_t(h - 'a' + 10);
            else
                throw std::invalid_argument(
                    "LazyFock deserialize: bad hash digit");
        }
        return v;
    }
};

const char* kindName(LazyNodeKind k) {
    switch (k) {
        case LazyNodeKind::Vacuum: return "vacuum";
        case LazyNodeKind::Occupation: return "occupation";
        case LazyNodeKind::GradedTensor: return "tensor";
        case LazyNodeKind::LocalMap: return "localMap";
        case LazyNodeKind::SectorSum: return "sectorSum";
        case LazyNodeKind::Wedge: return "wedge";
    }
    return "?";
}

}  // namespace

// ── internal structs ─────────────────────────────────────────────────────

/// A materialized sparse expansion: terms sorted by key. Exact with
/// respect to the stored node (read-path expansions never truncate).
struct LazyFockEngine::Expansion {
    std::vector<Term> terms{};
};

/// One operator-application request.
struct LazyFockEngine::OperatorSpec {
    enum class Kind { SupportMatrix, DGamma };
    Kind kind{Kind::SupportMatrix};
    std::vector<std::size_t> supportModes{};
    // SupportMatrix payload:
    LazyFockNode::SparseColOp colOp{};
    // DGamma payload: nonzero (globalRow, globalCol, value) hopping terms.
    std::vector<std::tuple<std::size_t, std::size_t, Complex>> hopping{};
    double normBound{0.0};
    /// +1 even, −1 odd, 0 mixed (parity homogeneity of the operator).
    int parity{+1};

    OperatorSpec negated() const {
        OperatorSpec out = *this;
        if (kind == Kind::SupportMatrix) {
            out.colOp = colOp * Complex(-1.0, 0.0);
        } else {
            for (auto& [r, c, v] : out.hopping) v = -v;
        }
        return out;
    }
};

// ── LazyFockState ────────────────────────────────────────────────────────

namespace {
void requireValidHandle(const std::shared_ptr<const LazyFockNode>& root) {
    if (!root)
        throw std::invalid_argument(
            "LazyFockState: default-constructed (invalid) handle");
}
}  // namespace

std::uint64_t LazyFockState::rootNodeId() const {
    requireValidHandle(root_);
    return root_->nodeId();
}
std::uint64_t LazyFockState::contentHash() const {
    requireValidHandle(root_);
    return root_->contentHash();
}
LazyNodeKind LazyFockState::kind() const {
    requireValidHandle(root_);
    return root_->kind();
}
std::vector<std::size_t> LazyFockState::modes() const {
    requireValidHandle(root_);
    return root_->modes();
}
std::vector<std::uint64_t> LazyFockState::childNodeIds() const {
    requireValidHandle(root_);
    std::vector<std::uint64_t> out;
    for (const auto& c : root_->children()) out.push_back(c->nodeId());
    return out;
}
std::vector<std::uint64_t> LazyFockState::childContentHashes() const {
    requireValidHandle(root_);
    std::vector<std::uint64_t> out;
    for (const auto& c : root_->children()) out.push_back(c->contentHash());
    return out;
}
std::size_t LazyFockState::nodeCount() const {
    requireValidHandle(root_);
    std::unordered_set<std::uint64_t> seen;
    std::vector<const LazyFockNode*> stack{root_.get()};
    while (!stack.empty()) {
        const LazyFockNode* n = stack.back();
        stack.pop_back();
        if (!seen.insert(n->nodeId()).second) continue;
        for (const auto& c : n->children()) stack.push_back(c.get());
    }
    return seen.size();
}
long long LazyFockState::definiteOccupation() const {
    requireValidHandle(root_);
    return root_->definiteOccupation();
}
int LazyFockState::definiteParity() const {
    requireValidHandle(root_);
    return root_->definiteParity();
}

// ── LazyFockEngine: construction / configuration ────────────────────────

LazyFockEngine::LazyFockEngine(std::size_t modeCount)
    : modeCount_(modeCount) {}

LazyFockEngine LazyFockEngine::fromRegistry(const EdgeModeRegistry& registry) {
    return LazyFockEngine(registry.modeCount());
}

std::uint64_t LazyFockEngine::stageDimension(std::size_t stageModeCount) {
    if (stageModeCount > 63)
        throw std::invalid_argument(
            "LazyFockEngine::stageDimension: 2^m overflows beyond m = 63; "
            "the engine carries such stages lazily instead of counting them");
    return std::uint64_t{1} << stageModeCount;
}

void LazyFockEngine::setTruncationThreshold(double threshold,
                                            double normTolerance) {
    if (!(threshold > 0.0) || !std::isfinite(threshold))
        throw std::invalid_argument(
            "LazyFockEngine::setTruncationThreshold: threshold must be a "
            "positive finite number (use clearTruncation for exact mode)");
    if (!(normTolerance >= 0.0) || !std::isfinite(normTolerance))
        throw std::invalid_argument(
            "LazyFockEngine::setTruncationThreshold: normTolerance must be "
            "a non-negative finite number");
    truncationThreshold_ = threshold;
    truncationNormTolerance_ = normTolerance;
}

void LazyFockEngine::clearTruncation() noexcept {
    truncationThreshold_ = 0.0;
    truncationNormTolerance_ = 0.0;
}

void LazyFockEngine::clearMemo() noexcept {
    memo_.clear();
    memoHits_ = 0;
    memoMisses_ = 0;
}

// ── node factories ───────────────────────────────────────────────────────

std::vector<std::size_t> LazyFockEngine::canonicalModes(
    const std::vector<std::size_t>& modes, const char* what) const {
    std::vector<std::size_t> out = modes;
    std::sort(out.begin(), out.end());
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] >= modeCount_)
            throw std::invalid_argument(std::string(what) +
                                        ": mode index out of range");
        if (i > 0 && out[i] == out[i - 1])
            throw std::invalid_argument(std::string(what) +
                                        ": duplicate mode index");
    }
    return out;
}

LazyFockEngine::NodePtr LazyFockEngine::makeVacuum(
    std::vector<std::size_t> modes) const {
    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::Vacuum;
    node->modes_ = std::move(modes);
    node->universeModeCount_ = modeCount_;
    node->definiteOccupation_ = 0;
    node->definiteParity_ = +1;
    node->nodeId_ = g_nodeCounter.fetch_add(1);
    HashChain h(0x5aull);
    h.mix(modeCount_);
    h.mix(node->modes_.size());
    for (std::size_t m : node->modes_) h.mix(m);
    node->contentHash_ = h.h;
    return node;
}

LazyFockEngine::NodePtr LazyFockEngine::makeOccupation(
    std::vector<std::size_t> modes, std::vector<Term> terms) const {
    // Canonicalize: sort by key, merge duplicates, drop exact zeros —
    // all algebraically lossless.
    std::sort(terms.begin(), terms.end(),
              [](const Term& a, const Term& b) {
                  return Bits::keyLess(a.first, b.first);
              });
    std::vector<Term> merged;
    merged.reserve(terms.size());
    for (auto& t : terms) {
        requireFinite(t.second, "LazyFockEngine occupation term");
        if (!merged.empty() && merged.back().first == t.first)
            merged.back().second += t.second;
        else
            merged.push_back(std::move(t));
    }
    std::vector<Term> nonzero;
    nonzero.reserve(merged.size());
    for (auto& t : merged)
        if (t.second != Complex(0.0, 0.0)) nonzero.push_back(std::move(t));

    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::Occupation;
    node->modes_ = std::move(modes);
    node->universeModeCount_ = modeCount_;
    node->terms_ = std::move(nonzero);

    // Validate coverage and derive definite sectors.
    long long occ = -2;  // -2 = unset, -1 = indefinite
    int par = 2;         // 2 = unset, 0 = indefinite
    for (const auto& [key, amp] : node->terms_) {
        for (std::size_t m : key.occupiedModes())
            if (!Bits::contains(node->modes_, m))
                throw std::invalid_argument(
                    "LazyFockEngine occupation term: occupied mode outside "
                    "the node's mode set");
        const long long n = static_cast<long long>(key.count());
        const int p = key.parity();
        occ = (occ == -2 || occ == n) ? n : -1;
        par = (par == 2 || par == p) ? p : 0;
    }
    node->definiteOccupation_ = (occ == -2) ? 0 : occ;   // empty block ~ 0
    node->definiteParity_ = (par == 2) ? +1 : par;
    node->nodeId_ = g_nodeCounter.fetch_add(1);

    HashChain h(0x0cull);
    h.mix(modeCount_);
    h.mix(node->modes_.size());
    for (std::size_t m : node->modes_) h.mix(m);
    h.mix(node->terms_.size());
    for (const auto& [key, amp] : node->terms_) {
        for (std::uint64_t c : key.chunks()) h.mix(c);
        h.mixComplex(amp);
    }
    node->contentHash_ = h.h;
    return node;
}

LazyFockEngine::NodePtr LazyFockEngine::makeTensor(NodePtr left,
                                                   NodePtr right) const {
    // Disjointness of the partition.
    {
        std::size_t i = 0, j = 0;
        const auto& a = left->modes();
        const auto& b = right->modes();
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j])
                throw std::invalid_argument(
                    "LazyFockEngine::gradedTensor: mode sets must be "
                    "disjoint");
            (a[i] < b[j]) ? ++i : ++j;
        }
    }
    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::GradedTensor;
    node->universeModeCount_ = modeCount_;
    node->modes_.resize(left->modes().size() + right->modes().size());
    std::merge(left->modes().begin(), left->modes().end(),
               right->modes().begin(), right->modes().end(),
               node->modes_.begin());
    if (left->definiteOccupation() >= 0 && right->definiteOccupation() >= 0)
        node->definiteOccupation_ =
            left->definiteOccupation() + right->definiteOccupation();
    if (left->definiteParity() != 0 && right->definiteParity() != 0)
        node->definiteParity_ = left->definiteParity() * right->definiteParity();
    node->children_ = {left, right};
    node->nodeId_ = g_nodeCounter.fetch_add(1);
    HashChain h(0x7eull);
    h.mix(modeCount_);
    h.mix(left->contentHash());
    h.mix(right->contentHash());
    node->contentHash_ = h.h;
    return node;
}

LazyFockEngine::NodePtr LazyFockEngine::makeSectorSum(
    std::vector<NodePtr> children, LazySectorKind kind) const {
    if (children.empty())
        throw std::invalid_argument(
            "LazyFockEngine::sectorSum: at least one child required");
    const auto& modes0 = children.front()->modes();
    std::vector<long long> labels;
    for (const auto& c : children) {
        if (c->modes() != modes0)
            throw std::invalid_argument(
                "LazyFockEngine::sectorSum: children must share one mode "
                "set");
        long long label;
        if (kind == LazySectorKind::Occupation) {
            label = c->definiteOccupation();
            if (label < 0)
                throw std::invalid_argument(
                    "LazyFockEngine::sectorSum: child has indefinite "
                    "occupation number");
        } else {
            label = c->definiteParity();
            if (label == 0)
                throw std::invalid_argument(
                    "LazyFockEngine::sectorSum: child has indefinite "
                    "parity");
        }
        labels.push_back(label);
    }
    // Canonical child order: ascending sector label.
    std::vector<std::size_t> order(children.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return labels[a] < labels[b];
    });
    std::vector<NodePtr> sorted;
    std::vector<long long> sortedLabels;
    for (std::size_t i : order) {
        if (!sortedLabels.empty() && sortedLabels.back() == labels[i])
            throw std::invalid_argument(
                "LazyFockEngine::sectorSum: two children share sector " +
                std::to_string(labels[i]) +
                " — a sector direct sum requires DISTINCT conserved "
                "sectors");
        sorted.push_back(children[i]);
        sortedLabels.push_back(labels[i]);
    }
    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::SectorSum;
    node->universeModeCount_ = modeCount_;
    node->modes_ = modes0;
    node->children_ = std::move(sorted);
    node->sectorKind_ = kind;
    node->sectorLabels_ = std::move(sortedLabels);
    if (node->children_.size() == 1) {
        node->definiteOccupation_ = node->children_[0]->definiteOccupation();
        node->definiteParity_ = node->children_[0]->definiteParity();
    } else if (kind == LazySectorKind::Occupation) {
        // Distinct N sectors: N indefinite; parity definite only if all
        // labels share popcount parity.
        bool sameParity = true;
        for (long long l : node->sectorLabels_)
            if (((l ^ node->sectorLabels_[0]) & 1) != 0) sameParity = false;
        if (sameParity)
            node->definiteParity_ =
                (node->sectorLabels_[0] % 2 == 0) ? +1 : -1;
    }
    node->nodeId_ = g_nodeCounter.fetch_add(1);
    HashChain h(0xd5ull);
    h.mix(modeCount_);
    h.mix(static_cast<std::uint64_t>(kind == LazySectorKind::Parity));
    h.mix(node->children_.size());
    for (std::size_t i = 0; i < node->children_.size(); ++i) {
        h.mix(node->children_[i]->contentHash());
        h.mix(static_cast<std::uint64_t>(node->sectorLabels_[i]));
    }
    node->contentHash_ = h.h;
    return node;
}

LazyFockEngine::NodePtr LazyFockEngine::makeLocalMap(
    NodePtr child, std::vector<std::size_t> supportModes,
    LazyFockNode::SparseColOp op) const {
    const std::size_t s = supportModes.size();
    const auto dim = static_cast<Eigen::Index>(std::size_t{1} << s);
    if (op.rows() != dim || op.cols() != dim)
        throw std::invalid_argument(
            "LazyFockEngine local map: operator must be 2^{|support|} "
            "square");
    if (!Bits::isSubset(supportModes, child->modes()))
        throw std::invalid_argument(
            "LazyFockEngine local map: support must lie inside the child's "
            "mode set (embedInVacuum first to extend coverage)");
    op.makeCompressed();
    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::LocalMap;
    node->universeModeCount_ = modeCount_;
    node->modes_ = child->modes();
    node->children_ = {child};
    node->supportModes_ = std::move(supportModes);
    node->supportCols_ = op;
    node->supportRows_ = op;  // conversion to row-major
    node->supportRows_.makeCompressed();
    node->operatorNormBound_ = operatorNormBound(op);

    // Occupation shift / parity homogeneity of the operator.
    bool any = false;
    long long shift = 0;
    bool shiftUniform = true;
    int opParity = +1;
    bool parityUniform = true;
    for (int k = 0; k < op.outerSize(); ++k)
        for (LazyFockNode::SparseColOp::InnerIterator it(op, k); it; ++it) {
            requireFinite(it.value(), "LazyFockEngine local map entry");
            const long long d =
                std::popcount(static_cast<std::uint64_t>(it.row())) -
                std::popcount(static_cast<std::uint64_t>(it.col()));
            const int p = (d % 2 == 0) ? +1 : -1;
            if (!any) {
                shift = d;
                opParity = p;
                any = true;
            } else {
                if (d != shift) shiftUniform = false;
                if (p != opParity) parityUniform = false;
            }
        }
    if (any && shiftUniform && child->definiteOccupation() >= 0) {
        const long long n = child->definiteOccupation() + shift;
        node->definiteOccupation_ = (n >= 0) ? n : -1;
    }
    if (any && parityUniform && child->definiteParity() != 0)
        node->definiteParity_ = child->definiteParity() * opParity;
    if (!any) {  // zero operator: the zero vector, vacuously definite
        node->definiteOccupation_ = child->definiteOccupation();
        node->definiteParity_ = child->definiteParity();
    }
    node->nodeId_ = g_nodeCounter.fetch_add(1);
    HashChain h(0x1full);
    h.mix(modeCount_);
    h.mix(child->contentHash());
    h.mix(node->supportModes_.size());
    for (std::size_t m : node->supportModes_) h.mix(m);
    for (int k = 0; k < node->supportCols_.outerSize(); ++k)
        for (LazyFockNode::SparseColOp::InnerIterator it(node->supportCols_, k);
             it; ++it) {
            h.mix(static_cast<std::uint64_t>(it.row()));
            h.mix(static_cast<std::uint64_t>(it.col()));
            h.mixComplex(it.value());
        }
    node->contentHash_ = h.h;
    return node;
}

LazyFockEngine::NodePtr LazyFockEngine::makeWedge(
    std::vector<std::size_t> modes, Eigen::MatrixXcd orbitals) const {
    if (static_cast<std::size_t>(orbitals.rows()) != modes.size())
        throw std::invalid_argument(
            "LazyFockEngine::wedgeState: orbitals must have one row per "
            "mode");
    for (Eigen::Index c = 0; c < orbitals.cols(); ++c)
        for (Eigen::Index r = 0; r < orbitals.rows(); ++r)
            requireFinite(orbitals(r, c), "LazyFockEngine wedge orbital");
    auto node = std::shared_ptr<LazyFockNode>(new LazyFockNode());
    node->kind_ = LazyNodeKind::Wedge;
    node->universeModeCount_ = modeCount_;
    node->modes_ = std::move(modes);
    node->orbitals_ = std::move(orbitals);
    node->definiteOccupation_ = node->orbitals_.cols();
    node->definiteParity_ = (node->orbitals_.cols() % 2 == 0) ? +1 : -1;
    node->nodeId_ = g_nodeCounter.fetch_add(1);
    HashChain h(0x3bull);
    h.mix(modeCount_);
    h.mix(node->modes_.size());
    for (std::size_t m : node->modes_) h.mix(m);
    h.mix(static_cast<std::uint64_t>(node->orbitals_.cols()));
    for (Eigen::Index c = 0; c < node->orbitals_.cols(); ++c)
        for (Eigen::Index r = 0; r < node->orbitals_.rows(); ++r)
            h.mixComplex(node->orbitals_(r, c));
    node->contentHash_ = h.h;
    return node;
}

// ── state builders ───────────────────────────────────────────────────────

void LazyFockEngine::validateState(const LazyFockState& state) const {
    if (!state.valid())
        throw std::invalid_argument(
            "LazyFockEngine: invalid (default-constructed) state handle");
    if (state.root_->universeModeCount() != modeCount_)
        throw std::invalid_argument(
            "LazyFockEngine: state belongs to a different mode universe");
}

LazyFockState LazyFockEngine::wrap(NodePtr node, double discardedNorm,
                                   std::string label) const {
    return LazyFockState(std::move(node), discardedNorm, std::move(label));
}

LazyFockState LazyFockEngine::vacuum() const {
    std::vector<std::size_t> all(modeCount_);
    for (std::size_t i = 0; i < modeCount_; ++i) all[i] = i;
    return wrap(makeVacuum(std::move(all)), 0.0);
}

LazyFockState LazyFockEngine::vacuumOn(
    const std::vector<std::size_t>& modes) const {
    return wrap(makeVacuum(canonicalModes(modes, "LazyFockEngine::vacuumOn")),
                0.0);
}

LazyFockState LazyFockEngine::occupationState(
    const std::vector<std::size_t>& modes,
    const std::vector<std::vector<std::size_t>>& occupations,
    const std::vector<Complex>& amplitudes) const {
    if (occupations.size() != amplitudes.size())
        throw std::invalid_argument(
            "LazyFockEngine::occupationState: occupations/amplitudes length "
            "mismatch");
    auto canonical = canonicalModes(modes, "LazyFockEngine::occupationState");
    std::vector<Term> terms;
    terms.reserve(occupations.size());
    for (std::size_t t = 0; t < occupations.size(); ++t)
        terms.emplace_back(
            OccupationBitset::fromOccupiedModes(modeCount_, occupations[t]),
            amplitudes[t]);
    return wrap(makeOccupation(std::move(canonical), std::move(terms)), 0.0);
}

LazyFockState LazyFockEngine::wedgeState(
    const std::vector<std::size_t>& modes,
    const Eigen::MatrixXcd& orbitals) const {
    return wrap(makeWedge(canonicalModes(modes, "LazyFockEngine::wedgeState"),
                          orbitals),
                0.0);
}

LazySlaterReference LazyFockEngine::slaterFromProjector(
    const std::vector<std::size_t>& modes, const Eigen::MatrixXcd& projector,
    double tolerance) const {
    auto canonical =
        canonicalModes(modes, "LazyFockEngine::slaterFromProjector");
    const auto m = static_cast<Eigen::Index>(canonical.size());
    if (projector.rows() != m || projector.cols() != m)
        throw std::invalid_argument(
            "LazyFockEngine::slaterFromProjector: projector must be square "
            "over the given modes");
    const double scale = std::max(1.0, projector.norm());
    const double idem = (projector * projector - projector).norm() / scale;
    const double herm = (projector - projector.adjoint()).norm() / scale;
    const double residual = std::max(idem, herm);
    if (!(residual <= tolerance))
        throw std::invalid_argument(
            "LazyFockEngine::slaterFromProjector: premise P^2 = P = P^t̄ "
            "violated (residual " + std::to_string(residual) +
            " > tolerance) — refusing a silent non-projector reference");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(
        Complex(0.5, 0.0) * (projector + projector.adjoint()));
    const auto rank = static_cast<std::size_t>(
        std::llround(projector.real().trace()));
    // Occupied orbitals = the `rank` top eigenvectors (eigenvalues ≈ 1).
    Eigen::MatrixXcd occupied(m, static_cast<Eigen::Index>(rank));
    for (std::size_t k = 0; k < rank; ++k)
        occupied.col(static_cast<Eigen::Index>(k)) =
            es.eigenvectors().col(m - 1 - static_cast<Eigen::Index>(k));
    LazySlaterReference out;
    out.state = wrap(makeWedge(std::move(canonical), std::move(occupied)), 0.0);
    out.rank = rank;
    out.projectorResidual = residual;
    out.certificate = cobordism::Certificate::structureExact(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::PositiveSemidefinite, residual,
        /*conditioning=*/1.0, tolerance);
    return out;
}

LazyFockState LazyFockEngine::gradedTensor(const LazyFockState& a,
                                           const LazyFockState& b) const {
    validateState(a);
    validateState(b);
    NodePtr node = makeTensor(a.root_, b.root_);
    double d = 0.0;
    if (a.discardedNorm() > 0.0 || b.discardedNorm() > 0.0) {
        const double na = std::sqrt(std::max(0.0, normSquaredOf(a.root_)));
        const double nb = std::sqrt(std::max(0.0, normSquaredOf(b.root_)));
        d = a.discardedNorm() * (nb + b.discardedNorm()) +
            b.discardedNorm() * na;
    }
    return wrap(std::move(node), d);
}

LazyFockState LazyFockEngine::sectorSum(
    const std::vector<LazyFockState>& children, LazySectorKind kind) const {
    std::vector<NodePtr> nodes;
    double d = 0.0;
    for (const auto& c : children) {
        validateState(c);
        nodes.push_back(c.root_);
        d += c.discardedNorm();
    }
    return wrap(makeSectorSum(std::move(nodes), kind), d);
}

LazyFockState LazyFockEngine::boundaryProductFixture(
    const std::vector<std::size_t>& modes,
    const std::vector<Complex>& emptyAmplitudes,
    const std::vector<Complex>& occupiedAmplitudes,
    const std::string& label) const {
    if (label.empty())
        throw std::invalid_argument(
            "LazyFockEngine::boundaryProductFixture: a stored product "
            "preparation MUST carry a non-empty label");
    auto canonical =
        canonicalModes(modes, "LazyFockEngine::boundaryProductFixture");
    if (canonical.empty() || emptyAmplitudes.size() != canonical.size() ||
        occupiedAmplitudes.size() != canonical.size())
        throw std::invalid_argument(
            "LazyFockEngine::boundaryProductFixture: need one (empty, "
            "occupied) amplitude pair per mode");
    NodePtr chain{};
    for (std::size_t i = 0; i < canonical.size(); ++i) {
        std::vector<Term> terms;
        terms.emplace_back(OccupationBitset(modeCount_), emptyAmplitudes[i]);
        terms.emplace_back(OccupationBitset::fromOccupiedModes(
                               modeCount_, {canonical[i]}),
                           occupiedAmplitudes[i]);
        NodePtr factor = makeOccupation({canonical[i]}, std::move(terms));
        chain = chain ? makeTensor(chain, factor) : factor;
    }
    return wrap(std::move(chain), 0.0, label);
}

LazyFockState LazyFockEngine::embedInVacuum(
    const LazyFockState& state, const std::vector<std::size_t>& newModes) const {
    validateState(state);
    auto canonical = canonicalModes(newModes, "LazyFockEngine::embedInVacuum");
    for (std::size_t m : canonical)
        if (Bits::contains(state.root_->modes(), m))
            throw std::invalid_argument(
                "LazyFockEngine::embedInVacuum: new modes must be disjoint "
                "from the state's modes");
    return wrap(makeTensor(state.root_, makeVacuum(std::move(canonical))),
                state.discardedNorm());
}

// ── expansion / evaluation core ─────────────────────────────────────────

std::shared_ptr<const LazyFockEngine::Expansion> LazyFockEngine::expand(
    const NodePtr& node) const {
    auto found = memo_.find(node->contentHash());
    if (found != memo_.end()) {
        ++memoHits_;
        return found->second;
    }
    ++memoMisses_;
    auto out = std::make_shared<Expansion>();
    switch (node->kind()) {
        case LazyNodeKind::Vacuum:
            out->terms.emplace_back(OccupationBitset(modeCount_),
                                    Complex(1.0, 0.0));
            break;
        case LazyNodeKind::Occupation:
            out->terms = node->terms();
            break;
        case LazyNodeKind::GradedTensor: {
            auto left = expand(node->children()[0]);
            auto right = expand(node->children()[1]);
            const std::size_t total = left->terms.size() * right->terms.size();
            if (left->terms.size() != 0 && right->terms.size() != 0 &&
                (total / left->terms.size() != right->terms.size() ||
                 total > maxExpansionTerms_))
                throw std::length_error(
                    "LazyFockEngine: graded-tensor expansion exceeds "
                    "maxExpansionTerms");
            out->terms.reserve(total);
            for (const auto& [kl, al] : left->terms) {
                const auto occL = kl.occupiedModes();
                for (const auto& [kr, ar] : right->terms) {
                    const auto occR = kr.occupiedModes();
                    const int sign = Bits::koszulSign(occL, occR);
                    OccupationBitset joint = kl;
                    for (std::size_t m : occR) joint.set(m);
                    out->terms.emplace_back(joint,
                                            double(sign) * al * ar);
                }
            }
            std::sort(out->terms.begin(), out->terms.end(),
                      [](const Term& a, const Term& b) {
                          return Bits::keyLess(a.first, b.first);
                      });
            break;
        }
        case LazyNodeKind::SectorSum: {
            for (const auto& c : node->children()) {
                auto ex = expand(c);
                out->terms.insert(out->terms.end(), ex->terms.begin(),
                                  ex->terms.end());
            }
            std::sort(out->terms.begin(), out->terms.end(),
                      [](const Term& a, const Term& b) {
                          return Bits::keyLess(a.first, b.first);
                      });
            break;
        }
        case LazyNodeKind::Wedge: {
            const auto& modes = node->modes();
            const auto& V = node->orbitals();
            const std::size_t m = modes.size();
            const auto n = static_cast<std::size_t>(V.cols());
            if (n > m) break;  // exactly zero (Pauli)
            // Guard C(m, n) against the refusal threshold.
            {
                double count = 1.0;
                for (std::size_t k = 0; k < n; ++k)
                    count *= double(m - k) / double(k + 1);
                if (count > double(maxExpansionTerms_))
                    throw std::length_error(
                        "LazyFockEngine: wedge expansion exceeds "
                        "maxExpansionTerms");
            }
            std::vector<std::size_t> comb(n);
            for (std::size_t k = 0; k < n; ++k) comb[k] = k;
            Eigen::MatrixXcd sub(static_cast<Eigen::Index>(n),
                                 static_cast<Eigen::Index>(n));
            while (true) {
                for (std::size_t r = 0; r < n; ++r)
                    sub.row(static_cast<Eigen::Index>(r)) =
                        V.row(static_cast<Eigen::Index>(comb[r]));
                const Complex det =
                    (n == 0) ? Complex(1.0, 0.0) : sub.determinant();
                if (det != Complex(0.0, 0.0)) {
                    std::vector<std::size_t> occupied;
                    occupied.reserve(n);
                    for (std::size_t r : comb) occupied.push_back(modes[r]);
                    out->terms.emplace_back(OccupationBitset::fromOccupiedModes(
                                                modeCount_, occupied),
                                            det);
                }
                // next combination
                if (n == 0) break;
                std::size_t k = n;
                while (k-- > 0) {
                    if (comb[k] != k + m - n) {
                        ++comb[k];
                        for (std::size_t j = k + 1; j < n; ++j)
                            comb[j] = comb[j - 1] + 1;
                        break;
                    }
                    if (k == 0) {
                        k = std::size_t(-1);
                        break;
                    }
                }
                if (k == std::size_t(-1)) break;
            }
            std::sort(out->terms.begin(), out->terms.end(),
                      [](const Term& a, const Term& b) {
                          return Bits::keyLess(a.first, b.first);
                      });
            break;
        }
        case LazyNodeKind::LocalMap: {
            // Read-path materialization is exact w.r.t. the stored node:
            // no truncation here (drops == nullptr).
            OperatorSpec spec;
            spec.kind = OperatorSpec::Kind::SupportMatrix;
            spec.supportModes = node->supportModes();
            spec.colOp = node->supportMatrixCols();
            NodePtr applied = applyToExpansion(
                *expand(node->children()[0]), node->modes(), spec, nullptr);
            out->terms = applied->terms();
            break;
        }
    }
    if (out->terms.size() > maxExpansionTerms_)
        throw std::length_error(
            "LazyFockEngine: expansion exceeds maxExpansionTerms");
    memo_.emplace(node->contentHash(), out);
    return out;
}

LazyFockEngine::Complex LazyFockEngine::amplitudeOf(
    const LazyFockNode& node, const OccupationBitset& key) const {
    // Block sparsity: out-of-sector reads are exactly zero.
    const long long n = static_cast<long long>(key.count());
    if (node.definiteOccupation() >= 0 && node.definiteOccupation() != n)
        return {0.0, 0.0};
    if (node.definiteParity() != 0 && node.definiteParity() != key.parity())
        return {0.0, 0.0};
    switch (node.kind()) {
        case LazyNodeKind::Vacuum:
            return (n == 0) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
        case LazyNodeKind::Occupation: {
            const auto& terms = node.terms();
            auto it = std::lower_bound(
                terms.begin(), terms.end(), key,
                [](const Term& t, const OccupationBitset& k) {
                    return Bits::keyLess(t.first, k);
                });
            if (it != terms.end() && it->first == key) return it->second;
            return {0.0, 0.0};
        }
        case LazyNodeKind::GradedTensor: {
            const auto& left = *node.children()[0];
            const auto& right = *node.children()[1];
            std::vector<std::size_t> occL, occR;
            for (std::size_t m : key.occupiedModes()) {
                if (Bits::contains(left.modes(), m)) occL.push_back(m);
                else if (Bits::contains(right.modes(), m)) occR.push_back(m);
                else return {0.0, 0.0};  // occupied outside the partition
            }
            const Complex al = amplitudeOf(
                left, OccupationBitset::fromOccupiedModes(modeCount_, occL));
            if (al == Complex(0.0, 0.0)) return {0.0, 0.0};
            const Complex ar = amplitudeOf(
                right, OccupationBitset::fromOccupiedModes(modeCount_, occR));
            if (ar == Complex(0.0, 0.0)) return {0.0, 0.0};
            return double(Bits::koszulSign(occL, occR)) * al * ar;
        }
        case LazyNodeKind::SectorSum: {
            const long long label =
                (node.sectorKind() == LazySectorKind::Occupation)
                    ? n
                    : static_cast<long long>(key.parity());
            const auto& labels = node.sectorLabels();
            auto it = std::lower_bound(labels.begin(), labels.end(), label);
            if (it == labels.end() || *it != label) return {0.0, 0.0};
            const auto idx =
                static_cast<std::size_t>(std::distance(labels.begin(), it));
            return amplitudeOf(*node.children()[idx], key);
        }
        case LazyNodeKind::LocalMap: {
            const auto& support = node.supportModes();
            for (std::size_t m : key.occupiedModes())
                if (!Bits::contains(node.modes(), m)) return {0.0, 0.0};
            const std::size_t row = Bits::supportIndex(key, support);
            OccupationBitset rest = key;
            for (std::size_t m : support) rest.reset(m);
            const auto occRest = rest.occupiedModes();
            const int signOut = Bits::koszulSign(
                Bits::supportOccupied(row, support), occRest);
            Complex sum{0.0, 0.0};
            const auto& rows = node.supportMatrixRows();
            for (LazyFockNode::SparseRowOp::InnerIterator it(
                     rows, static_cast<Eigen::Index>(row));
                 it; ++it) {
                const auto col = static_cast<std::size_t>(it.col());
                const auto occIn = Bits::supportOccupied(col, support);
                const int signIn = Bits::koszulSign(occIn, occRest);
                OccupationBitset childKey = rest;
                for (std::size_t m : occIn) childKey.set(m);
                const Complex childAmp =
                    amplitudeOf(*node.children()[0], childKey);
                if (childAmp == Complex(0.0, 0.0)) continue;
                sum += it.value() * double(signOut * signIn) * childAmp;
            }
            return sum;
        }
        case LazyNodeKind::Wedge: {
            const auto& modes = node.modes();
            const auto& V = node.orbitals();
            const auto k = static_cast<std::size_t>(V.cols());
            if (static_cast<std::size_t>(n) != k) return {0.0, 0.0};
            Eigen::MatrixXcd sub(static_cast<Eigen::Index>(k),
                                 static_cast<Eigen::Index>(k));
            std::size_t r = 0;
            for (std::size_t m : key.occupiedModes()) {
                auto pos = std::lower_bound(modes.begin(), modes.end(), m);
                if (pos == modes.end() || *pos != m) return {0.0, 0.0};
                sub.row(static_cast<Eigen::Index>(r++)) = V.row(
                    static_cast<Eigen::Index>(pos - modes.begin()));
            }
            return (k == 0) ? Complex(1.0, 0.0) : sub.determinant();
        }
    }
    return {0.0, 0.0};
}

double LazyFockEngine::normSquaredOf(const NodePtr& node) const {
    switch (node->kind()) {
        case LazyNodeKind::Vacuum:
            return 1.0;
        case LazyNodeKind::Occupation: {
            double s = 0.0;
            for (const auto& [k, a] : node->terms()) s += std::norm(a);
            return s;
        }
        case LazyNodeKind::GradedTensor:
            return normSquaredOf(node->children()[0]) *
                   normSquaredOf(node->children()[1]);
        case LazyNodeKind::SectorSum: {
            double s = 0.0;
            for (const auto& c : node->children()) s += normSquaredOf(c);
            return s;
        }
        case LazyNodeKind::Wedge: {
            const auto& V = node->orbitals();
            if (V.cols() == 0) return 1.0;
            const Eigen::MatrixXcd gram = V.adjoint() * V;
            return gram.determinant().real();
        }
        case LazyNodeKind::LocalMap: {
            double s = 0.0;
            for (const auto& [k, a] : expand(node)->terms) s += std::norm(a);
            return s;
        }
    }
    return 0.0;
}

// ── operator application ────────────────────────────────────────────────

LazyFockEngine::NodePtr LazyFockEngine::applyToExpansion(
    const Expansion& expansion, const std::vector<std::size_t>& modes,
    const OperatorSpec& op, double* drops) const {
    TermMap acc;
    if (op.kind == OperatorSpec::Kind::DGamma) {
        for (const auto& [key, amp] : expansion.terms) {
            for (const auto& [gi, gj, val] : op.hopping) {
                OccupationBitset c = key;
                const int s1 = c.applyAnnihilation(gj);
                if (s1 == 0) continue;
                const int s2 = c.applyCreation(gi);
                if (s2 == 0) continue;
                acc[c] += val * double(s1 * s2) * amp;
            }
        }
    } else {
        const auto& support = op.supportModes;
        for (const auto& [key, amp] : expansion.terms) {
            const std::size_t col = Bits::supportIndex(key, support);
            OccupationBitset rest = key;
            for (std::size_t m : support) rest.reset(m);
            const auto occRest = rest.occupiedModes();
            const int signIn = Bits::koszulSign(
                Bits::supportOccupied(col, support), occRest);
            for (LazyFockNode::SparseColOp::InnerIterator it(
                     op.colOp, static_cast<Eigen::Index>(col));
                 it; ++it) {
                const auto row = static_cast<std::size_t>(it.row());
                const auto occOut = Bits::supportOccupied(row, support);
                const int signOut = Bits::koszulSign(occOut, occRest);
                OccupationBitset out = rest;
                for (std::size_t m : occOut) out.set(m);
                acc[out] += it.value() * double(signIn * signOut) * amp;
            }
        }
    }
    std::vector<Term> terms;
    terms.reserve(acc.size());
    double dropped2 = 0.0;
    const bool truncate = (drops != nullptr) && !exactMode();
    for (auto& [key, amp] : acc) {
        if (truncate && std::abs(amp) <= truncationThreshold_ &&
            amp != Complex(0.0, 0.0)) {
            dropped2 += std::norm(amp);
            continue;
        }
        terms.emplace_back(key, amp);
    }
    if (truncate && dropped2 > 0.0) *drops += std::sqrt(dropped2);
    return makeOccupation(modes, std::move(terms));
}

LazyFockEngine::NodePtr LazyFockEngine::applyOperator(
    const NodePtr& node, const OperatorSpec& op, double* drops) const {
    switch (node->kind()) {
        case LazyNodeKind::Vacuum:
        case LazyNodeKind::Occupation:
            return applyToExpansion(*expand(node), node->modes(), op, drops);
        case LazyNodeKind::Wedge:
            if (op.kind == OperatorSpec::Kind::SupportMatrix)
                return makeLocalMap(node, op.supportModes, op.colOp);
            return applyToExpansion(*expand(node), node->modes(), op, drops);
        case LazyNodeKind::LocalMap:
            if (op.kind == OperatorSpec::Kind::SupportMatrix)
                return makeLocalMap(node, op.supportModes, op.colOp);
            return applyToExpansion(*expand(node), node->modes(), op, drops);
        case LazyNodeKind::GradedTensor: {
            const NodePtr& left = node->children()[0];
            const NodePtr& right = node->children()[1];
            if (Bits::isSubset(op.supportModes, left->modes())) {
                // Any-parity operators act as O ⊗̂ 1 on the left factor —
                // the sibling is shared, never expanded.
                return makeTensor(applyOperator(left, op, drops), right);
            }
            if (Bits::isSubset(op.supportModes, right->modes())) {
                if (op.parity == +1)
                    return makeTensor(left, applyOperator(right, op, drops));
                if (op.parity == -1 && left->definiteParity() != 0) {
                    // Koszul twist for an odd right-factor operator:
                    // O(x ⊗̂ y) = ((−1)^{N} x) ⊗̂ (O y) — a scalar
                    // (−1)^{parity(x)} folded into the operator.
                    const OperatorSpec effective =
                        (left->definiteParity() == -1) ? op.negated() : op;
                    return makeTensor(left,
                                      applyOperator(right, effective, drops));
                }
                // Mixed-parity operator or indefinite-parity sibling:
                // exact fallback through expansion (counted as a
                // crossing).
            }
            ++expansionCount_;
            return applyToExpansion(*expand(node), node->modes(), op, drops);
        }
        case LazyNodeKind::SectorSum: {
            std::vector<NodePtr> applied;
            applied.reserve(node->children().size());
            for (const auto& c : node->children())
                applied.push_back(applyOperator(c, op, drops));
            // Keep the sector structure when the functional is still
            // definite and pairwise distinct; otherwise merge losslessly.
            bool keep = true;
            std::vector<long long> labels;
            for (const auto& c : applied) {
                const long long label =
                    (node->sectorKind() == LazySectorKind::Occupation)
                        ? c->definiteOccupation()
                        : static_cast<long long>(c->definiteParity());
                const bool definite =
                    (node->sectorKind() == LazySectorKind::Occupation)
                        ? (label >= 0)
                        : (label != 0);
                if (!definite) keep = false;
                labels.push_back(label);
            }
            if (keep) {
                std::vector<long long> sorted = labels;
                std::sort(sorted.begin(), sorted.end());
                if (std::adjacent_find(sorted.begin(), sorted.end()) !=
                    sorted.end())
                    keep = false;
            }
            if (keep) return makeSectorSum(std::move(applied),
                                           node->sectorKind());
            std::vector<Term> terms;
            for (const auto& c : applied) {
                auto ex = expand(c);
                terms.insert(terms.end(), ex->terms.begin(), ex->terms.end());
            }
            return makeOccupation(node->modes(), std::move(terms));
        }
    }
    throw std::logic_error("LazyFockEngine::applyOperator: unreachable");
}

namespace {
void validateSupportInState(const std::vector<std::size_t>& support,
                            const std::vector<std::size_t>& stateModes) {
    if (!Bits::isSubset(support, stateModes))
        throw std::invalid_argument(
            "LazyFockEngine: operator support is outside the state's mode "
            "coverage — embedInVacuum first to extend the carrier stage");
}
}  // namespace

LazyFockState LazyFockEngine::applyLocalMapDense(
    const LazyFockState& state, const std::vector<std::size_t>& supportModes,
    const Eigen::MatrixXcd& op) const {
    validateState(state);
    auto support =
        canonicalModes(supportModes, "LazyFockEngine::applyLocalMapDense");
    if (support.size() > kMaxSupportModes)
        throw std::invalid_argument(
            "LazyFockEngine::applyLocalMapDense: support exceeds "
            "kMaxSupportModes (use applyDGamma for large quadratic "
            "generators)");
    const auto dim = static_cast<Eigen::Index>(std::size_t{1}
                                               << support.size());
    if (op.rows() != dim || op.cols() != dim)
        throw std::invalid_argument(
            "LazyFockEngine::applyLocalMapDense: operator must be "
            "2^{|support|} square");
    validateSupportInState(support, state.root_->modes());
    OperatorSpec spec;
    spec.kind = OperatorSpec::Kind::SupportMatrix;
    spec.supportModes = support;
    std::vector<Eigen::Triplet<Complex>> triplets;
    bool any = false;
    long long shift = 0;
    bool shiftUniform = true;
    int parity = +1;
    bool parityUniform = true;
    for (Eigen::Index c = 0; c < dim; ++c)
        for (Eigen::Index r = 0; r < dim; ++r) {
            const Complex v = op(r, c);
            if (v == Complex(0.0, 0.0)) continue;
            requireFinite(v, "LazyFockEngine::applyLocalMapDense entry");
            triplets.emplace_back(r, c, v);
            const long long d =
                std::popcount(static_cast<std::uint64_t>(r)) -
                std::popcount(static_cast<std::uint64_t>(c));
            const int p = (d % 2 == 0) ? +1 : -1;
            if (!any) {
                shift = d;
                parity = p;
                any = true;
            } else {
                if (d != shift) shiftUniform = false;
                if (p != parity) parityUniform = false;
            }
        }
    (void)shiftUniform;
    spec.colOp.resize(dim, dim);
    spec.colOp.setFromTriplets(triplets.begin(), triplets.end());
    spec.colOp.makeCompressed();
    spec.parity = !any ? +1 : (parityUniform ? parity : 0);
    spec.normBound = operatorNormBound(spec.colOp);
    double drops = 0.0;
    NodePtr result = applyOperator(state.root_, spec, &drops);
    return wrap(std::move(result),
                state.discardedNorm() * spec.normBound + drops);
}

LazyFockState LazyFockEngine::applyLocalMapCOO(
    const LazyFockState& state, const std::vector<std::size_t>& supportModes,
    const std::vector<std::int64_t>& rows,
    const std::vector<std::int64_t>& cols,
    const std::vector<Complex>& values) const {
    if (rows.size() != cols.size() || rows.size() != values.size())
        throw std::invalid_argument(
            "LazyFockEngine::applyLocalMapCOO: rows/cols/values length "
            "mismatch");
    auto support =
        canonicalModes(supportModes, "LazyFockEngine::applyLocalMapCOO");
    if (support.size() > kMaxSupportModes)
        throw std::invalid_argument(
            "LazyFockEngine::applyLocalMapCOO: support exceeds "
            "kMaxSupportModes");
    const auto dim = static_cast<std::int64_t>(std::size_t{1}
                                               << support.size());
    Eigen::MatrixXcd dense = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
    for (std::size_t k = 0; k < rows.size(); ++k) {
        if (rows[k] < 0 || rows[k] >= dim || cols[k] < 0 || cols[k] >= dim)
            throw std::invalid_argument(
                "LazyFockEngine::applyLocalMapCOO: triplet out of range");
        dense(static_cast<Eigen::Index>(rows[k]),
              static_cast<Eigen::Index>(cols[k])) += values[k];
    }
    return applyLocalMapDense(state, support, dense);
}

LazyFockState LazyFockEngine::applyCreation(const LazyFockState& state,
                                            std::size_t mode) const {
    Eigen::MatrixXcd op = Eigen::MatrixXcd::Zero(2, 2);
    op(1, 0) = Complex(1.0, 0.0);  // a†|0> = |1>
    return applyLocalMapDense(state, {mode}, op);
}

LazyFockState LazyFockEngine::applyAnnihilation(const LazyFockState& state,
                                                std::size_t mode) const {
    Eigen::MatrixXcd op = Eigen::MatrixXcd::Zero(2, 2);
    op(0, 1) = Complex(1.0, 0.0);  // a|1> = |0>
    return applyLocalMapDense(state, {mode}, op);
}

LazyFockState LazyFockEngine::applyDGamma(
    const LazyFockState& state, const std::vector<std::size_t>& supportModes,
    const Eigen::MatrixXcd& oneParticle) const {
    validateState(state);
    auto support = canonicalModes(supportModes, "LazyFockEngine::applyDGamma");
    const auto s = static_cast<Eigen::Index>(support.size());
    if (oneParticle.rows() != s || oneParticle.cols() != s)
        throw std::invalid_argument(
            "LazyFockEngine::applyDGamma: one-particle block must be "
            "|support| square");
    validateSupportInState(support, state.root_->modes());
    OperatorSpec spec;
    spec.kind = OperatorSpec::Kind::DGamma;
    // Effective support = modes touched by a nonzero entry of L (drives
    // the branch-descent laziness).
    std::vector<bool> touched(support.size(), false);
    for (Eigen::Index i = 0; i < s; ++i)
        for (Eigen::Index j = 0; j < s; ++j) {
            const Complex v = oneParticle(i, j);
            if (v == Complex(0.0, 0.0)) continue;
            requireFinite(v, "LazyFockEngine::applyDGamma entry");
            spec.hopping.emplace_back(support[static_cast<std::size_t>(i)],
                                      support[static_cast<std::size_t>(j)], v);
            touched[static_cast<std::size_t>(i)] = true;
            touched[static_cast<std::size_t>(j)] = true;
        }
    for (std::size_t k = 0; k < support.size(); ++k)
        if (touched[k]) spec.supportModes.push_back(support[k]);
    spec.parity = +1;  // dΓ is even (number conserving)
    spec.normBound =
        double(support.size()) * matrixSpectralBound(oneParticle);
    double drops = 0.0;
    NodePtr result = applyOperator(state.root_, spec, &drops);
    return wrap(std::move(result),
                state.discardedNorm() * spec.normBound + drops);
}

LazyFockState LazyFockEngine::materialize(const LazyFockState& state) const {
    validateState(state);
    auto ex = expand(state.root_);
    std::vector<Term> terms;
    terms.reserve(ex->terms.size());
    double dropped2 = 0.0;
    for (const auto& [key, amp] : ex->terms) {
        if (!exactMode() && std::abs(amp) <= truncationThreshold_) {
            dropped2 += std::norm(amp);
            continue;
        }
        terms.push_back({key, amp});
    }
    return wrap(makeOccupation(state.root_->modes(), std::move(terms)),
                state.discardedNorm() + std::sqrt(dropped2),
                state.boundaryFixtureLabel());
}

LazyFockState LazyFockEngine::permuteModes(
    const LazyFockState& state, const std::vector<std::size_t>& perm) const {
    validateState(state);
    if (perm.size() != modeCount_)
        throw std::invalid_argument(
            "LazyFockEngine::permuteModes: permutation must cover the mode "
            "universe");
    // Recursive signed relabeling (see header: children permute
    // independently; the tensor ε rule recomputes interleaving signs in
    // the new labels; permutationParity carries the within-block signs).
    struct Rec {
        const LazyFockEngine& eng;
        const std::vector<std::size_t>& perm;
        NodePtr run(const NodePtr& node) {
            std::vector<std::size_t> newModes;
            newModes.reserve(node->modes().size());
            for (std::size_t m : node->modes()) newModes.push_back(perm[m]);
            std::sort(newModes.begin(), newModes.end());
            switch (node->kind()) {
                case LazyNodeKind::Vacuum:
                    return eng.makeVacuum(std::move(newModes));
                case LazyNodeKind::Occupation: {
                    std::vector<Term> terms;
                    terms.reserve(node->terms().size());
                    for (const auto& [key, amp] : node->terms()) {
                        const int sign = key.permutationParity(perm);
                        terms.emplace_back(key.permuted(perm),
                                           double(sign) * amp);
                    }
                    return eng.makeOccupation(std::move(newModes),
                                              std::move(terms));
                }
                case LazyNodeKind::GradedTensor:
                    return eng.makeTensor(run(node->children()[0]),
                                          run(node->children()[1]));
                case LazyNodeKind::SectorSum: {
                    std::vector<NodePtr> children;
                    for (const auto& c : node->children())
                        children.push_back(run(c));
                    return eng.makeSectorSum(std::move(children),
                                             node->sectorKind());
                }
                case LazyNodeKind::Wedge: {
                    // Orbitals transform linearly (no sign): reorder rows
                    // to the sorted new-mode order.
                    const auto& oldModes = node->modes();
                    Eigen::MatrixXcd reordered(node->orbitals().rows(),
                                               node->orbitals().cols());
                    for (std::size_t r = 0; r < oldModes.size(); ++r) {
                        const std::size_t target = perm[oldModes[r]];
                        const auto pos = std::lower_bound(newModes.begin(),
                                                          newModes.end(),
                                                          target) -
                                         newModes.begin();
                        reordered.row(static_cast<Eigen::Index>(pos)) =
                            node->orbitals().row(static_cast<Eigen::Index>(r));
                    }
                    return eng.makeWedge(std::move(newModes),
                                         std::move(reordered));
                }
                case LazyNodeKind::LocalMap: {
                    NodePtr child = run(node->children()[0]);
                    const auto& oldSupport = node->supportModes();
                    std::vector<std::size_t> newSupport;
                    for (std::size_t m : oldSupport)
                        newSupport.push_back(perm[m]);
                    std::sort(newSupport.begin(), newSupport.end());
                    // Support-local induced permutation and its exact
                    // signed unitary (ExteriorAlgebra).
                    std::vector<std::size_t> sigma(oldSupport.size());
                    for (std::size_t k = 0; k < oldSupport.size(); ++k)
                        sigma[k] = static_cast<std::size_t>(
                            std::lower_bound(newSupport.begin(),
                                             newSupport.end(),
                                             perm[oldSupport[k]]) -
                            newSupport.begin());
                    ExteriorAlgebra alg(oldSupport.size());
                    const LazyFockNode::SparseColOp u =
                        alg.modePermutationMatrix(sigma);
                    const LazyFockNode::SparseColOp uAdj = u.adjoint();
                    LazyFockNode::SparseColOp prod =
                        (u * node->supportMatrixCols()).eval();
                    prod = (prod * uAdj).eval();
                    // Rebuild without explicit zeros (canonical content).
                    std::vector<Eigen::Triplet<Complex>> tr;
                    for (int k = 0; k < prod.outerSize(); ++k)
                        for (LazyFockNode::SparseColOp::InnerIterator it(
                                 prod, k);
                             it; ++it)
                            if (it.value() != Complex(0.0, 0.0))
                                tr.emplace_back(it.row(), it.col(),
                                                it.value());
                    LazyFockNode::SparseColOp conjugated(prod.rows(),
                                                         prod.cols());
                    conjugated.setFromTriplets(tr.begin(), tr.end());
                    return eng.makeLocalMap(child, std::move(newSupport),
                                            std::move(conjugated));
                }
            }
            throw std::logic_error("permuteModes: unreachable");
        }
    };
    Rec rec{*this, perm};
    return wrap(rec.run(state.root_), state.discardedNorm());
}

// ── reads ────────────────────────────────────────────────────────────────

cobordism::Certificate LazyFockEngine::readCertificate(
    double discardedNorm) const {
    if (discardedNorm == 0.0)
        return cobordism::Certificate::algebraicallyExact(
            cobordism::CertificateDomain::Static,
            cobordism::CertificateRegime::PositiveSemidefinite,
            /*residual=*/0.0, /*tolerance=*/0.0);
    // Absolute accumulated discarded-norm bound as the residual (a
    // deviation from the relative default).
    return cobordism::Certificate::certifiedNumerical(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::PositiveSemidefinite, discardedNorm,
        cobordism::Certificate::kUnmeasured, truncationNormTolerance_);
}

LazyScalarRead LazyFockEngine::amplitude(
    const LazyFockState& state,
    const std::vector<std::size_t>& occupiedModes) const {
    validateState(state);
    auto occupied = canonicalModes(occupiedModes, "LazyFockEngine::amplitude");
    const auto key = OccupationBitset::fromOccupiedModes(modeCount_, occupied);
    // Occupied modes outside the state's coverage: vacuum there → 0.
    for (std::size_t m : occupied)
        if (!Bits::contains(state.root_->modes(), m))
            return LazyScalarRead{Complex(0.0, 0.0), state.discardedNorm(),
                                  readCertificate(state.discardedNorm())};
    LazyScalarRead out;
    out.value = amplitudeOf(*state.root_, key);
    out.discardedNorm = state.discardedNorm();
    out.certificate = readCertificate(out.discardedNorm);
    return out;
}

LazyScalarRead LazyFockEngine::innerProduct(const LazyFockState& a,
                                            const LazyFockState& b) const {
    validateState(a);
    validateState(b);
    Complex value{0.0, 0.0};
    const auto* ra = a.root_.get();
    const auto* rb = b.root_.get();
    if (ra->kind() == LazyNodeKind::Wedge && rb->kind() == LazyNodeKind::Wedge &&
        ra->orbitals().cols() == rb->orbitals().cols()) {
        // Exact Gram identity ⟨∧v_i, ∧w_j⟩ = det[⟨v_i, w_j⟩] over the
        // union mode set (missing rows are zero).
        std::vector<std::size_t> unionModes(ra->modes());
        unionModes.insert(unionModes.end(), rb->modes().begin(),
                          rb->modes().end());
        std::sort(unionModes.begin(), unionModes.end());
        unionModes.erase(std::unique(unionModes.begin(), unionModes.end()),
                         unionModes.end());
        const auto n = ra->orbitals().cols();
        auto lift = [&](const LazyFockNode* w) {
            Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(
                static_cast<Eigen::Index>(unionModes.size()), n);
            for (std::size_t r = 0; r < w->modes().size(); ++r) {
                const auto pos = std::lower_bound(unionModes.begin(),
                                                  unionModes.end(),
                                                  w->modes()[r]) -
                                 unionModes.begin();
                out.row(static_cast<Eigen::Index>(pos)) =
                    w->orbitals().row(static_cast<Eigen::Index>(r));
            }
            return out;
        };
        const Eigen::MatrixXcd overlap =
            lift(ra).adjoint() * lift(rb);
        value = (n == 0) ? Complex(1.0, 0.0) : overlap.determinant();
    } else if (ra->kind() == LazyNodeKind::Occupation ||
               ra->kind() == LazyNodeKind::Vacuum) {
        // Drive by the sparse left side without expanding the right.
        for (const auto& [key, amp] : expand(a.root_)->terms)
            value += std::conj(amp) * amplitudeOf(*rb, key);
    } else if (rb->kind() == LazyNodeKind::Occupation ||
               rb->kind() == LazyNodeKind::Vacuum) {
        for (const auto& [key, amp] : expand(b.root_)->terms)
            value += std::conj(amplitudeOf(*ra, key)) * amp;
    } else {
        auto ea = expand(a.root_);
        auto eb = expand(b.root_);
        std::size_t i = 0, j = 0;
        while (i < ea->terms.size() && j < eb->terms.size()) {
            if (ea->terms[i].first == eb->terms[j].first) {
                value += std::conj(ea->terms[i].second) * eb->terms[j].second;
                ++i;
                ++j;
            } else if (Bits::keyLess(ea->terms[i].first, eb->terms[j].first)) {
                ++i;
            } else {
                ++j;
            }
        }
    }
    LazyScalarRead out;
    out.value = value;
    double bound = 0.0;
    if (a.discardedNorm() > 0.0 || b.discardedNorm() > 0.0) {
        const double na = std::sqrt(std::max(0.0, normSquaredOf(a.root_)));
        const double nb = std::sqrt(std::max(0.0, normSquaredOf(b.root_)));
        bound = a.discardedNorm() * (nb + b.discardedNorm()) +
                b.discardedNorm() * na;
    }
    out.discardedNorm = bound;
    out.certificate = readCertificate(bound);
    return out;
}

LazyScalarRead LazyFockEngine::normSquared(const LazyFockState& state) const {
    validateState(state);
    LazyScalarRead out;
    out.value = Complex(normSquaredOf(state.root_), 0.0);
    const double d = state.discardedNorm();
    const double bound =
        (d > 0.0) ? d * (2.0 * std::sqrt(std::abs(out.value.real())) + d)
                  : 0.0;
    out.discardedNorm = bound;
    out.certificate = readCertificate(bound);
    return out;
}

LazyCovarianceRead LazyFockEngine::covarianceMatrix(
    const LazyFockState& state) const {
    validateState(state);
    LazyCovarianceRead out;
    out.matrix = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(modeCount_),
                                        static_cast<Eigen::Index>(modeCount_));
    out.discardedNorm = state.discardedNorm();
    const auto* root = state.root_.get();
    if (root->kind() == LazyNodeKind::Wedge) {
        // Exact closed form Γ = V (V†V)⁻¹ V† (the span projector):
        // slaterFromProjector reads back Γ = P exactly.
        const auto& V = root->orbitals();
        const auto n = V.cols();
        if (n > 0) {
            const Eigen::MatrixXcd gram = V.adjoint() * V;
            const double gramDet = std::abs(gram.determinant());
            if (!(gramDet > 0.0))
                throw std::invalid_argument(
                    "LazyFockEngine::covarianceMatrix: dependent wedge "
                    "orbitals give the zero state — no normalized "
                    "covariance exists");
            const Eigen::MatrixXcd local =
                V * gram.ldlt().solve(V.adjoint());
            const auto& modes = root->modes();
            for (std::size_t r = 0; r < modes.size(); ++r)
                for (std::size_t c = 0; c < modes.size(); ++c)
                    out.matrix(static_cast<Eigen::Index>(modes[r]),
                               static_cast<Eigen::Index>(modes[c])) =
                        local(static_cast<Eigen::Index>(r),
                              static_cast<Eigen::Index>(c));
        }
        const double traceDefect =
            std::abs(out.matrix.trace() - Complex(double(n), 0.0)) /
            std::max(1.0, double(n));
        out.certificate = cobordism::Certificate::algebraicallyExact(
            cobordism::CertificateDomain::Static,
            cobordism::CertificateRegime::PositiveSemidefinite, traceDefect,
            1e-12);
        return out;
    }
    // General path: through the sparse expansion.
    auto ex = expand(state.root_);
    TermMap lookup;
    double norm2 = 0.0;
    for (const auto& [key, amp] : ex->terms) {
        lookup[key] = amp;
        norm2 += std::norm(amp);
    }
    if (!(norm2 > 0.0))
        throw std::invalid_argument(
            "LazyFockEngine::covarianceMatrix: zero state has no normalized "
            "covariance");
    for (const auto& [key, amp] : ex->terms) {
        for (std::size_t e : key.occupiedModes()) {
            for (std::size_t f : state.root_->modes()) {
                OccupationBitset c = key;
                const int s1 = c.applyAnnihilation(e);
                const int s2 = c.applyCreation(f);
                if (s2 == 0) continue;
                auto it = lookup.find(c);
                if (it == lookup.end()) continue;
                // Γ_ef = ⟨ψ| a_f† a_e |ψ⟩
                out.matrix(static_cast<Eigen::Index>(e),
                           static_cast<Eigen::Index>(f)) +=
                    std::conj(it->second) * amp * double(s1 * s2);
            }
        }
    }
    out.matrix /= norm2;
    const double hermDefect =
        (out.matrix - out.matrix.adjoint()).norm() /
        std::max(1.0, out.matrix.norm());
    out.certificate = cobordism::Certificate::algebraicallyExact(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::PositiveSemidefinite, hermDefect, 1e-12);
    return out;
}

Eigen::VectorXcd LazyFockEngine::denseVector(const LazyFockState& state) const {
    validateState(state);
    if (modeCount_ > kMaxDenseModes)
        throw std::invalid_argument(
            "LazyFockEngine::denseVector: a literal 2^M allocation beyond "
            "kMaxDenseModes is out of scope — read amplitudes lazily "
            "instead");
    Eigen::VectorXcd out = Eigen::VectorXcd::Zero(
        static_cast<Eigen::Index>(std::size_t{1} << modeCount_));
    for (const auto& [key, amp] : expand(state.root_)->terms)
        out(static_cast<Eigen::Index>(key.toIndex())) = amp;
    return out;
}

cobordism::CertifiedVector LazyFockEngine::freeSpectrum(
    const Eigen::MatrixXcd& oneParticle, int particles) const {
    if (oneParticle.rows() != oneParticle.cols())
        throw std::invalid_argument(
            "LazyFockEngine::freeSpectrum: one-particle operator must be "
            "square");
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(oneParticle);
    if (es.info() != Eigen::Success)
        throw std::runtime_error(
            "LazyFockEngine::freeSpectrum: eigensolver failed");
    std::vector<Complex> spectrum(es.eigenvalues().data(),
                                  es.eigenvalues().data() +
                                      es.eigenvalues().size());
    double residual = 0.0;
    const double scale = std::max(1.0, oneParticle.norm());
    for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k)
        residual = std::max(
            residual, (oneParticle * es.eigenvectors().col(k) -
                       es.eigenvalues()(k) * es.eigenvectors().col(k))
                              .norm() /
                          scale);
    double conditioning = cobordism::Certificate::kUnmeasured;
    if (es.eigenvectors().size() > 0) {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(es.eigenvectors());
        const auto& sv = svd.singularValues();
        if (sv.size() > 0 && sv(sv.size() - 1) > 0.0)
            conditioning = sv(0) / sv(sv.size() - 1);
    }
    cobordism::CertifiedVector out;
    out.values = cobordism::OccupationSpectra::subsetSums(
        spectrum, particles, maxExpansionTerms_);
    out.certificate = cobordism::Certificate::certifiedNumerical(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::NonNormal, residual, conditioning,
        1e-10);
    return out;
}

cobordism::CertifiedVector LazyFockEngine::freeSpectrumFromEigenvalues(
    const std::vector<Complex>& oneParticleSpectrum, int particles) const {
    cobordism::CertifiedVector out;
    out.values = cobordism::OccupationSpectra::subsetSums(
        oneParticleSpectrum, particles, maxExpansionTerms_);
    // Independent-path residual: the direct-sum identity evaluated through
    // OccupationSpectra::directSumSubsetSums on a half split.
    double residual = 0.0;
    if (oneParticleSpectrum.size() >= 2) {
        const std::size_t half = oneParticleSpectrum.size() / 2;
        std::vector<Complex> a(oneParticleSpectrum.begin(),
                               oneParticleSpectrum.begin() +
                                   static_cast<std::ptrdiff_t>(half));
        std::vector<Complex> b(oneParticleSpectrum.begin() +
                                   static_cast<std::ptrdiff_t>(half),
                               oneParticleSpectrum.end());
        const auto viaSplit = cobordism::OccupationSpectra::directSumSubsetSums(
            a, b, particles, maxExpansionTerms_);
        double scale = 1.0;
        for (const Complex& z : out.values)
            scale = std::max(scale, std::abs(z));
        for (std::size_t k = 0; k < out.values.size(); ++k)
            residual = std::max(residual,
                                std::abs(out.values[k] - viaSplit[k]) / scale);
    }
    out.certificate = cobordism::Certificate::algebraicallyExact(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::NonNormal, residual, 1e-12);
    return out;
}

LazyCompatibilityRead LazyFockEngine::inductiveCompatibility(
    const std::vector<std::size_t>& stageModes,
    const std::vector<std::size_t>& extendedModes,
    const std::vector<std::size_t>& stageSupport,
    const Eigen::MatrixXcd& stageOp,
    const std::vector<std::size_t>& extendedSupport,
    const Eigen::MatrixXcd& extendedOp,
    const std::vector<std::vector<std::size_t>>& activeBasis) const {
    auto stage = canonicalModes(stageModes, "inductiveCompatibility stage");
    auto extended =
        canonicalModes(extendedModes, "inductiveCompatibility extended");
    if (!Bits::isSubset(stage, extended))
        throw std::invalid_argument(
            "LazyFockEngine::inductiveCompatibility: stage modes must be a "
            "subset of the extended modes");
    // Distinct active basis states over the stage.
    std::vector<OccupationBitset> basis;
    for (const auto& occ : activeBasis) {
        auto sorted = canonicalModes(occ, "inductiveCompatibility basis");
        if (!Bits::isSubset(sorted, stage))
            throw std::invalid_argument(
                "LazyFockEngine::inductiveCompatibility: active basis state "
                "outside the stage modes");
        auto key = OccupationBitset::fromOccupiedModes(modeCount_, sorted);
        bool dup = false;
        for (const auto& k : basis)
            if (k == key) dup = true;
        if (!dup) basis.push_back(std::move(key));
    }
    if (basis.empty())
        throw std::invalid_argument(
            "LazyFockEngine::inductiveCompatibility: empty active subspace");
    // Column defects ι U_M |b⟩ − U_{M+1} ι |b⟩ (ι is the identity on
    // global occupation keys — the vacuum embedding adds empty modes).
    std::vector<TermMap> columns;
    for (const auto& key : basis) {
        std::vector<std::vector<std::size_t>> occ{key.occupiedModes()};
        const std::vector<Complex> one{Complex(1.0, 0.0)};
        LazyFockState sM = occupationState(stage, occ, one);
        LazyFockState sM1 = occupationState(extended, occ, one);
        LazyFockState uM = applyLocalMapDense(sM, stageSupport, stageOp);
        LazyFockState uM1 =
            applyLocalMapDense(sM1, extendedSupport, extendedOp);
        TermMap defect;
        for (const auto& [k, a] : expand(uM.root_)->terms) defect[k] += a;
        for (const auto& [k, a] : expand(uM1.root_)->terms) defect[k] -= a;
        columns.push_back(std::move(defect));
    }
    // Row index over the union support of the defects.
    std::vector<OccupationBitset> rowKeys;
    TermMap rowIndexMap;  // value.real() = row index
    for (const auto& col : columns)
        for (const auto& [k, a] : col)
            if (a != Complex(0.0, 0.0) &&
                rowIndexMap.find(k) == rowIndexMap.end()) {
                rowIndexMap[k] = Complex(double(rowKeys.size()), 0.0);
                rowKeys.push_back(k);
            }
    LazyCompatibilityRead out;
    out.activeDimension = basis.size();
    if (rowKeys.empty()) {
        out.epsilon = 0.0;
        out.certificate = cobordism::Certificate::certifiedNumerical(
            cobordism::CertificateDomain::Static,
            cobordism::CertificateRegime::PositiveSemidefinite, 0.0,
            cobordism::Certificate::kUnmeasured, 1e-10);
        return out;
    }
    Eigen::MatrixXcd defect = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(rowKeys.size()),
        static_cast<Eigen::Index>(columns.size()));
    for (std::size_t c = 0; c < columns.size(); ++c)
        for (const auto& [k, a] : columns[c]) {
            if (a == Complex(0.0, 0.0)) continue;
            const auto r = static_cast<Eigen::Index>(
                rowIndexMap.at(k).real());
            defect(r, static_cast<Eigen::Index>(c)) = a;
        }
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
        defect, Eigen::ComputeThinU | Eigen::ComputeThinV);
    out.epsilon = svd.singularValues().size() > 0 ? svd.singularValues()(0)
                                                  : 0.0;
    double residual = 0.0;
    if (svd.singularValues().size() > 0) {
        const Eigen::VectorXcd v = svd.matrixV().col(0);
        const Eigen::VectorXcd u = svd.matrixU().col(0);
        residual = (defect * v - out.epsilon * u).norm() /
                   std::max(1.0, out.epsilon);
    }
    out.certificate = cobordism::Certificate::certifiedNumerical(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::PositiveSemidefinite, residual,
        cobordism::Certificate::kUnmeasured, 1e-10);
    return out;
}

LazyInductiveLimitRead LazyFockEngine::inductiveLimit(
    const std::vector<FockRefinementStage>& stages,
    const std::vector<std::vector<std::size_t>>& activeBasis) const {
    if (stages.size() < 2)
        throw std::invalid_argument(
            "LazyFockEngine::inductiveLimit: a refinement sequence is at "
            "least two stages; one pair of stages measures one defect and "
            "establishes no limit");
    LazyInductiveLimitRead out;
    double residual = 0.0;
    for (std::size_t k = 0; k + 1 < stages.size(); ++k) {
        const LazyCompatibilityRead step = inductiveCompatibility(
            stages[k].modes, stages[k + 1].modes, stages[k].support,
            stages[k].map, stages[k + 1].support, stages[k + 1].map,
            activeBasis);
        residual = std::max(residual, step.certificate.residual);
        out.defects.push_back(step.epsilon);
        out.activeDimension = step.activeDimension;
        out.steps.push_back(step);
    }
    out.lastDefect = out.defects.back();
    out.falls = out.defects.size() > 1;
    for (std::size_t k = 0; k + 1 < out.defects.size(); ++k) {
        const double before = out.defects[k];
        const double after = out.defects[k + 1];
        // A vanishing predecessor has already reached compatibility; the
        // sequence goes on falling only if its successor vanishes too.
        const double ratio =
            before > 0.0 ? after / before
            : after > 0.0 ? std::numeric_limits<double>::infinity()
                          : 0.0;
        out.largestRatio = std::max(out.largestRatio, ratio);
        out.falls = out.falls && after < before;
    }
    out.certificate = cobordism::Certificate::certifiedNumerical(
        cobordism::CertificateDomain::Static,
        cobordism::CertificateRegime::PositiveSemidefinite, residual,
        cobordism::Certificate::kUnmeasured, 1e-10);
    return out;
}

// ── serialization ────────────────────────────────────────────────────────

std::string LazyFockEngine::serialize(const LazyFockState& state) const {
    validateState(state);
    // Topological order, children before parents, shared nodes once.
    std::vector<const LazyFockNode*> order;
    std::unordered_map<std::uint64_t, std::size_t> index;  // nodeId → index
    struct Frame {
        const LazyFockNode* node;
        std::size_t child;
    };
    std::vector<Frame> stack{{state.root_.get(), 0}};
    while (!stack.empty()) {
        Frame& top = stack.back();
        if (index.count(top.node->nodeId())) {
            stack.pop_back();
            continue;
        }
        if (top.child < top.node->children().size()) {
            const LazyFockNode* next =
                top.node->children()[top.child++].get();
            if (!index.count(next->nodeId())) stack.push_back({next, 0});
            continue;
        }
        index[top.node->nodeId()] = order.size();
        order.push_back(top.node);
        stack.pop_back();
    }
    std::ostringstream os;
    os << "{\"schema\":\"" << kSerializationSchema << "\","
       << "\"version\":" << kSerializationVersion << ","
       << "\"modeCount\":" << modeCount_ << ","
       << "\"discardedNorm\":";
    JsonOut::number(os, state.discardedNorm());
    os << ",\"fixtureLabel\":";
    JsonOut::string(os, state.boundaryFixtureLabel());
    os << ",\"root\":" << index.at(state.root_->nodeId()) << ",\"nodes\":[";
    for (std::size_t i = 0; i < order.size(); ++i) {
        const LazyFockNode* n = order[i];
        if (i) os << ",";
        os << "{\"kind\":\"" << kindName(n->kind()) << "\",\"hash\":\""
           << JsonOut::hex64(n->contentHash()) << "\",\"modes\":[";
        for (std::size_t k = 0; k < n->modes().size(); ++k)
            os << (k ? "," : "") << n->modes()[k];
        os << "]";
        switch (n->kind()) {
            case LazyNodeKind::Vacuum:
                break;
            case LazyNodeKind::Occupation: {
                os << ",\"keys\":[";
                for (std::size_t t = 0; t < n->terms().size(); ++t) {
                    os << (t ? "," : "") << "[";
                    const auto occ = n->terms()[t].first.occupiedModes();
                    for (std::size_t k = 0; k < occ.size(); ++k)
                        os << (k ? "," : "") << occ[k];
                    os << "]";
                }
                os << "],\"amps\":[";
                for (std::size_t t = 0; t < n->terms().size(); ++t) {
                    os << (t ? "," : "") << "[";
                    JsonOut::number(os, n->terms()[t].second.real());
                    os << ",";
                    JsonOut::number(os, n->terms()[t].second.imag());
                    os << "]";
                }
                os << "]";
                break;
            }
            case LazyNodeKind::GradedTensor:
                os << ",\"children\":["
                   << index.at(n->children()[0]->nodeId()) << ","
                   << index.at(n->children()[1]->nodeId()) << "]";
                break;
            case LazyNodeKind::SectorSum: {
                os << ",\"sectorKind\":\""
                   << (n->sectorKind() == LazySectorKind::Occupation
                           ? "occupation"
                           : "parity")
                   << "\",\"children\":[";
                for (std::size_t k = 0; k < n->children().size(); ++k)
                    os << (k ? "," : "")
                       << index.at(n->children()[k]->nodeId());
                os << "]";
                break;
            }
            case LazyNodeKind::LocalMap: {
                os << ",\"child\":" << index.at(n->children()[0]->nodeId())
                   << ",\"support\":[";
                for (std::size_t k = 0; k < n->supportModes().size(); ++k)
                    os << (k ? "," : "") << n->supportModes()[k];
                os << "],\"opRows\":[";
                bool first = true;
                for (int k = 0; k < n->supportMatrixCols().outerSize(); ++k)
                    for (LazyFockNode::SparseColOp::InnerIterator it(
                             n->supportMatrixCols(), k);
                         it; ++it) {
                        os << (first ? "" : ",") << it.row();
                        first = false;
                    }
                os << "],\"opCols\":[";
                first = true;
                for (int k = 0; k < n->supportMatrixCols().outerSize(); ++k)
                    for (LazyFockNode::SparseColOp::InnerIterator it(
                             n->supportMatrixCols(), k);
                         it; ++it) {
                        os << (first ? "" : ",") << it.col();
                        first = false;
                    }
                os << "],\"opVals\":[";
                first = true;
                for (int k = 0; k < n->supportMatrixCols().outerSize(); ++k)
                    for (LazyFockNode::SparseColOp::InnerIterator it(
                             n->supportMatrixCols(), k);
                         it; ++it) {
                        os << (first ? "" : ",") << "[";
                        JsonOut::number(os, it.value().real());
                        os << ",";
                        JsonOut::number(os, it.value().imag());
                        os << "]";
                        first = false;
                    }
                os << "]";
                break;
            }
            case LazyNodeKind::Wedge: {
                os << ",\"orbitalCols\":" << n->orbitals().cols()
                   << ",\"orbitals\":[";
                bool first = true;
                for (Eigen::Index c = 0; c < n->orbitals().cols(); ++c)
                    for (Eigen::Index r = 0; r < n->orbitals().rows(); ++r) {
                        os << (first ? "" : ",") << "[";
                        JsonOut::number(os, n->orbitals()(r, c).real());
                        os << ",";
                        JsonOut::number(os, n->orbitals()(r, c).imag());
                        os << "]";
                        first = false;
                    }
                os << "]";
                break;
            }
        }
        os << "}";
    }
    os << "]}";
    return os.str();
}

LazyFockState LazyFockEngine::deserialize(const std::string& json) const {
    const MiniJson doc = MiniJson::parse(json);
    if (doc.type != MiniJson::Type::Object)
        throw std::invalid_argument("LazyFock deserialize: not an object");
    if (doc.at("schema").asString() != kSerializationSchema)
        throw std::invalid_argument("LazyFock deserialize: unknown schema");
    if (doc.at("version").asIndex() !=
        static_cast<std::size_t>(kSerializationVersion))
        throw std::invalid_argument(
            "LazyFock deserialize: unknown schema version");
    if (doc.at("modeCount").asIndex() != modeCount_)
        throw std::invalid_argument(
            "LazyFock deserialize: mode universe mismatch");
    const double discardedNorm = doc.at("discardedNorm").asNumber();
    const std::string label = doc.at("fixtureLabel").asString();
    const auto& nodesJson = doc.at("nodes").asArray();
    std::vector<NodePtr> nodes;
    nodes.reserve(nodesJson.size());
    auto modesOf = [](const MiniJson& n) {
        std::vector<std::size_t> out;
        for (const auto& v : n.at("modes").asArray()) out.push_back(v.asIndex());
        return out;
    };
    auto childAt = [&nodes](std::size_t idx) {
        if (idx >= nodes.size())
            throw std::invalid_argument(
                "LazyFock deserialize: child index out of (topological) "
                "order");
        return nodes[idx];
    };
    for (const auto& nj : nodesJson) {
        const std::string& kind = nj.at("kind").asString();
        NodePtr node;
        if (kind == "vacuum") {
            node = makeVacuum(modesOf(nj));
        } else if (kind == "occupation") {
            const auto& keys = nj.at("keys").asArray();
            const auto& amps = nj.at("amps").asArray();
            if (keys.size() != amps.size())
                throw std::invalid_argument(
                    "LazyFock deserialize: keys/amps mismatch");
            std::vector<Term> terms;
            for (std::size_t t = 0; t < keys.size(); ++t) {
                std::vector<std::size_t> occ;
                for (const auto& v : keys[t].asArray())
                    occ.push_back(v.asIndex());
                const auto& pair = amps[t].asArray();
                if (pair.size() != 2)
                    throw std::invalid_argument(
                        "LazyFock deserialize: amplitude must be [re, im]");
                terms.emplace_back(
                    OccupationBitset::fromOccupiedModes(modeCount_, occ),
                    Complex(pair[0].asNumber(), pair[1].asNumber()));
            }
            node = makeOccupation(modesOf(nj), std::move(terms));
        } else if (kind == "tensor") {
            const auto& ch = nj.at("children").asArray();
            if (ch.size() != 2)
                throw std::invalid_argument(
                    "LazyFock deserialize: tensor needs two children");
            node = makeTensor(childAt(ch[0].asIndex()),
                              childAt(ch[1].asIndex()));
        } else if (kind == "sectorSum") {
            const std::string& sk = nj.at("sectorKind").asString();
            LazySectorKind kindEnum;
            if (sk == "occupation") kindEnum = LazySectorKind::Occupation;
            else if (sk == "parity") kindEnum = LazySectorKind::Parity;
            else
                throw std::invalid_argument(
                    "LazyFock deserialize: unknown sector kind");
            std::vector<NodePtr> children;
            for (const auto& v : nj.at("children").asArray())
                children.push_back(childAt(v.asIndex()));
            node = makeSectorSum(std::move(children), kindEnum);
        } else if (kind == "localMap") {
            NodePtr child = childAt(nj.at("child").asIndex());
            std::vector<std::size_t> support;
            for (const auto& v : nj.at("support").asArray())
                support.push_back(v.asIndex());
            const auto& rws = nj.at("opRows").asArray();
            const auto& cls = nj.at("opCols").asArray();
            const auto& vls = nj.at("opVals").asArray();
            if (rws.size() != cls.size() || rws.size() != vls.size())
                throw std::invalid_argument(
                    "LazyFock deserialize: operator triplet mismatch");
            const auto dim = static_cast<Eigen::Index>(std::size_t{1}
                                                       << support.size());
            std::vector<Eigen::Triplet<Complex>> triplets;
            for (std::size_t k = 0; k < rws.size(); ++k) {
                const auto r = rws[k].asIndex();
                const auto c = cls[k].asIndex();
                if (r >= static_cast<std::size_t>(dim) ||
                    c >= static_cast<std::size_t>(dim))
                    throw std::invalid_argument(
                        "LazyFock deserialize: operator triplet out of "
                        "range");
                const auto& pair = vls[k].asArray();
                if (pair.size() != 2)
                    throw std::invalid_argument(
                        "LazyFock deserialize: operator value must be "
                        "[re, im]");
                triplets.emplace_back(
                    static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c),
                    Complex(pair[0].asNumber(), pair[1].asNumber()));
            }
            LazyFockNode::SparseColOp op(dim, dim);
            op.setFromTriplets(triplets.begin(), triplets.end());
            node = makeLocalMap(std::move(child), std::move(support),
                                std::move(op));
        } else if (kind == "wedge") {
            auto modes = modesOf(nj);
            const auto cols = nj.at("orbitalCols").asIndex();
            const auto& entries = nj.at("orbitals").asArray();
            if (entries.size() != cols * modes.size())
                throw std::invalid_argument(
                    "LazyFock deserialize: orbital entry count mismatch");
            Eigen::MatrixXcd orbitals(
                static_cast<Eigen::Index>(modes.size()),
                static_cast<Eigen::Index>(cols));
            std::size_t k = 0;
            for (Eigen::Index c = 0;
                 c < static_cast<Eigen::Index>(cols); ++c)
                for (Eigen::Index r = 0;
                     r < static_cast<Eigen::Index>(modes.size()); ++r) {
                    const auto& pair = entries[k++].asArray();
                    if (pair.size() != 2)
                        throw std::invalid_argument(
                            "LazyFock deserialize: orbital entry must be "
                            "[re, im]");
                    orbitals(r, c) =
                        Complex(pair[0].asNumber(), pair[1].asNumber());
                }
            node = makeWedge(std::move(modes), std::move(orbitals));
        } else {
            throw std::invalid_argument(
                "LazyFock deserialize: unknown node kind '" + kind + "'");
        }
        // Integrity: the recomputed canonical content hash must equal the
        // stored one — a tampered or drifted checkpoint is rejected.
        const std::uint64_t stored =
            JsonOut::parseHex64(nj.at("hash").asString());
        if (node->contentHash() != stored)
            throw std::invalid_argument(
                "LazyFock deserialize: content-hash mismatch on node " +
                std::to_string(nodes.size()) +
                " — checkpoint rejected (tampered or drifted)");
        nodes.push_back(std::move(node));
    }
    const std::size_t rootIdx = doc.at("root").asIndex();
    if (rootIdx >= nodes.size())
        throw std::invalid_argument("LazyFock deserialize: root out of range");
    return wrap(nodes[rootIdx], discardedNorm, label);
}

}  // namespace tessera::quantum
