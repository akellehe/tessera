// Implementation of the exterior-algebra / graded-tensor primitives declared
// in include/quantum/GradedFock.h. Everything here is an exact
// integer/algebraic identity compiled into matrices — no approximation, no
// tolerance, no runtime guards beyond argument validation.

#include "quantum/GradedFock.h"

#include <algorithm>
#include <bit>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::quantum {

namespace {
using Complex = std::complex<double>;
using Triplet = Eigen::Triplet<Complex>;
}  // namespace

// ─── OccupationBitset ─────────────────────────────────────────────────────

OccupationBitset::OccupationBitset(std::size_t modeCount)
    : chunks_((modeCount + kBitsPerChunk - 1) / kBitsPerChunk, 0ull),
      modeCount_(modeCount) {}

OccupationBitset OccupationBitset::fromOccupiedModes(
    std::size_t modeCount, const std::vector<std::size_t>& modes) {
    OccupationBitset b(modeCount);
    for (std::size_t m : modes) {
        if (m >= modeCount) {
            throw std::invalid_argument(
                "OccupationBitset::fromOccupiedModes: mode " +
                std::to_string(m) + " out of range for modeCount " +
                std::to_string(modeCount));
        }
        if (b.test(m)) {
            throw std::invalid_argument(
                "OccupationBitset::fromOccupiedModes: duplicate mode " +
                std::to_string(m));
        }
        b.set(m);
    }
    return b;
}

OccupationBitset OccupationBitset::fromIndex(std::size_t modeCount,
                                             std::uint64_t index) {
    if (modeCount > 64) {
        throw std::invalid_argument(
            "OccupationBitset::fromIndex: modeCount " +
            std::to_string(modeCount) +
            " > 64; a 64-bit index cannot address that many modes");
    }
    if (modeCount < 64 && index >= (1ull << modeCount)) {
        throw std::invalid_argument(
            "OccupationBitset::fromIndex: index " + std::to_string(index) +
            " >= 2^" + std::to_string(modeCount));
    }
    OccupationBitset b(modeCount);
    if (modeCount > 0) b.chunks_[0] = index;
    return b;
}

bool OccupationBitset::test(std::size_t mode) const {
    if (mode >= modeCount_) {
        throw std::invalid_argument("OccupationBitset::test: mode " +
                                    std::to_string(mode) +
                                    " out of range for modeCount " +
                                    std::to_string(modeCount_));
    }
    return (chunks_[mode / kBitsPerChunk] >> (mode % kBitsPerChunk)) & 1ull;
}

void OccupationBitset::set(std::size_t mode) {
    if (mode >= modeCount_) {
        throw std::invalid_argument("OccupationBitset::set: mode " +
                                    std::to_string(mode) + " out of range");
    }
    chunks_[mode / kBitsPerChunk] |= (1ull << (mode % kBitsPerChunk));
}

void OccupationBitset::reset(std::size_t mode) {
    if (mode >= modeCount_) {
        throw std::invalid_argument("OccupationBitset::reset: mode " +
                                    std::to_string(mode) + " out of range");
    }
    chunks_[mode / kBitsPerChunk] &= ~(1ull << (mode % kBitsPerChunk));
}

std::size_t OccupationBitset::count() const noexcept {
    std::size_t n = 0;
    for (std::uint64_t c : chunks_) n += static_cast<std::size_t>(std::popcount(c));
    return n;
}

int OccupationBitset::parity() const noexcept {
    return (count() % 2 == 0) ? +1 : -1;
}

std::size_t OccupationBitset::prefixPopcount(std::size_t mode) const {
    if (mode > modeCount_) {
        throw std::invalid_argument(
            "OccupationBitset::prefixPopcount: mode " + std::to_string(mode) +
            " > modeCount " + std::to_string(modeCount_));
    }
    const std::size_t fullChunks = mode / kBitsPerChunk;
    std::size_t n = 0;
    for (std::size_t c = 0; c < fullChunks; ++c) {
        n += static_cast<std::size_t>(std::popcount(chunks_[c]));
    }
    const std::size_t rem = mode % kBitsPerChunk;
    if (rem != 0) {
        const std::uint64_t mask = (1ull << rem) - 1ull;
        n += static_cast<std::size_t>(std::popcount(chunks_[fullChunks] & mask));
    }
    return n;
}

int OccupationBitset::applyCreation(std::size_t mode) {
    if (test(mode)) return 0;  // Pauli exclusion: a_i† on an occupied mode.
    const int sign = (prefixPopcount(mode) % 2 == 0) ? +1 : -1;
    set(mode);
    return sign;
}

int OccupationBitset::applyAnnihilation(std::size_t mode) {
    if (!test(mode)) return 0;  // a_i on an empty mode.
    const int sign = (prefixPopcount(mode) % 2 == 0) ? +1 : -1;
    reset(mode);
    return sign;
}

std::uint64_t OccupationBitset::toIndex() const {
    if (modeCount_ > 64) {
        throw std::invalid_argument(
            "OccupationBitset::toIndex: modeCount " +
            std::to_string(modeCount_) + " > 64 does not fit a 64-bit index");
    }
    return chunks_.empty() ? 0ull : chunks_[0];
}

std::vector<std::size_t> OccupationBitset::occupiedModes() const {
    std::vector<std::size_t> out;
    out.reserve(count());
    for (std::size_t c = 0; c < chunks_.size(); ++c) {
        std::uint64_t word = chunks_[c];
        while (word != 0) {
            const int bit = std::countr_zero(word);
            out.push_back(c * kBitsPerChunk + static_cast<std::size_t>(bit));
            word &= word - 1;  // clear lowest set bit
        }
    }
    return out;
}

void OccupationBitset::validatePermutation(
    const std::vector<std::size_t>& perm, std::size_t modeCount) {
    if (perm.size() != modeCount) {
        throw std::invalid_argument(
            "mode permutation has size " + std::to_string(perm.size()) +
            ", expected modeCount " + std::to_string(modeCount));
    }
    std::vector<bool> seen(modeCount, false);
    for (std::size_t p : perm) {
        if (p >= modeCount || seen[p]) {
            throw std::invalid_argument(
                "mode permutation is not a bijection on {0..M-1}");
        }
        seen[p] = true;
    }
}

OccupationBitset OccupationBitset::permuted(
    const std::vector<std::size_t>& perm) const {
    validatePermutation(perm, modeCount_);
    OccupationBitset out(modeCount_);
    for (std::size_t m : occupiedModes()) out.set(perm[m]);
    return out;
}

int OccupationBitset::permutationParity(
    const std::vector<std::size_t>& perm) const {
    validatePermutation(perm, modeCount_);
    // Images of the occupied modes in ascending original-mode order; the
    // sign is (−1)^{#inversions} of that image sequence (the parity of the
    // sort that rebuilds the canonical ascending wedge word).
    std::vector<std::size_t> images;
    images.reserve(count());
    for (std::size_t m : occupiedModes()) images.push_back(perm[m]);
    std::size_t inversions = 0;
    for (std::size_t i = 0; i < images.size(); ++i) {
        for (std::size_t j = i + 1; j < images.size(); ++j) {
            if (images[i] > images[j]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? +1 : -1;
}

bool OccupationBitset::operator==(const OccupationBitset& other) const noexcept {
    return modeCount_ == other.modeCount_ && chunks_ == other.chunks_;
}

std::string OccupationBitset::str() const {
    std::string bits;
    bits.reserve(modeCount_);
    for (std::size_t m = modeCount_; m-- > 0;) bits.push_back(test(m) ? '1' : '0');
    return "|" + bits + ">";
}

// ─── ExteriorAlgebra ──────────────────────────────────────────────────────

ExteriorAlgebra::ExteriorAlgebra(std::size_t modeCount)
    : modeCount_(modeCount) {
    if (modeCount > kMaxMatrixModes) {
        throw std::invalid_argument(
            "ExteriorAlgebra: modeCount " + std::to_string(modeCount) +
            " exceeds kMaxMatrixModes = " + std::to_string(kMaxMatrixModes) +
            " (the 2^M matrix layer would not be materializable); use "
            "OccupationBitset for bit-level operations at arbitrary mode "
            "counts");
    }
    dim_ = std::size_t{1} << modeCount_;
}

void ExteriorAlgebra::validateMode(std::size_t mode) const {
    if (mode >= modeCount_) {
        throw std::invalid_argument("ExteriorAlgebra: mode " +
                                    std::to_string(mode) +
                                    " out of range for modeCount " +
                                    std::to_string(modeCount_));
    }
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::creationMatrix(
    std::size_t mode) const {
    validateMode(mode);
    const std::uint64_t bit = 1ull << mode;
    const std::uint64_t below = bit - 1ull;
    std::vector<Triplet> trips;
    trips.reserve(dim_ / 2);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        if (b & bit) continue;
        const int sign = (std::popcount(b & below) % 2 == 0) ? +1 : -1;
        trips.emplace_back(static_cast<Eigen::Index>(b | bit),
                           static_cast<Eigen::Index>(b),
                           Complex(static_cast<double>(sign), 0.0));
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::annihilationMatrix(
    std::size_t mode) const {
    validateMode(mode);
    const std::uint64_t bit = 1ull << mode;
    const std::uint64_t below = bit - 1ull;
    std::vector<Triplet> trips;
    trips.reserve(dim_ / 2);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        if (!(b & bit)) continue;
        const int sign = (std::popcount(b & below) % 2 == 0) ? +1 : -1;
        trips.emplace_back(static_cast<Eigen::Index>(b & ~bit),
                           static_cast<Eigen::Index>(b),
                           Complex(static_cast<double>(sign), 0.0));
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::numberMatrix(std::size_t mode) const {
    validateMode(mode);
    const std::uint64_t bit = 1ull << mode;
    std::vector<Triplet> trips;
    trips.reserve(dim_ / 2);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        if (b & bit) {
            trips.emplace_back(static_cast<Eigen::Index>(b),
                               static_cast<Eigen::Index>(b), Complex(1.0, 0.0));
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::totalNumberMatrix() const {
    std::vector<Triplet> trips;
    trips.reserve(dim_);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        const int n = std::popcount(b);
        if (n != 0) {
            trips.emplace_back(static_cast<Eigen::Index>(b),
                               static_cast<Eigen::Index>(b),
                               Complex(static_cast<double>(n), 0.0));
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::parityMatrix() const {
    std::vector<Triplet> trips;
    trips.reserve(dim_);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        const double sign = (std::popcount(b) % 2 == 0) ? +1.0 : -1.0;
        trips.emplace_back(static_cast<Eigen::Index>(b),
                           static_cast<Eigen::Index>(b), Complex(sign, 0.0));
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::sectorProjector(
    std::size_t occupation) const {
    std::vector<Triplet> trips;
    for (std::uint64_t b = 0; b < dim_; ++b) {
        if (static_cast<std::size_t>(std::popcount(b)) == occupation) {
            trips.emplace_back(static_cast<Eigen::Index>(b),
                               static_cast<Eigen::Index>(b), Complex(1.0, 0.0));
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::subsetSectorProjector(
    const std::vector<std::size_t>& modes, std::size_t occupation) const {
    std::uint64_t mask = 0;
    for (std::size_t m : modes) {
        validateMode(m);
        const std::uint64_t bit = 1ull << m;
        if (mask & bit) {
            throw std::invalid_argument(
                "ExteriorAlgebra::subsetSectorProjector: duplicate mode " +
                std::to_string(m));
        }
        mask |= bit;
    }
    std::vector<Triplet> trips;
    for (std::uint64_t b = 0; b < dim_; ++b) {
        if (static_cast<std::size_t>(std::popcount(b & mask)) == occupation) {
            trips.emplace_back(static_cast<Eigen::Index>(b),
                               static_cast<Eigen::Index>(b), Complex(1.0, 0.0));
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

Eigen::VectorXcd ExteriorAlgebra::vacuumState() const {
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(dim_));
    v[0] = Complex(1.0, 0.0);
    return v;
}

Eigen::VectorXcd ExteriorAlgebra::basisState(const OccupationBitset& b) const {
    if (b.modeCount() != modeCount_) {
        throw std::invalid_argument(
            "ExteriorAlgebra::basisState: bitset modeCount " +
            std::to_string(b.modeCount()) + " != algebra modeCount " +
            std::to_string(modeCount_));
    }
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(dim_));
    v[static_cast<Eigen::Index>(b.toIndex())] = Complex(1.0, 0.0);
    return v;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::creationOperator(
    const Eigen::VectorXcd& v) const {
    if (static_cast<std::size_t>(v.size()) != modeCount_) {
        throw std::invalid_argument(
            "ExteriorAlgebra::creationOperator: vector size " +
            std::to_string(v.size()) + " != modeCount " +
            std::to_string(modeCount_));
    }
    std::vector<Triplet> trips;
    for (std::size_t i = 0; i < modeCount_; ++i) {
        if (v[static_cast<Eigen::Index>(i)] == Complex(0.0, 0.0)) continue;
        const std::uint64_t bit = 1ull << i;
        const std::uint64_t below = bit - 1ull;
        for (std::uint64_t b = 0; b < dim_; ++b) {
            if (b & bit) continue;
            const double sign = (std::popcount(b & below) % 2 == 0) ? +1.0 : -1.0;
            trips.emplace_back(static_cast<Eigen::Index>(b | bit),
                               static_cast<Eigen::Index>(b),
                               sign * v[static_cast<Eigen::Index>(i)]);
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::annihilationOperator(
    const Eigen::VectorXcd& v) const {
    if (static_cast<std::size_t>(v.size()) != modeCount_) {
        throw std::invalid_argument(
            "ExteriorAlgebra::annihilationOperator: vector size " +
            std::to_string(v.size()) + " != modeCount " +
            std::to_string(modeCount_));
    }
    std::vector<Triplet> trips;
    for (std::size_t i = 0; i < modeCount_; ++i) {
        const Complex coeff = std::conj(v[static_cast<Eigen::Index>(i)]);
        if (coeff == Complex(0.0, 0.0)) continue;
        const std::uint64_t bit = 1ull << i;
        const std::uint64_t below = bit - 1ull;
        for (std::uint64_t b = 0; b < dim_; ++b) {
            if (!(b & bit)) continue;
            const double sign = (std::popcount(b & below) % 2 == 0) ? +1.0 : -1.0;
            trips.emplace_back(static_cast<Eigen::Index>(b & ~bit),
                               static_cast<Eigen::Index>(b), sign * coeff);
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

Eigen::VectorXcd ExteriorAlgebra::wedge(
    const std::vector<Eigen::VectorXcd>& vectors) const {
    Eigen::VectorXcd state = vacuumState();
    // v1 ∧ … ∧ vn = a†(v1)…a†(vn) Ω: the rightmost factor acts first.
    for (auto it = vectors.rbegin(); it != vectors.rend(); ++it) {
        const Eigen::VectorXcd& v = *it;
        if (static_cast<std::size_t>(v.size()) != modeCount_) {
            throw std::invalid_argument(
                "ExteriorAlgebra::wedge: factor size " +
                std::to_string(v.size()) + " != modeCount " +
                std::to_string(modeCount_));
        }
        Eigen::VectorXcd next =
            Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(dim_));
        for (std::size_t i = 0; i < modeCount_; ++i) {
            const Complex coeff = v[static_cast<Eigen::Index>(i)];
            if (coeff == Complex(0.0, 0.0)) continue;
            const std::uint64_t bit = 1ull << i;
            const std::uint64_t below = bit - 1ull;
            for (std::uint64_t b = 0; b < dim_; ++b) {
                const Complex amp = state[static_cast<Eigen::Index>(b)];
                if (amp == Complex(0.0, 0.0) || (b & bit)) continue;
                const double sign =
                    (std::popcount(b & below) % 2 == 0) ? +1.0 : -1.0;
                next[static_cast<Eigen::Index>(b | bit)] += sign * coeff * amp;
            }
        }
        state.swap(next);
    }
    return state;
}

Eigen::VectorXcd ExteriorAlgebra::contract(const Eigen::VectorXcd& w,
                                           const Eigen::VectorXcd& state) const {
    if (static_cast<std::size_t>(w.size()) != modeCount_) {
        throw std::invalid_argument("ExteriorAlgebra::contract: covector size " +
                                    std::to_string(w.size()) +
                                    " != modeCount " +
                                    std::to_string(modeCount_));
    }
    if (static_cast<std::size_t>(state.size()) != dim_) {
        throw std::invalid_argument("ExteriorAlgebra::contract: state size " +
                                    std::to_string(state.size()) +
                                    " != fockDimension " + std::to_string(dim_));
    }
    Eigen::VectorXcd out = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(dim_));
    for (std::size_t i = 0; i < modeCount_; ++i) {
        const Complex coeff = std::conj(w[static_cast<Eigen::Index>(i)]);
        if (coeff == Complex(0.0, 0.0)) continue;
        const std::uint64_t bit = 1ull << i;
        const std::uint64_t below = bit - 1ull;
        for (std::uint64_t b = 0; b < dim_; ++b) {
            const Complex amp = state[static_cast<Eigen::Index>(b)];
            if (amp == Complex(0.0, 0.0) || !(b & bit)) continue;
            const double sign = (std::popcount(b & below) % 2 == 0) ? +1.0 : -1.0;
            out[static_cast<Eigen::Index>(b & ~bit)] += sign * coeff * amp;
        }
    }
    return out;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::dGamma(
    const Eigen::MatrixXcd& oneParticle) const {
    if (static_cast<std::size_t>(oneParticle.rows()) != modeCount_ ||
        static_cast<std::size_t>(oneParticle.cols()) != modeCount_) {
        throw std::invalid_argument(
            "ExteriorAlgebra::dGamma: one-particle block is " +
            std::to_string(oneParticle.rows()) + "x" +
            std::to_string(oneParticle.cols()) + ", expected " +
            std::to_string(modeCount_) + "x" + std::to_string(modeCount_));
    }
    // Per-column nonzero rows of L, so the basis sweep skips zero entries.
    std::vector<std::vector<std::size_t>> rowsOfCol(modeCount_);
    for (std::size_t j = 0; j < modeCount_; ++j) {
        for (std::size_t i = 0; i < modeCount_; ++i) {
            if (oneParticle(static_cast<Eigen::Index>(i),
                            static_cast<Eigen::Index>(j)) !=
                Complex(0.0, 0.0)) {
                rowsOfCol[j].push_back(i);
            }
        }
    }
    std::vector<Triplet> trips;
    for (std::uint64_t b = 0; b < dim_; ++b) {
        for (std::size_t j = 0; j < modeCount_; ++j) {
            const std::uint64_t bitJ = 1ull << j;
            if (!(b & bitJ)) continue;  // a_j needs mode j occupied
            const int signJ =
                (std::popcount(b & (bitJ - 1ull)) % 2 == 0) ? +1 : -1;
            const std::uint64_t mid = b & ~bitJ;
            for (std::size_t i : rowsOfCol[j]) {
                const std::uint64_t bitI = 1ull << i;
                if (mid & bitI) continue;  // a_i† needs mode i free
                const int signI =
                    (std::popcount(mid & (bitI - 1ull)) % 2 == 0) ? +1 : -1;
                trips.emplace_back(
                    static_cast<Eigen::Index>(mid | bitI),
                    static_cast<Eigen::Index>(b),
                    static_cast<double>(signI * signJ) *
                        oneParticle(static_cast<Eigen::Index>(i),
                                    static_cast<Eigen::Index>(j)));
            }
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

ExteriorAlgebra::SparseOp ExteriorAlgebra::modePermutationMatrix(
    const std::vector<std::size_t>& perm) const {
    std::vector<Triplet> trips;
    trips.reserve(dim_);
    for (std::uint64_t b = 0; b < dim_; ++b) {
        const OccupationBitset bits = OccupationBitset::fromIndex(modeCount_, b);
        const OccupationBitset image = bits.permuted(perm);
        const int sign = bits.permutationParity(perm);
        trips.emplace_back(static_cast<Eigen::Index>(image.toIndex()),
                           static_cast<Eigen::Index>(b),
                           Complex(static_cast<double>(sign), 0.0));
    }
    SparseOp op(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

// ─── GradedTensorComplex ──────────────────────────────────────────────────

void GradedTensorComplex::validateFactor(
    const std::vector<std::size_t>& dims,
    const std::vector<Eigen::MatrixXcd>& diff, double boundaryTolerance,
    const char* name) {
    if (dims.empty()) {
        throw std::invalid_argument(std::string("GradedTensorComplex: factor ") +
                                    name + " has no chain groups");
    }
    if (diff.size() + 1 != dims.size()) {
        throw std::invalid_argument(
            std::string("GradedTensorComplex: factor ") + name + " has " +
            std::to_string(diff.size()) + " differentials for " +
            std::to_string(dims.size()) +
            " chain groups (need dims.size() - 1)");
    }
    for (std::size_t k = 0; k < diff.size(); ++k) {
        if (static_cast<std::size_t>(diff[k].rows()) != dims[k] ||
            static_cast<std::size_t>(diff[k].cols()) != dims[k + 1]) {
            throw std::invalid_argument(
                std::string("GradedTensorComplex: factor ") + name +
                " differential " + std::to_string(k) + " is " +
                std::to_string(diff[k].rows()) + "x" +
                std::to_string(diff[k].cols()) + ", expected " +
                std::to_string(dims[k]) + "x" + std::to_string(dims[k + 1]));
        }
    }
    for (std::size_t k = 1; k < diff.size(); ++k) {
        const double maxAbs =
            (diff[k - 1] * diff[k]).cwiseAbs().maxCoeff();
        if (maxAbs > boundaryTolerance) {
            std::ostringstream msg;
            msg << "GradedTensorComplex: factor " << name
                << " violates boundary-of-boundary = 0 at degree " << (k + 1)
                << " (max |entry| = " << maxAbs << " > tolerance "
                << boundaryTolerance << ")";
            throw std::invalid_argument(msg.str());
        }
    }
}

GradedTensorComplex::GradedTensorComplex(std::vector<std::size_t> dimsA,
                                         std::vector<Eigen::MatrixXcd> diffA,
                                         std::vector<std::size_t> dimsB,
                                         std::vector<Eigen::MatrixXcd> diffB,
                                         double boundaryTolerance)
    : dimsA_(std::move(dimsA)),
      diffA_(std::move(diffA)),
      dimsB_(std::move(dimsB)),
      diffB_(std::move(diffB)) {
    validateFactor(dimsA_, diffA_, boundaryTolerance, "A");
    validateFactor(dimsB_, diffB_, boundaryTolerance, "B");
}

std::size_t GradedTensorComplex::maxDegree() const noexcept {
    return (dimsA_.size() - 1) + (dimsB_.size() - 1);
}

std::vector<std::pair<std::size_t, std::size_t>> GradedTensorComplex::blocks(
    std::size_t degree) const {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    const std::size_t pMax = dimsA_.size() - 1;
    const std::size_t qMax = dimsB_.size() - 1;
    for (std::size_t p = 0; p <= pMax && p <= degree; ++p) {
        const std::size_t q = degree - p;
        if (q > qMax) continue;
        out.emplace_back(p, q);
    }
    return out;
}

std::size_t GradedTensorComplex::chainDimension(std::size_t degree) const {
    std::size_t dim = 0;
    for (const auto& [p, q] : blocks(degree)) dim += dimsA_[p] * dimsB_[q];
    return dim;
}

Eigen::MatrixXcd GradedTensorComplex::kronDense(const Eigen::MatrixXcd& p,
                                                const Eigen::MatrixXcd& q) {
    Eigen::MatrixXcd out(p.rows() * q.rows(), p.cols() * q.cols());
    for (Eigen::Index rp = 0; rp < p.rows(); ++rp) {
        for (Eigen::Index cp = 0; cp < p.cols(); ++cp) {
            out.block(rp * q.rows(), cp * q.cols(), q.rows(), q.cols()) =
                p(rp, cp) * q;
        }
    }
    return out;
}

Eigen::MatrixXcd GradedTensorComplex::differential(std::size_t degree) const {
    if (degree == 0 || degree > maxDegree()) {
        throw std::invalid_argument(
            "GradedTensorComplex::differential: degree " +
            std::to_string(degree) + " out of range [1, " +
            std::to_string(maxDegree()) + "]");
    }
    const auto rowBlocks = blocks(degree - 1);
    const auto colBlocks = blocks(degree);
    // Offsets of each (p, q) block inside C_{degree-1} / C_degree.
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> rowOffset;
    {
        std::size_t off = 0;
        for (const auto& pq : rowBlocks) {
            rowOffset[pq] = off;
            off += dimsA_[pq.first] * dimsB_[pq.second];
        }
    }
    Eigen::MatrixXcd d = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(chainDimension(degree - 1)),
        static_cast<Eigen::Index>(chainDimension(degree)));
    std::size_t colOff = 0;
    for (const auto& [p, q] : colBlocks) {
        const std::size_t blockCols = dimsA_[p] * dimsB_[q];
        if (p >= 1) {  // ∂^A_p ⊗ 1 into block (p−1, q)
            const auto it = rowOffset.find({p - 1, q});
            if (it != rowOffset.end()) {
                const Eigen::MatrixXcd blk = kronDense(
                    diffA_[p - 1],
                    Eigen::MatrixXcd::Identity(
                        static_cast<Eigen::Index>(dimsB_[q]),
                        static_cast<Eigen::Index>(dimsB_[q])));
                d.block(static_cast<Eigen::Index>(it->second),
                        static_cast<Eigen::Index>(colOff), blk.rows(),
                        blk.cols()) = blk;
            }
        }
        if (q >= 1) {  // (−1)^p 1 ⊗ ∂^B_q into block (p, q−1)
            const auto it = rowOffset.find({p, q - 1});
            if (it != rowOffset.end()) {
                const double koszul = (p % 2 == 0) ? +1.0 : -1.0;
                const Eigen::MatrixXcd blk =
                    koszul * kronDense(Eigen::MatrixXcd::Identity(
                                           static_cast<Eigen::Index>(dimsA_[p]),
                                           static_cast<Eigen::Index>(dimsA_[p])),
                                       diffB_[q - 1]);
                d.block(static_cast<Eigen::Index>(it->second),
                        static_cast<Eigen::Index>(colOff), blk.rows(),
                        blk.cols()) = blk;
            }
        }
        colOff += blockCols;
    }
    return d;
}

Eigen::MatrixXcd GradedTensorComplex::laplacian(std::size_t degree) const {
    if (degree > maxDegree()) {
        throw std::invalid_argument("GradedTensorComplex::laplacian: degree " +
                                    std::to_string(degree) + " > maxDegree " +
                                    std::to_string(maxDegree()));
    }
    const std::size_t dim = chainDimension(degree);
    Eigen::MatrixXcd lap = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
    if (degree >= 1) {
        const Eigen::MatrixXcd down = differential(degree);
        lap += down.adjoint() * down;
    }
    if (degree < maxDegree()) {
        const Eigen::MatrixXcd up = differential(degree + 1);
        lap += up * up.adjoint();
    }
    return lap;
}

Eigen::MatrixXcd GradedTensorComplex::factorLaplacian(
    const std::vector<std::size_t>& dims,
    const std::vector<Eigen::MatrixXcd>& diff, std::size_t degree,
    const char* name) {
    if (degree >= dims.size()) {
        throw std::invalid_argument(
            std::string("GradedTensorComplex: factor ") + name + " degree " +
            std::to_string(degree) + " out of range [0, " +
            std::to_string(dims.size() - 1) + "]");
    }
    const Eigen::Index n = static_cast<Eigen::Index>(dims[degree]);
    Eigen::MatrixXcd lap = Eigen::MatrixXcd::Zero(n, n);
    if (degree >= 1) {
        const Eigen::MatrixXcd& down = diff[degree - 1];
        lap += down.adjoint() * down;
    }
    if (degree < diff.size()) {
        const Eigen::MatrixXcd& up = diff[degree];
        lap += up * up.adjoint();
    }
    return lap;
}

Eigen::MatrixXcd GradedTensorComplex::factorLaplacianA(
    std::size_t degree) const {
    return factorLaplacian(dimsA_, diffA_, degree, "A");
}

Eigen::MatrixXcd GradedTensorComplex::factorLaplacianB(
    std::size_t degree) const {
    return factorLaplacian(dimsB_, diffB_, degree, "B");
}

// ─── FockDirectSum ────────────────────────────────────────────────────────

FockDirectSum::FockDirectSum(std::size_t modesA, std::size_t modesB)
    : modesA_(modesA), modesB_(modesB) {
    if (modesA + modesB > ExteriorAlgebra::kMaxMatrixModes) {
        throw std::invalid_argument(
            "FockDirectSum: joint mode count " +
            std::to_string(modesA + modesB) + " exceeds kMaxMatrixModes = " +
            std::to_string(ExteriorAlgebra::kMaxMatrixModes));
    }
}

ExteriorAlgebra FockDirectSum::jointAlgebra() const {
    return ExteriorAlgebra(modesA_ + modesB_);
}

ExteriorAlgebra FockDirectSum::leftAlgebra() const {
    return ExteriorAlgebra(modesA_);
}

ExteriorAlgebra FockDirectSum::rightAlgebra() const {
    return ExteriorAlgebra(modesB_);
}

FockDirectSum::SparseOp FockDirectSum::kronSparse(const SparseOp& p,
                                                  const SparseOp& q) {
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(p.nonZeros()) *
                  static_cast<std::size_t>(q.nonZeros()));
    for (int kp = 0; kp < p.outerSize(); ++kp) {
        for (SparseOp::InnerIterator itP(p, kp); itP; ++itP) {
            for (int kq = 0; kq < q.outerSize(); ++kq) {
                for (SparseOp::InnerIterator itQ(q, kq); itQ; ++itQ) {
                    trips.emplace_back(itP.row() * q.rows() + itQ.row(),
                                       itP.col() * q.cols() + itQ.col(),
                                       itP.value() * itQ.value());
                }
            }
        }
    }
    SparseOp out(p.rows() * q.rows(), p.cols() * q.cols());
    out.setFromTriplets(trips.begin(), trips.end());
    return out;
}

FockDirectSum::SparseOp FockDirectSum::liftLeft(const SparseOp& opA) const {
    const Eigen::Index dimA = static_cast<Eigen::Index>(1ull << modesA_);
    if (opA.rows() != dimA || opA.cols() != dimA) {
        throw std::invalid_argument("FockDirectSum::liftLeft: operator is " +
                                    std::to_string(opA.rows()) + "x" +
                                    std::to_string(opA.cols()) +
                                    ", expected " + std::to_string(dimA) +
                                    "x" + std::to_string(dimA));
    }
    // Joint index n(b) = i_A + 2^{M_A} i_B, so the B index is the outer
    // (slow) factor: X_A ⊗ 1 compiles to kron(Id_B, X_A).
    SparseOp idB(static_cast<Eigen::Index>(1ull << modesB_),
                 static_cast<Eigen::Index>(1ull << modesB_));
    idB.setIdentity();
    return kronSparse(idB, opA);
}

FockDirectSum::SparseOp FockDirectSum::liftRight(const SparseOp& opB,
                                                 bool oddOperator) const {
    const Eigen::Index dimB = static_cast<Eigen::Index>(1ull << modesB_);
    if (opB.rows() != dimB || opB.cols() != dimB) {
        throw std::invalid_argument("FockDirectSum::liftRight: operator is " +
                                    std::to_string(opB.rows()) + "x" +
                                    std::to_string(opB.cols()) +
                                    ", expected " + std::to_string(dimB) +
                                    "x" + std::to_string(dimB));
    }
    if (oddOperator) {
        // Koszul sign of the graded tensor product: the odd right factor
        // crosses every occupied A mode — kron(Y, (−1)^{N_A}).
        return kronSparse(opB, ExteriorAlgebra(modesA_).parityMatrix());
    }
    SparseOp idA(static_cast<Eigen::Index>(1ull << modesA_),
                 static_cast<Eigen::Index>(1ull << modesA_));
    idA.setIdentity();
    return kronSparse(opB, idA);
}

FockDirectSum::SparseOp FockDirectSum::gradedSwapMatrix() const {
    const std::uint64_t dimA = 1ull << modesA_;
    const std::uint64_t dimB = 1ull << modesB_;
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(dimA * dimB));
    for (std::uint64_t iA = 0; iA < dimA; ++iA) {
        const int pA = std::popcount(iA) % 2;
        for (std::uint64_t iB = 0; iB < dimB; ++iB) {
            const int pB = std::popcount(iB) % 2;
            const double sign = (pA == 1 && pB == 1) ? -1.0 : +1.0;
            trips.emplace_back(static_cast<Eigen::Index>(iB + dimB * iA),
                               static_cast<Eigen::Index>(iA + dimA * iB),
                               Complex(sign, 0.0));
        }
    }
    SparseOp op(static_cast<Eigen::Index>(dimA * dimB),
                static_cast<Eigen::Index>(dimA * dimB));
    op.setFromTriplets(trips.begin(), trips.end());
    return op;
}

Eigen::MatrixXcd FockDirectSum::assembleBlockOneParticle(
    const Eigen::MatrixXcd& blockA, const Eigen::MatrixXcd& blockB,
    const Eigen::MatrixXcd& coupling) {
    if (blockA.rows() != blockA.cols() || blockB.rows() != blockB.cols()) {
        throw std::invalid_argument(
            "FockDirectSum::assembleBlockOneParticle: diagonal blocks must "
            "be square");
    }
    if (coupling.rows() != blockA.rows() || coupling.cols() != blockB.rows()) {
        throw std::invalid_argument(
            "FockDirectSum::assembleBlockOneParticle: coupling is " +
            std::to_string(coupling.rows()) + "x" +
            std::to_string(coupling.cols()) + ", expected " +
            std::to_string(blockA.rows()) + "x" + std::to_string(blockB.rows()));
    }
    const Eigen::Index mA = blockA.rows();
    const Eigen::Index mB = blockB.rows();
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(mA + mB, mA + mB);
    L.topLeftCorner(mA, mA) = blockA;
    L.bottomRightCorner(mB, mB) = blockB;
    L.topRightCorner(mA, mB) = coupling;
    L.bottomLeftCorner(mB, mA) = coupling.adjoint();
    return L;
}

FockDirectSum::SparseOp FockDirectSum::dGammaBlock(
    const Eigen::MatrixXcd& blockA, const Eigen::MatrixXcd& blockB,
    const Eigen::MatrixXcd& coupling) const {
    if (static_cast<std::size_t>(blockA.rows()) != modesA_ ||
        static_cast<std::size_t>(blockB.rows()) != modesB_) {
        throw std::invalid_argument(
            "FockDirectSum::dGammaBlock: block sizes " +
            std::to_string(blockA.rows()) + ", " +
            std::to_string(blockB.rows()) + " do not match the bipartition " +
            std::to_string(modesA_) + " + " + std::to_string(modesB_));
    }
    return jointAlgebra().dGamma(
        assembleBlockOneParticle(blockA, blockB, coupling));
}

// ─── EdgeModeRegistry ─────────────────────────────────────────────────────

EdgeModeRegistry EdgeModeRegistry::fromSpacetime(
    const ::tessera::spacetime::Spacetime& spacetime,
    std::string lineageKey) {
    EdgeModeRegistry registry;
    const auto& edges = spacetime.getEdgeList();
    if (edges == nullptr) return registry;
    for (const auto* edge : edges->toVector()) {
        if (edge == nullptr) continue;
        const auto& source = edge->getSource();
        const auto& target = edge->getTarget();
        if (source == nullptr || target == nullptr) continue;
        // The stored source -> target direction is the mode's orientation, so
        // the sign against the canonical min -> max direction stays derivable
        // from the ids alone (canonicalOrientationSign).
        registry.addEdge(source->getId(), target->getId(), +1, lineageKey);
    }
    return registry;
}

std::uint64_t EdgeModeRegistry::addEdge(std::uint64_t vertexA,
                                        std::uint64_t vertexB,
                                        int orientationSign,
                                        std::string lineageKey) {
    if (vertexA == vertexB) {
        throw std::invalid_argument(
            "EdgeModeRegistry::addEdge: self-loop on vertex " +
            std::to_string(vertexA));
    }
    if (orientationSign != +1 && orientationSign != -1) {
        throw std::invalid_argument(
            "EdgeModeRegistry::addEdge: orientationSign must be +1 or -1, "
            "got " +
            std::to_string(orientationSign));
    }
    const std::uint64_t lo = std::min(vertexA, vertexB);
    const std::uint64_t hi = std::max(vertexA, vertexB);
    for (const EdgeModeRecord& r : records_) {
        if (std::min(r.vertexA, r.vertexB) == lo &&
            std::max(r.vertexA, r.vertexB) == hi) {
            throw std::invalid_argument(
                "EdgeModeRegistry::addEdge: edge {" + std::to_string(lo) +
                ", " + std::to_string(hi) +
                "} already indexes mode " + std::to_string(r.modeId) +
                " (one two-level factor per edge)");
        }
    }
    EdgeModeRecord rec;
    rec.vertexA = vertexA;
    rec.vertexB = vertexB;
    rec.orientationSign = orientationSign;
    rec.modeId = static_cast<std::uint64_t>(records_.size());
    rec.lineageKey = std::move(lineageKey);
    records_.push_back(std::move(rec));
    return records_.back().modeId;
}

void EdgeModeRegistry::validateModeId(std::uint64_t modeId) const {
    if (modeId >= records_.size()) {
        throw std::invalid_argument("EdgeModeRegistry: unknown modeId " +
                                    std::to_string(modeId));
    }
}

const EdgeModeRecord& EdgeModeRegistry::record(std::uint64_t modeId) const {
    validateModeId(modeId);
    return records_[static_cast<std::size_t>(modeId)];
}

void EdgeModeRegistry::reverseStoredDirection(std::uint64_t modeId) {
    validateModeId(modeId);
    EdgeModeRecord& r = records_[static_cast<std::size_t>(modeId)];
    std::swap(r.vertexA, r.vertexB);
    r.orientationSign = -r.orientationSign;
}

void EdgeModeRegistry::flipOrientation(std::uint64_t modeId) {
    validateModeId(modeId);
    EdgeModeRecord& r = records_[static_cast<std::size_t>(modeId)];
    r.orientationSign = -r.orientationSign;
}

int EdgeModeRegistry::canonicalOrientationSign(std::uint64_t modeId) const {
    validateModeId(modeId);
    const EdgeModeRecord& r = records_[static_cast<std::size_t>(modeId)];
    return r.vertexA < r.vertexB ? r.orientationSign : -r.orientationSign;
}

std::vector<std::uint64_t> EdgeModeRegistry::canonicalModeOrder() const {
    std::vector<std::uint64_t> order(records_.size());
    for (std::size_t i = 0; i < records_.size(); ++i) {
        order[i] = records_[i].modeId;
    }
    std::sort(order.begin(), order.end(),
              [this](std::uint64_t a, std::uint64_t b) {
                  const EdgeModeRecord& ra = records_[static_cast<std::size_t>(a)];
                  const EdgeModeRecord& rb = records_[static_cast<std::size_t>(b)];
                  if (ra.lineageKey != rb.lineageKey) {
                      return ra.lineageKey < rb.lineageKey;
                  }
                  const std::uint64_t loA = std::min(ra.vertexA, ra.vertexB);
                  const std::uint64_t loB = std::min(rb.vertexA, rb.vertexB);
                  if (loA != loB) return loA < loB;
                  return std::max(ra.vertexA, ra.vertexB) <
                         std::max(rb.vertexA, rb.vertexB);
              });
    return order;
}

std::vector<std::size_t> EdgeModeRegistry::compilationPositions() const {
    const std::vector<std::uint64_t> order = canonicalModeOrder();
    std::vector<std::size_t> positions(records_.size(), 0);
    for (std::size_t pos = 0; pos < order.size(); ++pos) {
        positions[static_cast<std::size_t>(order[pos])] = pos;
    }
    return positions;
}

EdgeModeRegistry EdgeModeRegistry::relabeled(
    const std::unordered_map<std::uint64_t, std::uint64_t>& vertexMap) const {
    // The map must cover every used vertex and stay injective on them —
    // a merging relabeling would silently identify distinct edges.
    std::unordered_set<std::uint64_t> used;
    for (const EdgeModeRecord& r : records_) {
        for (std::uint64_t v : {r.vertexA, r.vertexB}) {
            if (vertexMap.find(v) == vertexMap.end()) {
                throw std::invalid_argument(
                    "EdgeModeRegistry::relabeled: vertex " +
                    std::to_string(v) + " missing from the relabeling map");
            }
            used.insert(v);
        }
    }
    std::unordered_set<std::uint64_t> images;
    for (std::uint64_t v : used) {
        if (!images.insert(vertexMap.at(v)).second) {
            throw std::invalid_argument(
                "EdgeModeRegistry::relabeled: relabeling map is not "
                "injective on the used vertices (image " +
                std::to_string(vertexMap.at(v)) + " repeats)");
        }
    }
    EdgeModeRegistry out;
    out.records_ = records_;
    for (EdgeModeRecord& r : out.records_) {
        r.vertexA = vertexMap.at(r.vertexA);
        r.vertexB = vertexMap.at(r.vertexB);
    }
    return out;
}

std::vector<std::size_t> EdgeModeRegistry::orderPermutation(
    const EdgeModeRegistry& before, const EdgeModeRegistry& after) {
    if (before.modeCount() != after.modeCount()) {
        throw std::invalid_argument(
            "EdgeModeRegistry::orderPermutation: registries hold " +
            std::to_string(before.modeCount()) + " vs " +
            std::to_string(after.modeCount()) + " modes");
    }
    const std::vector<std::size_t> posBefore = before.compilationPositions();
    const std::vector<std::size_t> posAfter = after.compilationPositions();
    // records_ are indexed by modeId in both registries (modeIds are dense
    // registration indices), so matching by modeId is positional here.
    std::vector<std::size_t> perm(before.modeCount(), 0);
    for (std::size_t modeId = 0; modeId < before.modeCount(); ++modeId) {
        perm[posBefore[modeId]] = posAfter[modeId];
    }
    return perm;
}

}  // namespace tessera::quantum
