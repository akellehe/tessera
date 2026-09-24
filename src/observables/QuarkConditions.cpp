// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/QuarkConditions.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tessera::observables {

namespace {

void requireNumber(int number, const char *where) {
    if (number < 1 || number > QuarkConditions::kConditionCount)
        throw std::invalid_argument(
            std::string(where) + ": the quark conditions are numbered 1 to " +
            std::to_string(QuarkConditions::kConditionCount) + "; got " +
            std::to_string(number));
}

}  // namespace

std::vector<std::string> QuarkConditions::conditionNames() {
    return {"persistent-cluster", "color-spin-fiber", "anchor-atlas",
            "odd-occupation",     "color-transport",  "lineage",
            "fingerprint"};
}

std::string QuarkConditions::statement(int number) {
    requireNumber(number, "QuarkConditions::statement");
    switch (number) {
        case 1:
            return "Q is a persistent cluster certified as in Section 5, "
                   "however its support was proposed.";
        case 2:
            return "Its color-spin fiber is E-bar_Q (x) C^3: a per-sheet base "
                   "band (the coexact triplet at flat connection, or the "
                   "j = 1/2 doublet of an odd-monopole support) on a "
                   "three-sheeted support.";
        case 3:
            return "Its anchor atlas (projective profile, invariant "
                   "coordinates, and determinant transitions) is stable.";
        case 4:
            return "It is a single occupied mode of that fiber: its "
                   "occupation parity is odd.";
        case 5:
            return "Its color transport S in GL(3, C) stays full rank, and its "
                   "base transport has numerically bounded leakage, over its "
                   "lifetime.";
        case 6:
            return "Its oriented cluster lineage has intersection number "
                   "N_Q = +1 with a separating cut, and, where the determinant "
                   "family closes interferometrically, relative winding "
                   "nu = +1.";
        default:
            return "Its total spectral fingerprint is stable under refinement "
                   "and vertex relabeling.";
    }
}

std::vector<std::string> QuarkConditions::requiredEvidence(int number) {
    requireNumber(number, "QuarkConditions::requiredEvidence");
    switch (number) {
        case 1:
            return {"persistent-support",   "localized-projector-rank",
                    "contour-separation",   "successor-overlap",
                    "multi-frame-lifetime", "external-leakage"};
        case 2:
            return {"three-sheeted-support", "sheet-isomorphism",
                    "protected-base-band", "base-band-sector", "fibre-lift"};
        case 3:
            return {"anchor-profile-nonzero", "anchor-covariance",
                    "anchor-transitions", "anchor-stable-across-frames"};
        case 4:
            return {"odd-occupation-parity"};
        case 5:
            return {"color-transport-full-rank", "base-transport-leakage",
                    "transport-over-lifetime"};
        case 6:
            return {"lineage-intersection"};
        default:
            return {"refinement-stability", "relabeling-stability"};
    }
}

QuarkConditionRead QuarkConditions::evaluateCondition(
    int number, const std::vector<QuarkConditionEvidence>& evidence) {
    requireNumber(number, "QuarkConditions::evaluateCondition");
    QuarkConditionRead read;
    read.number = number;
    read.name = conditionNames()[static_cast<std::size_t>(number - 1)];
    read.statement = statement(number);
    for (const auto& item : evidence) {
        if (item.name.empty())
            throw std::invalid_argument(
                "QuarkConditions::evaluateCondition: every evidence item "
                "names the certificate it reports");
        read.evidence.push_back(item);
        if (item.held.has_value() && !*item.held)
            read.failing.push_back(item.name);
    }
    for (const auto& required : requiredEvidence(number)) {
        const auto found = std::find_if(
            evidence.begin(), evidence.end(),
            [&](const QuarkConditionEvidence& item) {
                return item.name == required && item.held.has_value();
            });
        if (found == evidence.end()) {
            read.missing.push_back(required);
            const bool listed = std::any_of(
                evidence.begin(), evidence.end(),
                [&](const QuarkConditionEvidence& item) {
                    return item.name == required;
                });
            if (!listed)
                read.evidence.push_back(
                    {required, std::nullopt, "not supplied by the caller"});
        }
    }
    if (!read.failing.empty())
        read.status = QuarkConditionStatus::Failed;
    else if (!read.missing.empty())
        read.status = QuarkConditionStatus::NotEvaluable;
    else
        read.status = QuarkConditionStatus::Passed;
    return read;
}

QuarkVerdict QuarkConditions::evaluate(
    const std::array<std::vector<QuarkConditionEvidence>, 7>& evidence) {
    QuarkVerdict verdict;
    verdict.certified = true;
    for (int number = 1; number <= kConditionCount; ++number) {
        auto read = evaluateCondition(
            number, evidence[static_cast<std::size_t>(number - 1)]);
        if (read.status == QuarkConditionStatus::Failed)
            verdict.failed.push_back(read.name);
        if (read.status == QuarkConditionStatus::NotEvaluable)
            verdict.notEvaluable.push_back(read.name);
        if (read.status != QuarkConditionStatus::Passed)
            verdict.certified = false;
        verdict.conditions.push_back(std::move(read));
    }
    return verdict;
}

}  // namespace tessera::observables
