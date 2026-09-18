// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/Record.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tessera::observables {

void Record::splitComplex(Map &into, const std::string &name,
                          std::complex<double> value) {
  into[name + "_re"] = value.real();
  into[name + "_im"] = value.imag();
}

void Record::splitComplex(Map &into, const std::string &name,
                          const std::vector<std::complex<double>> &values) {
  List re;
  List im;
  re.reserve(values.size());
  im.reserve(values.size());
  for (const auto &z : values) {
    re.emplace_back(z.real());
    im.emplace_back(z.imag());
  }
  into[name + "_re"] = std::move(re);
  into[name + "_im"] = std::move(im);
}

namespace {

[[noreturn]] void throwKeyMismatch(const Record::Map &a, const Record::Map &b) {
  std::ostringstream oss;
  oss << "record keys differ: {";
  bool first = true;
  for (const auto &kv : a) {
    oss << (first ? "" : ", ") << kv.first;
    first = false;
  }
  oss << "} vs {";
  first = true;
  for (const auto &kv : b) {
    oss << (first ? "" : ", ") << kv.first;
    first = false;
  }
  oss << "}";
  throw std::invalid_argument(oss.str());
}

// Integer view of a scalar: bools are 0/1, doubles truncate toward zero.
std::int64_t asIntegerValue(const Record &r) {
  switch (r.type()) {
    case Record::Type::Bool:
      return r.asBool() ? 1 : 0;
    case Record::Type::Int:
      return r.asInt();
    case Record::Type::Double:
      return static_cast<std::int64_t>(r.asDouble());
    default:
      return 0;
  }
}

}  // namespace

double Record::reportDelta(const Record &a, const Record &b) {
  const double inf = std::numeric_limits<double>::infinity();

  // Maps must carry identical keys; the delta is the maximum over them.
  if (a.type_ == Type::Map || b.type_ == Type::Map) {
    if (!(a.type_ == Type::Map && b.type_ == Type::Map)) {
      return inf;
    }
    if (a.map_.size() != b.map_.size()) {
      throwKeyMismatch(a.map_, b.map_);
    }
    double worst = 0.0;
    for (const auto &kv : a.map_) {
      auto it = b.map_.find(kv.first);
      if (it == b.map_.end()) {
        throwKeyMismatch(a.map_, b.map_);
      }
      worst = std::max(worst, reportDelta(kv.second, it->second));
    }
    return worst;
  }

  // Strings and nulls must be equal; any difference reports infinity.
  if (a.type_ == Type::String || b.type_ == Type::String) {
    if (a.type_ == Type::String && b.type_ == Type::String &&
        a.string_ == b.string_) {
      return 0.0;
    }
    return inf;
  }
  if (a.type_ == Type::Null || b.type_ == Type::Null) {
    return (a.type_ == Type::Null && b.type_ == Type::Null) ? 0.0 : inf;
  }

  // Lists must match in length; the delta is the maximum over elements.
  if (a.type_ == Type::List || b.type_ == Type::List) {
    if (!(a.type_ == Type::List && b.type_ == Type::List)) {
      return inf;
    }
    if (a.list_.size() != b.list_.size()) {
      return inf;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < a.list_.size(); ++i) {
      worst = std::max(worst, reportDelta(a.list_[i], b.list_[i]));
    }
    return worst;
  }

  // Bools compare as 0/1, checked ahead of the numeric branch.
  if (a.type_ == Type::Bool || b.type_ == Type::Bool) {
    return std::fabs(
        static_cast<double>(asIntegerValue(a) - asIntegerValue(b)));
  }

  // Numbers: two NaNs agree (delta 0); a NaN against a number is infinite.
  const double x = (a.type_ == Type::Int) ? static_cast<double>(a.int_)
                                          : a.double_;
  const double y = (b.type_ == Type::Int) ? static_cast<double>(b.int_)
                                          : b.double_;
  if (std::isnan(x) || std::isnan(y)) {
    return (std::isnan(x) && std::isnan(y)) ? 0.0 : inf;
  }
  return std::fabs(x - y);
}

}  // namespace tessera::observables
