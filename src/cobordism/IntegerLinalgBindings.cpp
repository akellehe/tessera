// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: integer linear algebra and the
// characteristic numbers. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_integer_linalg(py::module_ &m) {
  // Exact integer / GF(2) / inertia primitives. Matrices are passed flat
  // row-major with explicit dims.
  py::class_<SmithNormalForm>(m, "SmithNormalForm")
      .def_readonly("rank", &SmithNormalForm::rank)
      .def_readonly("invariant_factors", &SmithNormalForm::invariantFactors);
  m.def("smith_normal_form", &smithNormalForm, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"), "Smith Normal Form (rank + invariant factors) over Z.");
  m.def("integer_rank", &integerRank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over Q of an integer matrix.");
  m.def("gf2_rank", &gf2Rank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over GF(2) of a 0/1 matrix.");
  m.def("gf2_nullspace", &gf2Nullspace, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"),
        "Basis of the GF(2) kernel of a 0/1 matrix, as a list of nullity "
        "length-cols vectors (each x with matrix·x == 0 mod 2; independent over "
        "GF(2); nullity == cols - gf2_rank). The cocycles Z1 = ker(d2^T mod 2).");
  m.def("gf2_span", &gf2Span, py::arg("basis"), py::arg("cols"),
        "All 2^k GF(2) combinations of a basis of k length-cols vectors (first "
        "is the zero vector). For a gf2_nullspace basis, the flat Z2 connections.");
  m.def("integer_nullspace", &integerNullspace, py::arg("matrix"),
        py::arg("rows"), py::arg("cols"),
        "Basis of the rational kernel of an integer matrix as INTEGER vectors "
        "(exact Gauss-Jordan over Q; coprime entries; nullity == cols - rank). "
        "Raises OverflowError when the exact elimination would overflow 64-bit "
        "intermediates -- never rounded.");

  py::class_<Inertia>(m, "Inertia")
      .def_readonly("n_pos", &Inertia::nPos)
      .def_readonly("n_neg", &Inertia::nNeg)
      .def_readonly("n_zero", &Inertia::nZero)
      .def("signature", &Inertia::signature);
  m.def("symmetric_inertia", &symmetricInertia, py::arg("matrix"), py::arg("n"),
        py::arg("tol") = 1e-9,
        "Inertia (#pos,#neg,#zero eigenvalues) of a symmetric integer matrix.");

  // ----- Characteristic numbers -----
  // Scalar invariants are Observables; families come from CharacteristicNumbers.
  py::class_<EulerCharacteristic, std::shared_ptr<EulerCharacteristic>>(
      m, "EulerCharacteristic",
      "Observable: Euler characteristic chi = sum_k (-1)^k |C_k|.")
      .def(py::init<>())
      .def("compute", &EulerCharacteristic::compute, py::arg("spacetime"));
  // Qualified, since tessera::spacetime::Signature (metric signature) is also
  // in scope via the using-directives.
  py::class_<cobordism::Signature, std::shared_ptr<cobordism::Signature>>(
      m, "Signature",
      "Observable: signature b+ - b- of the H_2 intersection form (closed "
      "oriented 4-manifold; 0 if n != 4 or b2 == 0).")
      .def(py::init<>())
      .def("compute", &cobordism::Signature::compute, py::arg("spacetime"));

  py::class_<CharacteristicNumbers>(m, "CharacteristicNumbers",
      "Topological invariants of a closed PL n-manifold: Euler characteristic, "
      "signature (4-manifolds), Stiefel-Whitney numbers (pending), and "
      "Pontryagin numbers (4-manifolds: p1 = 3*signature).")
      .def_readonly("euler", &CharacteristicNumbers::euler,
                    "Euler characteristic (alternating count of cells).")
      .def_readonly("signature", &CharacteristicNumbers::signature,
                    "Signature of the intersection form; None unless an "
                    "orientable 4-manifold.")
      .def_readonly("stiefel_whitney_numbers",
                    &CharacteristicNumbers::stiefelWhitneyNumbers,
                    "Mod-2 Stiefel-Whitney numbers, keyed by monomial (pending; "
                    "currently empty).")
      .def_readonly("pontryagin_numbers",
                    &CharacteristicNumbers::pontryaginNumbers,
                    "Pontryagin numbers (4-manifolds: {'p1': 3*signature}).")
      .def_static("of", &CharacteristicNumbers::of, py::arg("spacetime"),
                  py::arg("oriented") = true,
                  "Compute the characteristic numbers of the given manifold.");
}
