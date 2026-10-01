// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>

#include "numerics/TruncatedSeries.h"

namespace py = pybind11;
using namespace tessera::numerics;

void register_numerics(py::module_ m) {
  using Complex = TruncatedSeries::Complex;

  py::class_<TruncatedSeries>(m, "TruncatedSeries",
      R"doc(The Taylor coefficients c_0, ..., c_p of a function
f(t) = sum_k c_k t^k + O(t^(p+1)) of one parameter t, to a declared order p
between 0 and 10.

Evaluating a formula on series instead of numbers propagates the Taylor
coefficients of its arguments to those of its value exactly: every operation
computes the coefficients of its result by a finite recurrence that is an
identity between power series, with no difference quotient and no contour
integral. Two series enter a binary operation only when they have the same
order; a number enters as the constant series of that value.

``TruncatedSeries(order)`` is the zero series of that order and
``TruncatedSeries(coefficients)`` the series with the listed coefficients;
both raise ``ValueError`` for an order outside 0 to 10.)doc")
      .def(py::init<int>(), py::arg("order"),
           "The zero series of the given order (0 to 10).")
      .def(py::init<std::vector<Complex>>(), py::arg("coefficients"),
           "The series with coefficients c_0, ..., c_p; the order is their "
           "number less one.")
      .def_static("constant", &TruncatedSeries::constant, py::arg("order"),
                  py::arg("value"),
                  "The constant series of the given value and order.")
      .def_static("variable", &TruncatedSeries::variable, py::arg("order"),
                  py::arg("point"), py::arg("direction") = Complex(1.0, 0.0),
                  "The series point + direction * t: an independent variable "
                  "at `point` moving along `direction`.")
      .def_property_readonly("order", &TruncatedSeries::order,
                             "The declared order p.")
      .def_property_readonly(
          "coefficients",
          [](const TruncatedSeries &f) { return f.coefficients(); },
          "The coefficients c_0, ..., c_p.")
      .def("coefficient", &TruncatedSeries::coefficient, py::arg("k"),
           "The coefficient c_k; IndexError outside 0 to the order.")
      .def("evaluate", &TruncatedSeries::evaluate, py::arg("t"),
           "The value sum_k c_k t^k of the truncated series at t.")
      .def("truncated", &TruncatedSeries::truncated, py::arg("order"),
           "The same series with the coefficients above `order` left out.")
      .def("derivative", &TruncatedSeries::derivative,
           "The derivative, of order p - 1: coefficient k is (k+1) c_(k+1). "
           "ValueError for a series of order 0.")
      .def("integral", &TruncatedSeries::integral,
           py::arg("constant") = Complex(0.0, 0.0),
           "The integral constant + int_0^t f, of order p + 1 (order 10 for "
           "a series of order 10): coefficient k >= 1 is c_(k-1) / k.")
      .def("__neg__", [](const TruncatedSeries &f) { return -f; },
           py::is_operator())
      .def("__add__",
           [](const TruncatedSeries &f, const TruncatedSeries &g) {
             return f + g;
           },
           py::is_operator())
      .def("__add__",
           [](const TruncatedSeries &f, Complex value) { return f + value; },
           py::is_operator())
      .def("__radd__",
           [](const TruncatedSeries &f, Complex value) { return value + f; },
           py::is_operator())
      .def("__sub__",
           [](const TruncatedSeries &f, const TruncatedSeries &g) {
             return f - g;
           },
           py::is_operator())
      .def("__sub__",
           [](const TruncatedSeries &f, Complex value) { return f - value; },
           py::is_operator())
      .def("__rsub__",
           [](const TruncatedSeries &f, Complex value) { return value - f; },
           py::is_operator())
      .def("__mul__",
           [](const TruncatedSeries &f, const TruncatedSeries &g) {
             return f * g;
           },
           py::is_operator(),
           "The Cauchy product: (fg)_k = sum_{j=0..k} f_j g_(k-j).")
      .def("__mul__",
           [](const TruncatedSeries &f, Complex value) { return f * value; },
           py::is_operator())
      .def("__rmul__",
           [](const TruncatedSeries &f, Complex value) { return value * f; },
           py::is_operator())
      .def("__truediv__",
           [](const TruncatedSeries &f, const TruncatedSeries &g) {
             return f / g;
           },
           py::is_operator(),
           "The quotient h = f / g by g_0 h_k = f_k - sum_{j=1..k} g_j "
           "h_(k-j); ValueError when g_0 is exactly zero.")
      .def("__truediv__",
           [](const TruncatedSeries &f, Complex value) { return f / value; },
           py::is_operator())
      .def("__rtruediv__",
           [](const TruncatedSeries &f, Complex value) { return value / f; },
           py::is_operator())
      .def("__pow__",
           [](const TruncatedSeries &f, int exponent) {
             return tessera::numerics::pow(f, exponent);
           },
           py::is_operator(),
           "The integer power; see `pow`.")
      .def("__repr__", [](const TruncatedSeries &f) {
        std::ostringstream out;
        out << "TruncatedSeries(order=" << f.order() << ", coefficients=[";
        for (std::size_t k = 0; k < f.coefficients().size(); ++k) {
          const Complex c = f.coefficients()[k];
          out << (k == 0 ? "" : ", ") << "(" << c.real()
              << (c.imag() < 0.0 ? "" : "+") << c.imag() << "j)";
        }
        out << "])";
        return out.str();
      });
  m.attr("MAXIMUM_ORDER") = TruncatedSeries::kMaximumOrder;

  m.def("pow", &tessera::numerics::pow, py::arg("f"), py::arg("exponent"),
        R"doc(The integer power f^n. For n >= 0 it is formed by repeated squaring, by
Cauchy products alone, and f^0 is the constant series 1. For n < 0 it is
(1/f)^|n|; ValueError when f_0 is exactly zero.)doc");
  m.def("sqrt", &tessera::numerics::sqrt, py::arg("f"),
        py::arg("branch") = py::none(),
        R"doc(The square root g of f by 2 g_0 g_k = f_k - sum_{j=1..k-1} g_j g_(k-j).

`branch` is the value of g_0, the root of f_0 the caller continues from; the
principal root of f_0 when it is None. ValueError when f_0, or the given
branch, is exactly zero.)doc");
  m.def("exp", &tessera::numerics::exp, py::arg("f"),
        "The exponential g of f by k g_k = sum_{j=1..k} j f_j g_(k-j), "
        "g_0 = exp(f_0).");
  m.def("log", &tessera::numerics::log, py::arg("f"),
        py::arg("branch") = py::none(),
        R"doc(The logarithm g of f by k f_0 g_k = k f_k - sum_{j=1..k-1} j g_j f_(k-j).

`branch` is the value of g_0, the logarithm of f_0 the caller continues from;
the principal logarithm of f_0 when it is None. ValueError when f_0 is exactly
zero.)doc");
  m.def("sinCos", &tessera::numerics::sinCos, py::arg("f"),
        R"doc(The pair (sin f, cos f) by the coupled recurrence
k s_k = sum_{j=1..k} j f_j c_(k-j), k c_k = -sum_{j=1..k} j f_j s_(k-j).)doc");
  m.def("sin", &tessera::numerics::sin, py::arg("f"),
        "The sine of f; see `sinCos`.");
  m.def("cos", &tessera::numerics::cos, py::arg("f"),
        "The cosine of f; see `sinCos`.");
  m.def("atan2", &tessera::numerics::atan2, py::arg("y"), py::arg("x"),
        py::arg("branch") = py::none(),
        R"doc(The angle theta = atan2(y, x) of the point (x, y), from
theta = (log(x + i y) - log(x - i y)) / (2i): its coefficients of order k >= 1
are those of the two logarithms, each by the recurrence of `log`.

`branch` is the order-0 value the caller continues from. When it is None the
order-0 value is math.atan2 of the real parts for real x_0 and y_0, and the
same formula with principal logarithms otherwise. ValueError when x_0 + i y_0
or x_0 - i y_0 is exactly zero (for real arguments: at the origin).)doc");
  m.def("acos", &tessera::numerics::acos, py::arg("f"),
        py::arg("branch") = py::none(),
        R"doc(The inverse cosine theta of f through its derivative theta' = -f' / s, with
s = sin(theta) the square root of 1 - f^2 whose order-0 value is sin(theta_0).

`branch` is theta_0, the angle the caller continues from, with
cos(theta_0) = f_0; the principal inverse cosine of f_0 when it is None.
ValueError when f_0 is exactly +1 or -1.)doc");
  m.def("composePolynomial", &tessera::numerics::composePolynomial,
        py::arg("polynomial"), py::arg("f"),
        R"doc(The composition sum_j a_j f^j of the polynomial with coefficients
(a_0, ..., a_n) with the series f, by Horner's rule in series arithmetic.)doc");

  py::class_<TruncatedSeriesMatrix>(m, "TruncatedSeriesMatrix",
      R"doc(A dense matrix of series: the coefficient matrices A_0, ..., A_p of
A(t) = sum_k A_k t^k + O(t^(p+1)), all of one shape, to a declared order p
between 0 and 10.

``TruncatedSeriesMatrix(coefficients)`` takes the list of coefficient
matrices and ``TruncatedSeriesMatrix(order, rows, cols)`` is the zero matrix
of that shape. ``A @ B`` is the matrix product.)doc")
      .def(py::init<std::vector<Eigen::MatrixXcd>>(), py::arg("coefficients"),
           "The matrix series with coefficient matrices A_0, ..., A_p.")
      .def(py::init<int, Eigen::Index, Eigen::Index>(), py::arg("order"),
           py::arg("rows"), py::arg("cols"),
           "The zero matrix of the given order and shape.")
      .def_static("identity", &TruncatedSeriesMatrix::identity,
                  py::arg("order"), py::arg("size"),
                  "The constant series of the identity matrix.")
      .def_property_readonly("order", &TruncatedSeriesMatrix::order,
                             "The declared order p.")
      .def_property_readonly("rows", &TruncatedSeriesMatrix::rows,
                             "The number of rows.")
      .def_property_readonly("cols", &TruncatedSeriesMatrix::cols,
                             "The number of columns.")
      .def_property_readonly(
          "coefficients",
          [](const TruncatedSeriesMatrix &A) { return A.coefficients(); },
          "The coefficient matrices A_0, ..., A_p.")
      .def("entry", &TruncatedSeriesMatrix::entry, py::arg("row"),
           py::arg("col"), "The series of entry (row, col).")
      .def("setEntry", &TruncatedSeriesMatrix::setEntry, py::arg("row"),
           py::arg("col"), py::arg("series"),
           "Replaces entry (row, col) by a series of the order of the matrix.")
      .def("evaluate", &TruncatedSeriesMatrix::evaluate, py::arg("t"),
           "The matrix sum_k A_k t^k at the parameter value t.")
      .def("__add__",
           [](const TruncatedSeriesMatrix &A, const TruncatedSeriesMatrix &B) {
             return A + B;
           },
           py::is_operator())
      .def("__sub__",
           [](const TruncatedSeriesMatrix &A, const TruncatedSeriesMatrix &B) {
             return A - B;
           },
           py::is_operator())
      .def("__matmul__",
           [](const TruncatedSeriesMatrix &A, const TruncatedSeriesMatrix &B) {
             return A * B;
           },
           py::is_operator(),
           "The matrix product: (AB)_k = sum_{j=0..k} A_j B_(k-j).")
      .def("inverse", &TruncatedSeriesMatrix::inverse,
           R"doc(The inverse B of a square matrix series: A_0 B_0 = I and
A_0 B_k = -sum_{j=1..k} A_j B_(k-j), every order solved with the one
factorization of A_0. ValueError when the elimination of A_0 meets a pivot
that is exactly zero.)doc")
      .def("determinant", &TruncatedSeriesMatrix::determinant,
           R"doc(The determinant of a square matrix series, as a series, by Bird's
division-free algorithm in series arithmetic: with mu(X) the upper triangular
matrix that keeps the entries of X above the diagonal and has the diagonal
entries -sum_{j>i} X_jj, iterate X <- mu(X) A from X = A, n - 1 times; the
determinant is (-1)^(n-1) times entry (0, 0) of the result. Sums and products
only, so it has a value for every matrix series, a singular A_0 included.)doc")
      .def("rieszProjector", &TruncatedSeriesMatrix::rieszProjector,
           py::arg("inGroup"),
           R"doc(The series P(t) of the Riesz (spectral) projector of a group of eigenvalues
of the square matrix series A(t).

`inGroup` is a callable that receives an eigenvalue of A_0 and returns True
when it belongs to the group. P_0 is the spectral projector of A_0 for the
group, and the higher coefficients follow order by order from P^2 = P and
A P = P A with no contour integral: P^2 = P fixes the two diagonal blocks of
P_k in the basis adapted to P_0, and [A, P] = 0 fixes the two blocks between
the group and its complement as the solutions of Sylvester equations with the
reordered Schur form of A_0, solved by back substitution.

ValueError when an eigenvalue of the group is exactly equal to an eigenvalue
outside it; eigenvalues that differ by any nonzero amount are divided by as
they stand.)doc");

  py::class_<SeriesReversion>(m, "SeriesReversion",
      R"doc(The result of `revert`: the path s(t) = sum_{k=1..p} s_k t^k with
F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^(p+1)).)doc")
      .def_readonly("residual", &SeriesReversion::residual,
                    "The residual F(x_0) at the base point.")
      .def_readonly("coefficients", &SeriesReversion::coefficients,
                    "The coefficients s_0 = 0, s_1, ..., s_p of the path, one "
                    "vector per order.")
      .def("evaluate", &SeriesReversion::evaluate, py::arg("t"),
           "The path s(t) at the parameter value t.")
      .def("step", &SeriesReversion::step,
           "The order-p step s(1) = s_1 + ... + s_p.");

  m.def("jacobian", &tessera::numerics::jacobian, py::arg("map"),
        py::arg("point"),
        R"doc(The Jacobian F'(x_0) of a map evaluated in series arithmetic: column i is
the coefficient of t of F(x_0 + t e_i), read from one evaluation of `map` on
series of order 1 per column.

`map` receives a list of TruncatedSeries, one per unknown, and returns a list
of TruncatedSeries, one per equation, of the order of its arguments.)doc");
  m.def("revert", &tessera::numerics::revert, py::arg("map"), py::arg("point"),
        py::arg("order"), py::arg("solve"),
        R"doc(The reversion of the equations F(x) = 0 about the point x_0 to order p: the
series s(t) = sum_{k=1..p} s_k t^k with
F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^(p+1)).

With J = F'(x_0), the path solves J s_1 = -F(x_0) and J s_k = -N_k for
k >= 2, where N_k is the coefficient of t^k of
F(x_0 + s_1 t + ... + s_(k-1) t^(k-1)), read from one evaluation of `map` on
series of order k. Each linear system is handed to `solve`, which receives
the right-hand side and returns s_k, so the caller decides how a
rank-deficient J is treated. The order-p step is `SeriesReversion.step()`;
at order 1 it is Newton's step.

`map` receives a list of TruncatedSeries, one per unknown, and returns a list
of TruncatedSeries, one per equation, of the order of its arguments.)doc");
}
