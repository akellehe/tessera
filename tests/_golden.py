# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Tolerance-based comparison against saved golden dumps.

A dump records floats as hex strings (``float.hex()``), which pins every bit of
the mantissa. Comparing those strings with ``==`` asserts bit-identical
arithmetic, and that does not hold across toolchains: a different compiler
reassociates and contracts differently, so a last-ulp difference fails a test
that is otherwise measuring the right quantity. Pinning a compiler to keep that
guarantee costs more than the guarantee is worth.

So compare the DECODED floats within a tolerance instead. The hex encoding stays
in the dump files -- it round-trips exactly and stays readable in a diff -- but
the assertion is numeric.

Scope. This is for dumps saved on some other machine. An operation compared
against its own inverse inside one process (a Pachner rollback, a cone-out then
cone-in) is a different claim: exact invertibility is an algorithmic property,
not a compiler artifact, and those comparisons should stay exact.

Choosing a tolerance. Measured against the block-residual dump, the quantities
fall into tiers:

  * most reads -- edges, transfer, r_u, trace, block residuals -- land within
    1e-13 relative. That is compiler noise.
  * ascent directions reach 7e-10 relative. They come off an iterative
    relaxation, which amplifies a last-ulp difference in its input into a
    visibly different step.

`RELATIVE` is set an order of magnitude above the worst of those. A real change
to the arithmetic moves these values far further than that, so the test still
does its job.

Some quantities are not reproducible at any tolerance and should not be
compared entry by entry at all -- the imaginary parts of the holomorphic
periods, for instance, are pure cancellation noise around zero, and vary by a
factor of three between builds while the monodromy they induce still agrees to
1e-15. Assert the invariant those quantities feed, not the quantities.
"""

import math

import numpy as np

# Relative agreement required of a saved dump, against the largest magnitude in
# the structure being compared. See the tiers in the module docstring.
RELATIVE = 1e-8

# Absolute floor, for entries that sit at zero by cancellation.
ABSOLUTE = 1e-12


def _is_hex_float(value):
    if not isinstance(value, str):
        return False
    try:
        float.fromhex(value)
    except (ValueError, OverflowError):
        return False
    return True


def _decode(value):
    """The float a leaf denotes, or None when the leaf is not a float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return value
    if _is_hex_float(value):
        return float.fromhex(value)
    return None


def _scale_of(structure):
    """Largest float magnitude anywhere in a dump structure."""
    largest = 0.0
    stack = [structure]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        else:
            value = _decode(node)
            if value is not None and math.isfinite(value):
                largest = max(largest, abs(value))
    return largest


def golden_diff(got, expected, rtol=RELATIVE, atol=ABSOLUTE, path="",
                scale=None) -> list:
    """Compare nested dump structures, returning a list of difference strings.

    Walks lists and dicts. Floats are compared against `atol + rtol * scale`,
    where `scale` is the largest magnitude anywhere in the structure. Scaling by
    the structure rather than by each entry matters: a dump mixes quantities of
    very different size, and an imaginary part near zero sitting next to a real
    part near one is small through cancellation, not through precision --
    judging it against its own magnitude would demand accuracy the arithmetic
    never had.

    Everything that is not a float -- ints, keys, labels -- must still match
    exactly, so a changed cell count or a reordered edge is still a failure.
    """
    here = path or "<root>"
    if scale is None:
        scale = max(_scale_of(expected), 1.0)
    tolerance = atol + rtol * scale

    g, e = _decode(got), _decode(expected)
    if g is not None and e is not None:
        if math.isnan(g) and math.isnan(e):
            return []
        if not (abs(g - e) <= tolerance):
            return [f"{here}: {g!r} != {e!r} "
                    f"(|diff| {abs(g - e):.3g} > {tolerance:.3g})"]
        return []

    if isinstance(got, dict) and isinstance(expected, dict):
        if got.keys() != expected.keys():
            return [f"{here}: keys {sorted(got)} != {sorted(expected)}"]
        out = []
        for key in got:
            out += golden_diff(got[key], expected[key], rtol, atol,
                               f"{here}.{key}", scale)
        return out

    if isinstance(got, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(got) != len(expected):
            return [f"{here}: length {len(got)} != {len(expected)}"]
        out = []
        for i, (a, b) in enumerate(zip(got, expected)):
            out += golden_diff(a, b, rtol, atol, f"{here}[{i}]", scale)
        return out

    if got != expected:
        return [f"{here}: {got!r} != {expected!r}"]
    return []


def assert_golden(got, expected, what="value", rtol=RELATIVE, atol=ABSOLUTE):
    """Assert a dump structure matches a saved one within tolerance."""
    diffs = golden_diff(got, expected, rtol, atol)
    if diffs:
        shown = "\n  ".join(diffs[:12])
        more = "" if len(diffs) <= 12 else f"\n  ... and {len(diffs) - 12} more"
        raise AssertionError(
            f"{what} differs from the saved dump "
            f"(rtol {rtol:g}, atol {atol:g}):\n  {shown}{more}")
