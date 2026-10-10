# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The analytic-first certificate vocabulary (#764).

Grades name the CLAIM CLASS (algebraically exact / structure-exact /
certified numerical / heuristic discovery); holds() reports whether the
measured residual met the declared tolerance. Unmeasured quantities are NaN,
never zero, and a heuristic discovery never holds.
"""

import math
import unittest

import tessera

cob = tessera.cobordism


class TestCertificate(unittest.TestCase):
    def test_algebraically_exact_holds_within_tolerance(self):
        cert = cob.Certificate.algebraically_exact(
            cob.CertificateDomain.Static,
            cob.CertificateRegime.PositiveSemidefinite, 1e-16, 1e-12)
        self.assertEqual(cert.grade, cob.CertificateGrade.AlgebraicallyExact)
        self.assertEqual(cert.domain, cob.CertificateDomain.Static)
        self.assertEqual(cert.regime,
                         cob.CertificateRegime.PositiveSemidefinite)
        self.assertTrue(cert.holds())
        self.assertEqual(cert.residual, 1e-16)
        self.assertEqual(cert.tolerance, 1e-12)
        # Conditioning was not measured for a closed-form identity: NaN.
        self.assertTrue(math.isnan(cert.conditioning))
        self.assertTrue(math.isnan(cert.dense_reference_error))

    def test_residual_above_tolerance_does_not_hold(self):
        cert = cob.Certificate.algebraically_exact(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal,
            1e-6, 1e-12)
        self.assertFalse(cert.holds())

    def test_structure_exact_carries_conditioning(self):
        cert = cob.Certificate.structure_exact(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal,
            1e-15, 42.0, 1e-12)
        self.assertEqual(cert.grade, cob.CertificateGrade.StructureExact)
        self.assertEqual(cert.conditioning, 42.0)
        self.assertTrue(cert.holds())

    def test_certified_numerical_band_window(self):
        cert = cob.Certificate.certified_numerical(
            cob.CertificateDomain.BandWindow,
            cob.CertificateRegime.HermitianIndefinite, 1e-9, 1e3, 1e-8)
        self.assertEqual(cert.grade, cob.CertificateGrade.CertifiedNumerical)
        self.assertEqual(cert.domain, cob.CertificateDomain.BandWindow)
        self.assertTrue(cert.holds())

    def test_heuristic_discovery_never_holds(self):
        cert = cob.Certificate.heuristic_discovery(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal)
        self.assertEqual(cert.grade, cob.CertificateGrade.HeuristicDiscovery)
        self.assertFalse(cert.holds())

    def test_default_certificate_is_uncertified(self):
        cert = cob.Certificate()
        self.assertEqual(cert.grade, cob.CertificateGrade.HeuristicDiscovery)
        self.assertFalse(cert.holds())
        self.assertTrue(math.isnan(cert.residual))

    def test_unmeasured_residual_never_holds(self):
        # NaN residual (not measured) must not pass any tolerance.
        cert = cob.Certificate.algebraically_exact(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal,
            float("nan"), 1e-12)
        self.assertFalse(cert.holds())

    def test_dense_reference_error_attaches(self):
        cert = cob.Certificate.certified_numerical(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal,
            1e-12, 10.0, 1e-10)
        self.assertTrue(math.isnan(cert.dense_reference_error))
        cert.set_dense_reference_error(3e-14)
        self.assertEqual(cert.dense_reference_error, 3e-14)

    def test_describe_names_grade_and_verdict(self):
        cert = cob.Certificate.structure_exact(
            cob.CertificateDomain.Static, cob.CertificateRegime.NonNormal,
            1e-15, 2.0, 1e-12)
        text = cert.describe()
        self.assertIn("structure-exact", text)
        self.assertIn("holds=yes", text)


if __name__ == "__main__":
    unittest.main()
