# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The Python namespace of a bound type matches the directory that declares it.

The bindings in ``src/<area>/`` bind the types declared in ``include/<area>/``
into ``tessera.<area>``, and the files directly under ``src/`` and
``include/`` bind into the root ``tessera`` namespace. These checks pin the
types whose bindings once lived in another directory's file.
"""

import unittest

import tessera


class TestBindingNamespaces(unittest.TestCase):
    def test_the_poset_family_is_in_the_root_namespace(self):
        for name in ("Poset", "OrderAgreement", "compareOrders"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(tessera._tessera, name))
                self.assertFalse(hasattr(tessera._tessera.quantum, name))

    def test_the_interaction_simulation_is_in_simulations(self):
        for name in ("InteractionSimulation", "InteractionConfig", "InitialChargeMode"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(tessera.simulations, name))
                self.assertFalse(hasattr(tessera._tessera.quantum, name))
                self.assertIs(getattr(tessera, name), getattr(tessera.simulations, name))

    def test_matter_is_its_own_submodule(self):
        import tessera.matter as matter
        for name in ("MatterConfiguration", "HingeType"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(matter, name))
                self.assertIs(getattr(tessera, name), getattr(matter, name))
        self.assertTrue(hasattr(tessera._tessera, "matter"))


if __name__ == "__main__":
    unittest.main()
