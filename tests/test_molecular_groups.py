import unittest

from octane.molecular_groups import split_molecule_to_substructures


class TestMolecularGroups(unittest.TestCase):
    tests_cases = {
        'CCC': {'CH3': 2, 'CH2': 1},
        'CCCC': {'CH3': 2, 'CH2': 2},
        'CC(C)C': {'CH3': 3, 'CH': 1},
        'CC(C)(C)C': {'CH3': 4, 'C': 1},
        'CC=C': {'CH3': 1, '=CH': 1, '=CH2': 1},
    }
    def test_cases(self):
        for test_case in TestMolecularGroups.test_cases:
            self.assertEqual(split_molecule_to_substructures, False)


if __name__ == '__main__':
    unittest.main()
