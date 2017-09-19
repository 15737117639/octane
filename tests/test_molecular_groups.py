import unittest

from octane.molecular_groups import split_molecule_to_substructures


# dummy test
class TestMolecularGroups(unittest.TestCase):

    test_cases = {
        'CCC': {'CH3': 2, 'CH2': 1},
        'CCCC': {'CH3': 2, 'CH2': 2},
        'CC(C)C': {'CH3': 3, 'CH': 1},
        'CC(C)(C)C': {'CH3': 4, 'C': 1},
        'CC=C': {'CH3': 1, '=CH': 1, '=CH2': 1},
    }

    def test(self):
        for molecule in self.test_cases:
            supers = split_molecule_to_substructures(molecule)
            sub = self.test_cases[molecule]
            # get common elements of dictionaries
            common = {key: supers[key]
                      for key in set(supers.keys()).intersection(set(sub.keys()))}
            self.assertEqual(common, sub)

if __name__ == '__main__':
    unittest.main()
