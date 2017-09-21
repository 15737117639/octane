from pybel import *
from collections import OrderedDict


class MolecularGroup:
    """ a class that represents a single group of atoms
        that can be a part of a molecule"""

    def __init__(self, pattern, **parametrs):

        self._pattern = pattern
        self._smart_object = Smarts(pattern)
        self._params = parametrs

    def __getattr__(self, attr):
        try:
            return self._params[attr]
        except KeyError:
            return None

    def contains_in(self, molecule):
        """ checks if the molecular group contains in some molecule
            and returns the number of that groups """

        positions = self._smart_object.findall(molecule)
        s = sum([sum(p) for p in positions])
        s = 0
        return s * len(positions)

    def __contains__(self, molecule):
        return bool(self.contains_in(molecule))


# a dictionary with different groups (parafins and olefins for now)
_molecular_groups = {
    'CH3': {'pattern': '[CH3]', 'ron': -2.315, 'mon': -0.202},
    'CH2': {'pattern': '[CX4&CH2]', 'ron': -8.488, 'mon': -9.082},
    'CH': {'pattern': '[CX4&CH1]', 'ron': -0.176, 'mon': -1.821},
    'C': {'pattern': '[CX4&CH0]', 'ron': 11.94, 'mon': 11.90},
    '=CH': {'pattern': 'C=[CH1]', 'ron': 0.392, 'mon': -2.293},
    '=C': {'pattern': 'C=[CH0]', 'ron': 8.697, 'mon': 2.703},
    '=CH2': {'pattern': 'C=[CH2]', 'ron': 3.623, 'mon': -0.254},
    '=C=': {'pattern': 'C=C=C', 'ron': -37.37, 'mon': -42.43},
    '=tCH': {'pattern': 'C=[CH1]', 'ron': 6.449, 'mon': 4.743},
    '=cCH': {'pattern': 'C=[CH1]', 'ron': 6.269, 'mon': 2.725},
    '#CH': {'pattern': 'C#[CH1]', 'ron': 18.36, 'mon': 21.36},
    '#C': {'pattern': 'C#[CH0]', 'ron': -7.20, 'mon': -12.96},
}

molecular_groups = OrderedDict(
    {key: MolecularGroup(**_molecular_groups[key]) for key in _molecular_groups}
)

molecular_groups_count = len(molecular_groups)


def split_molecule_to_substructures(smiles_molecule):
    """ """

    molecule = readstring('smi', smiles_molecule)
    return {group_key: molecular_groups[group_key].contains_in(molecule)
            for group_key in molecular_groups}
