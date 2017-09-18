from pybel import *


class MolecularGroup:
    """ a class that represents a single group of atoms
        that can be a part of a molecule"""

    def __init__(self, smiles_structure, **parametrs):

        self._smiles_structure = smiles_structure
        self._smart_object = Smarts(smiles_structure)
        self._params = parametrs

    def __getattr__(self, attr):
        try:
            return self._params[attr]
        except KeyError:
            return None

    def contains_in(self, molecule):
        """ checks if the molecular group contains in some molecule
            and returns the number of that groups """

        return len(self._smart_object.findall(molecule))

    def __contains__(self, molecule):
        return bool(self.contains_in(molecule))


# a dictionary with different groups (parafins and olefins for now)
molecular_groups = {
    'CH3': MolecularGroup('[CH3]', ron=-2.315, mon=-0.202),
    'CH2': MolecularGroup('[CX2H2]', ron=-8.488, mon=-9.082),
    'CH': MolecularGroup('[CX3H1]', ron=-0.176, mon=-1.821),
    'C': MolecularGroup('[CX4H0]', ron=11.94, mon=11.90),
    '=CH': MolecularGroup('C=[CH1]', ron=0.392, mon=-2.293),
    '=C': MolecularGroup('C=[CH0]', ron=8.697, mon=2.703),
    '=CH2': MolecularGroup('C=[CH2]', ron=3.623, mon=-0.254),
    '=C=': MolecularGroup('C=C=C', ron=-37.37, mon=-42.43),
    '=tCH': MolecularGroup('C=[CH1]', ron=6.449, mon=4.743),
    '=cCH': MolecularGroup('C=[CH1]', ron=6.269, mon=2.725),
    '#CH': MolecularGroup('C#[CH1]', ron=18.36, mon=21.36),
    '#C': MolecularGroup('C#[CH0]', ron=-7.20, mon=-12.96),
}


def split_molecule_to_substructures(smiles_molecule):
    """ """

    molecule = readstring('smi', smiles_molecule)
    return {group_key: molecular_groups[group_key].contains_in(molecule)
            for group_key in molecular_groups}
