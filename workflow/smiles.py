from rdkit import Chem
from rdkit.Chem import AllChem


def sanitize_smiles(smiles_string: str):
    test_molecule = Chem.MolFromSmiles(smiles_string)

    Me = Chem.MolFromSmiles('C')
    Dy = Chem.MolFromSmiles('[Dy]')

    new_mol = AllChem.ReplaceSubstructs(test_molecule, Dy, Me)[0]

    Chem.SanitizeMol(new_mol)

    return Chem.MolToSmiles(new_mol)
