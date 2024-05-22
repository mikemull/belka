from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


def generate_diverse_set_from_file(smiles_file, num_molecules):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)

    with Chem.SmilesMolSupplier(smiles_file, delimiter=',') as suppl:
        mols = [mol for mol in suppl if mol is not None]

    print(mols[0], type(mols[0]), Chem.MolToSmiles(mols[0]))

    fps = [fpgen.GetFingerprint(mol) for mol in mols]
    picker = MaxMinPicker()
    pick_indices = picker.LazyBitVectorPick(fps, len(fps), num_molecules, seed=23)

    return list(pick_indices)


def generate_diverse_set(df_smiles, num_molecules):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3)

    ms = [Chem.MolFromSmiles(smiles) for smiles, id in df_smiles.itertuples(index=False)]

    fps = [fpgen.GetFingerprint(x) for x in ms]
    picker = MaxMinPicker()
    pick_indices = picker.LazyBitVectorPick(fps, len(fps), num_molecules, seed=23)

    return list(pick_indices)
