import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path
from sklearn.metrics import average_precision_score

from chemprop import data, featurizers, models


def predict(checkpoint_path, df_test, smiles_column='molecule_smiles'):

    # Need to use load_from_file to load the model, and map_location to load the model on CPU
    mpnn = models.MPNN.load_from_file(checkpoint_path, map_location=torch.device('cpu'))

    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in df_test[smiles_column]]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
        )
        test_preds = trainer.predict(mpnn, test_loader)

    test_preds = np.concatenate(test_preds, axis=0)
    df_test['pred'] = test_preds

    return df_test


def score(checkpoint_path, test_path, smile_column='molecule_smiles'):
    df_pred = predict(checkpoint_path, test_path, smile_column)

    return average_precision_score(df_pred.binds, df_pred.pred, average='micro')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('checkpoint', type=Path, help='Path to the model checkpoint')
    parser.add_argument('test', type=Path, help='Path to the test dataset')
    args = parser.parse_args()

    score = predict(args.checkpoint, args.test)
    print(score)
