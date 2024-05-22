import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path
from sklearn.metrics import average_precision_score

from chemprop import data, featurizers, models


def predict(checkpoint_path, test_path, smiles_column='molecule_smiles'):

    mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)

    df_test = pd.read_csv(test_path)

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

    return average_precision_score(df_test.binds, df_test.pred, average='micro')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('checkpoint', type=Path, help='Path to the model checkpoint')
    parser.add_argument('test', type=Path, help='Path to the test dataset')
    args = parser.parse_args()

    score = predict(args.checkpoint, args.test)
    print(score)
