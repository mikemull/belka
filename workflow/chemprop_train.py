import logging

import pandas as pd
import psutil

from pathlib import Path
from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def setup_and_train(input_file_path, max_epochs=10, checkpoint=None):

    logger.info(f'{max_epochs=} {checkpoint=}')
    num_workers = 0
    smiles_column = 'molecule_smiles'
    target_columns = ['binds']

    process = psutil.Process()
    print(process.memory_info().rss)  # in bytes

    df_input = pd.read_parquet(input_file_path)
    logger.info(f'Loaded {len(df_input)} rows from input file')
    logger.info(process.memory_info().rss)

    smis = df_input.loc[:, smiles_column].values
    ys = df_input.loc[:, target_columns].values

    logger.info('Building molecule datapoints from SMILES')
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    logger.info(process.memory_info().rss)

    logger.info('Splitting data into train, val, test')
    mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    logger.info(f'Train: {len(train_data)} Val: {len(val_data)} Test: {len(test_data)}')
    logger.info(process.memory_info().rss)

    logger.info('Featurizing molecules')

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer)
    # scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data, featurizer)
    # val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data, featurizer)

    logger.info('Building dataloaders')
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)  # , persistent_workers=True)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()

    # output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.BinaryClassificationFFN()  # output_transform=output_transform)
    batch_norm = True
    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()]

    if checkpoint:
        logger.info(f'Loading model from checkpoint {checkpoint}')
        mpnn = models.MPNN.load_from_checkpoint(checkpoint)
    else:
        mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    logger.info(process.memory_info().rss)

    logger.info('Training MPNN')

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs
    )

    logger.info(process.memory_info().rss)

    trainer.fit(mpnn, train_loader, val_loader, ckpt_path=checkpoint)

    logger.info(process.memory_info().rss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('training_data', type=Path, help='Path to the training dataset')
    parser.add_argument('--max-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--checkpoint', type=Path, help='Path to the model checkpoint')
    args = parser.parse_args()

    setup_and_train(args.training_data, max_epochs=args.max_epochs, checkpoint=args.checkpoint)
