import functools
import pandas as pd
import psutil
import timeit

from chemprop import data, featurizers, models, nn
from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher

from lightning.pytorch.utilities import CombinedLoader


def test_data_loader(input_file_path, num_workers=0, cached=False):

    smiles_column = 'molecule_smiles'
    target_columns = ['binds']

    process = psutil.Process()
    print(process.memory_info().rss)  # in bytes

    df_input = pd.read_parquet(input_file_path)

    smis = df_input.loc[:, smiles_column].values
    ys = df_input.loc[:, target_columns].values

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data, featurizer)
    train_dset.cache = cached
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, persistent_workers=num_workers > 0)

    data_fetcher = _PrefetchDataFetcher()
    data_fetcher.setup(CombinedLoader([train_loader]))
    data_fetcher

    print(process.memory_info().rss)  # in bytes

    return data_fetcher


def load_data(data_fetcher):
    iter(data_fetcher)
    while True:
        try:
            batch, batch_idx, dataloader_idx = next(data_fetcher)

        except StopIteration:
            print(batch_idx)
            # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
            break


if __name__ == '__main__':

    fetcher = test_data_loader('./data/brd4_200k_20240514.parquet', cached=True, num_workers=2)
    t = timeit.Timer(functools.partial(load_data, fetcher))
    print(t.repeat(number=1))
