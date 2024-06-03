import logging
from typing import Tuple

import pandas as pd

from workflow.diverse import generate_diverse_set


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


diverse = False
sample_size = None
num_diverse = 200_000


def build_balanced(df: pd.DataFrame, num_positives: int, num_total: int, diverse: bool = False) -> pd.DataFrame:
    df_binds = df[df.binds == 1].sample(num_positives)

    df_no_binds = df[df.binds == 0].sample(num_total - num_positives)

    if diverse:
        index = generate_diverse_set(df_no_binds[['molecule_smiles', 'id']], num_diverse)
        df_no_binds = df_no_binds.iloc[index]

    return pd.concat([df_binds, df_no_binds])


def build_training_set(
        file_path: str,
        balanced: bool = False,
        diverse: bool = False,
        num_positives: int = 1000,
        num_total: int = 2000
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logging.info('Loading dataset')
    df = pd.read_parquet(file_path)
    logger.info(df.info())

    if balanced:
        logging.info('Balancing dataset')
        df = build_balanced(df, num_positives, num_total, diverse=diverse)

    return df


def sample_training_data(df_input, frac=0.8, sample_size=None, num_diverse=None):
    logging.info('Sampling from balanced dataset')
    df_test = None
    if num_diverse:
        logging.info('Generating diverse set')
        index = generate_diverse_set('../brd4_smiles.csv', num_diverse)
        df_sample = df_input.iloc[index]
    else:
        if sample_size:
            df_sample = df_input.sample(sample_size, axis='index')
        else:
            df_sample = df_input.sample(frac=frac, axis='index')
            df_test = df_input.drop(df_sample.index)

    return (
        df_sample[['id', 'molecule_smiles', 'Canonical_Smiles', 'binds']],
        df_test[['id', 'molecule_smiles', 'Canonical_Smiles', 'binds']]
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-b', '--build', action='store_true')
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--binds', type=int, default=1000)
    parser.add_argument('--total', type=int, default=2000)

    args = parser.parse_args()

    df_training_data = build_training_set(
        args.input,
        balanced=args.balanced,
        num_positives=args.binds,
        num_total=args.total)
    df_training_data, df_test_data = sample_training_data(df_training_data, frac=0.8)
    logger.info(df_training_data.info())
    logger.info(df_test_data.info())
    df_training_data.to_parquet(f'{args.output}_train.parquet')
    df_test_data.to_parquet(f'{args.output}_test.parquet')
    logger.info('Done!')
