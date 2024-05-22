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


def sample_training_data():
    logging.info('Sampling from balanced dataset')
    df_balanced = pd.read_parquet('../brd4_train_balanced.parquet')
    if diverse:
        logging.info('Generating diverse set')
        index = generate_diverse_set('../brd4_smiles.csv', num_diverse)
        df_sample = df_balanced.iloc[index]
    else:
        if sample_size:
            df_sample = df_balanced.sample(sample_size, axis='index')
        else:
            df_sample = df_balanced

    df_sample[['molecule_smiles', 'binds']].to_csv('brd4_smiles_sample.csv', index=False)
    logging.info('Done!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('-i', '--input')
    parser.add_argument('-b', '--build', action='store_true')
    parser.add_argument('--balanced', action='store_true')

    args = parser.parse_args()

    df_training_data = build_training_set(args.input, balanced=args.balanced)
    logger.info(df_training_data.info())
    logger.info('Done!')
