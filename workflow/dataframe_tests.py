import functools
import psutil
import timeit

import pandas as pd
import polars as pl


def pl_sample_test(df):
    sample_df = df.filter(pl.int_range(pl.len()).shuffle().over('buildingblock1_smiles') < 7500)


def pd_sample_test(df):
    sample_df = df.groupby('buildingblock1_smiles').sample(n=7500)


def sampling_test(framework='polars'):
    process = psutil.Process()

    if framework == 'polars':
        df = pl.read_parquet('./data/train_brd4.parquet')
        func = functools.partial(pl_sample_test, df)

    if framework == 'pandas':
        df = pd.read_parquet('./data/train_brd4.parquet')
        func = functools.partial(pd_sample_test, df)

    t = timeit.Timer(func)
    print(f"{framework},{t.timeit(1)},{process.memory_info().rss}")


def pl_merge_test(df1, df2):
    df1.join(df2, on='molecule_smiles')


def pd_merge_test(df1, df2):

    df1.merge(df2, on='molecule_smiles')


def merge_test(framework='polars'):
    process = psutil.Process()

    if framework == 'polars':
        df1 = pl.read_csv('./data/foundation_set.csv')
        df2 = pl.read_parquet('./data/train_brd4.parquet')

        func = functools.partial(pl_merge_test, df1, df2)

    if framework == 'pandas':
        df1 = pd.read_csv('./data/foundation_set.csv')
        df2 = pd.read_parquet('./data/train_brd4.parquet')

        func = functools.partial(pd_merge_test, df1, df2)

    t = timeit.Timer(func)
    print(f"{framework},{t.timeit(1)},{process.memory_info().rss}")


def load_test(framework='polars'):

    func = functools.partial(pd.read_parquet, './data/train_brd4.parquet')

    t = timeit.Timer(func)
    print(f"{framework},{t.timeit(1)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Dataframe tests')
    parser.add_argument('lib', help='polars or pandas')
    parser.add_argument('test', help='Test name')
    args = parser.parse_args()

    if args.test == 'sample':
        sampling_test(args.lib)

    if args.test == 'merge':
        merge_test(args.lib)

    if args.test == 'load':
        load_test(args.lib)
