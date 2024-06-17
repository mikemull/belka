import os

import pandas as pd
import polars

DATA_DIR = os.environ.get('DATA_DIR', 'data')


def load(file_name):
    return pd.read_parquet(os.path.join(DATA_DIR, file_name))


def pl_load(file_name):
    return polars.read_parquet(os.path.join(DATA_DIR, file_name))
