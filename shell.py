import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = os.environ.get('DATA_DIR', 'data')
dpath = Path(DATA_DIR)


def load(file_name):
    return pd.read_parquet(os.path.join(DATA_DIR, file_name))


def pl_load(file_name):
    return pl.read_parquet(os.path.join(DATA_DIR, file_name))


def flist(pattern):
    for filename in dpath.glob(pattern):
        print(filename)
