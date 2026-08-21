import os
from pathlib import Path
import config

import pandas as pd

def data_import(path):
    data_path = os.path.join(Path(config.__file__).resolve().parent, path)
    return pd.read_csv(data_path)

def data_absolute_path(path):
    return os.path.join(Path(config.__file__).resolve().parent, path)
if __name__ == '__main__':
    print(data_import('data/train.csv'))
    print(data_absolute_path('data/train.csv'))