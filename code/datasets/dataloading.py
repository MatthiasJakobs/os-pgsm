import numpy as np
import torch
from itertools import product
from datasets.monash_forecasting import load_dataset as load_monash
from datasets.monash_forecasting import _get_ds_names as get_monash_names
from datasets.legacy_datasets import load_dataset as load_legacy
from datasets.legacy_datasets import all_legacy_names as get_legacy_names

monash_names = get_monash_names()
monash_indices = list(range(5))
monash_configs = product(monash_names, monash_indices)
legacy_names = get_legacy_names
legacy_indices = [0]
legacy_configs = product(legacy_names, legacy_indices)

implemented_datasets = list(monash_configs) + list(legacy_configs)

def load_dataset(ds_name, ds_index):
    if ds_name in legacy_names:
        return load_legacy(ds_name, ds_index)
    if ds_name in monash_names:
        return load_monash(ds_name, ds_index)
    raise Exception("Unknown ds name", ds_name)

    