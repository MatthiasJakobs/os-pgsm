import numpy as np
import torch
from itertools import product
from datasets.monash_forecasting import load_dataset as load_monash
from datasets.monash_forecasting import _get_ds_names as get_monash_names
from datasets.legacy_datasets import load_dataset as load_legacy
from datasets.legacy_datasets import all_legacy_names as get_legacy_names

def get_monash_configs():
    idx_mapping = {
        'electricity_hourly': [0, 1, 2, 5, 6, 7, 9, 10, 12, 13, 14],
        'kdd_cup_2018': [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        'm4_daily': [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        'm4_weekly': [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'weather': list(range(15)),
        'pedestrian_counts': [0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14],
        'solar_10_minutes': [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
    }
    monash_names = []
    monash_configs = []
    for name in idx_mapping.keys():
        monash_names.append(name)
        monash_configs.extend(list(product([name], idx_mapping[name])))

    return monash_names, monash_configs

legacy_names = get_legacy_names
legacy_indices = [0]
legacy_configs = product(legacy_names, legacy_indices)

monash_names, monash_configs = get_monash_configs()

implemented_datasets = monash_configs + list(legacy_configs)

def load_dataset(ds_name, ds_index):
    if ds_name in legacy_names:
        return load_legacy(ds_name, ds_index)
    if ds_name in monash_names:
        return load_monash(ds_name, ds_index)
    raise Exception("Unknown ds name", ds_name)