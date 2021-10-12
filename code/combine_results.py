import torch
import numpy as np
import argparse
import time
import pandas as pd
import glob

from tqdm import tqdm, trange
from datasets import M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import OS_PGSM
from datasets.utils import windowing, train_test_split, _apply_window, sliding_split, _val_split
from os.path import exists
from experiments import single_models, implemented_datasets, lag_mapping, load_model, skip_models_composit, m4_data_path, ospgsm_experiment_configurations, test_keys
from utils import smape
from sklearn.metrics import mean_squared_error

def combine(ds_name, lag):
    model_names = list(single_models.keys())

    pd_val = pd.DataFrame()
    pd_test = pd.DataFrame()

    # Ground truth values
    y_test = np.genfromtxt(f"results/lag{lag}/{ds_name}/y_test.csv")
    y_val = np.genfromtxt(f"results/lag{lag}/{ds_name}/y_val.csv")
    pd_val["# y"] = y_val
    pd_test["# y"] = y_test

    for model_name in model_names:
        single_test_result_path = f"results/lag{lag}/{ds_name}/{model_name}_test.csv"
        single_val_result_path = f"results/lag{lag}/{ds_name}/{model_name}_val.csv"

        single_test_result = np.genfromtxt(single_test_result_path)
        single_val_result = np.genfromtxt(single_val_result_path)
        pd_test["pred_" + model_name] = single_test_result
        pd_val["pred_" + model_name] = single_val_result

    for experiment_name in ospgsm_experiment_configurations.keys():
        comp_test_results = np.genfromtxt(f"results/lag{lag}/{ds_name}/{experiment_name}_test.csv")
        pd_test["pred_" + experiment_name] = comp_test_results

    pd_test.to_csv(f"results/{ds_name}_lag{lag}_test.csv", index=False)
    pd_val.to_csv(f"results/{ds_name}_lag{lag}_val.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", help="choose datasets to use for training", nargs="+")
    parser.add_argument("--lag", action="store", help="choose lag to use for training", default=5, type=int)
    args = parser.parse_args()
    combine(args.dataset[0], args.lag)
