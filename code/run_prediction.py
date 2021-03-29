import torch
import numpy as np
import argparse
import time
import pandas as pd
import glob

from main import evaluate_test
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Temperature, Bike_Registered, M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import GC_Large, GC_Large_Adaptive_Hoeffding, GC_Small, GC_Large_Euclidian, GC_Small_Euclidian, Baseline, BaseAdaptive
from datasets.utils import windowing, train_test_split, _apply_window, sliding_split, _val_split
from collections import defaultdict
from csv import DictReader
from os.path import exists
from experiments import single_models, implemented_datasets, lag_mapping, load_model, val_keys, comps, comp_names, test_keys, skip_models_composit
from tsx.models.forecaster import Simple_LSTM

def run_comparison(models, runtimes, model_names, override, x_val_small, x_val_big, X_val, X_test, ds_name, lag, overwrite=False):

    def _run_and_save(t, comp, comp_name, x_val):
        modified = False
        if "pred_" + comp_name not in t.columns or comp_name in override:
            if len(x_val.shape) == 1:
                preds = comp.run(x_val, X_test, big_lag=lag_mapping[str(lag)])
            else:
                preds, _ = comp.run(x_val, X_test)
            t["pred_" + comp_name] = np.squeeze(preds)
            modified = True
        else:
            print("Compositor {} already ran, skipping".format(comp_name))
        return modified, t

    test_path = "results/{}_lag{}_test.csv".format(ds_name, lag)
    val_path = "results/{}_lag{}_val.csv".format(ds_name, lag)

    modified_single = False

    # Assume both or neither exists
    if exists(test_path):
        pd_val = pd.read_csv(val_path, delimiter=",")
        pd_test = pd.read_csv(test_path, delimiter=",")

    else:
        pd_val = pd.DataFrame(columns=["# y"])
        pd_test = pd.DataFrame(columns=["# y"])
        pd_val["# y"] = np.squeeze(X_val.numpy())
        pd_test["# y"] = np.squeeze(X_test.numpy())

    for i, m in enumerate(models):

        if "pred_" + model_names[i] in pd_val.columns and model_names[i] not in override:
            print("skip {} because it already exists".format(model_names[i]))
            continue

        # meassure runtime
        before = time.time()
        preds_test, _ = evaluate_test(m, X_test, reuse_predictions=False, lag=lag)
        after = time.time()
        if runtimes is not None:
            runtimes["pred_" + model_names[i]].loc[ds_name] = after-before

        x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])
        pd_val["pred_" + model_names[i]] = preds_val
        pd_test["pred_" + model_names[i]] = preds_test
        modified_single = True
    
    # remove unwanted models from compositions
    comp_models = []
    for m in models:
        if type(m) not in skip_models_composit:
            comp_models.append(m)

    models = comp_models
    start_idx = len(val_keys)

    for idx in range(len(comp_names)):
        c_idx = comps[idx](models, lag, lag_mapping[str(lag)])

        before = time.time()
        if isinstance(c_idx, BaseAdaptive):
            modified_comp, pd_test = _run_and_save(pd_test, c_idx, comp_names[idx], X_val)
        else:
            modified_comp, pd_test = _run_and_save(pd_test, c_idx, comp_names[idx], x_val_big)
        after = time.time()
        if modified_comp and runtimes is not None:
            runtimes["pred_" + comp_names[idx]].loc[ds_name] = after-before

    if modified_single or modified_comp:
        pd_test.to_csv(test_path, index=False)
        pd_val.to_csv(val_path, index=False)
    return runtimes

def _get_m4_ds(name):
    if "hourly" in name:
        ds = M4_Hourly
    elif "daily" in name:
        ds = M4_Daily
    elif "monthly" in name:
        ds = M4_Monthly
    elif "quaterly" in name:
        ds = M4_Quaterly
    elif "weekly" in name:
        ds = M4_Weekly
    else:
        raise NotImplementedError("Unknown M4 dataset", name)

    return ds

def _get_m4_full(part, index):
    if "hourly" in part:
        char = "H"
    elif "daily" in part:
        char = "D"
    elif "monthly" in part:
        char = "M"
    elif "quaterly" in part:
        char = "Q"
    elif "weekly" in part:
        char = "W"
    else:
        raise NotImplementedError("Unknown M4 dataset", part, index)

    return part + "_" + char + str(index)

def main(lag, ds_names=None, override=None):
    if ds_names is not None:
        dataset_names = ds_names
    else:
        dataset_names = implemented_datasets.keys()

    if override is None:
        override = []
    
    model_names = single_models.keys()

    if exists("results/runtimes_{}.csv".format(lag)):
        runtimes = pd.read_csv("results/runtimes_{}.csv".format(lag), index_col=0)
    else:
        runtimes = pd.DataFrame(index=list(implemented_datasets.keys()), columns=test_keys[1:])

    idx_range = list(range(1, 21))

    for d_ind, d_name in enumerate(dataset_names):

        # assume: m4_hourly, m4_weekly, m4_quaterly, m4_monthly etc.
        if d_name.startswith("m4"):

            ds = _get_m4_ds(d_name)()
            keys = ds.train_data.columns
            for idx in idx_range:
                # get all trained single models (some models will not train because the lag is too small)
                ds_full_name = _get_m4_full(d_name, idx)
                all_singles = glob.glob("models/*/{}_lag{}.pth".format(ds_full_name, lag))

                if len(all_singles) != len(list(single_models.keys())):
                    print("Dataset {} {} with lag {} does not contain all single models, skipping".format(d_name, idx, lag))
                    continue

                models = []
                for m_name in model_names:
                    # if m_name not in skip_models_composit:
                    models.append(load_model(m_name, ds_full_name, lag, lag))

                designator = keys[idx-1]
                X_train, X_test = ds.get(designator)
                X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

                if len(X_val) <= lag_mapping[str(lag)]:
                    print(designator, "is to short, skipping")
                    continue

                x_val_big = _val_split(X_val, lag, lag_mapping[str(lag)], use_torch=True)
                x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
                x_val_small = x_val_small.unsqueeze(1)

                X_test = X_test.float()
                X_val = X_val.float()
                run_comparison(models, None, list(model_names), override, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)

        else:
            ds_full_name = d_name
            ds = implemented_datasets[d_name]['ds']()

            X = ds.torch()
            [x_train, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

            ts_length = lag 

            models = []
            for m_name in model_names:
                models.append(load_model(m_name, d_name, lag, ts_length))
        
            runtimes = run_comparison(models, runtimes, list(model_names), override, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)

    runtimes.to_csv("results/runtimes_{}.csv".format(lag))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store", help="choose datasets to use for training", nargs="+")
    parser.add_argument("--override", action="store", help="", nargs="+")
    parser.add_argument("--lag", action="store", help="choose lag to use for training", default=5, type=int)
    args = parser.parse_args()
    main(args.lag, ds_names=args.datasets, override=args.override)
