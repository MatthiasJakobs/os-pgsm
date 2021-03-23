import torch
import numpy as np
import argparse
import time
import pandas as pd

from main import evaluate_test
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Temperature, Bike_Registered, M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import GC_Large, GC_Large_Adaptive_Hoeffding, GC_Small, GC_Large_Euclidian, GC_Small_Euclidian, Baseline, BaseAdaptive
from datasets.utils import windowing, train_test_split, _apply_window, sliding_split, _val_split
from collections import defaultdict
from csv import DictReader
from os.path import exists
from experiments import single_models, implemented_datasets, lag_mapping, load_model, val_keys, comps, comp_names, test_keys, skip_models_composit
from tsx.models.forecaster import Simple_LSTM

def run_comparison(models, model_names, override, x_val_small, x_val_big, X_val, X_test, ds_name, lag, overwrite=False):

    def _run_and_save(modified, t, comp, comp_name, x_val):
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

    modified = False

    r_times = np.zeros(len(test_keys)-1)

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
        #if np.sum(val_npy[:, i+1]) == 0:

        if "pred_" + model_names[i] in pd_val.columns:
            print("skip {} because it already exists".format(model_names[i]))
            continue

        # meassure runtime
        before = time.time()
        preds_test, _ = evaluate_test(m, X_test, reuse_predictions=False, lag=lag)
        after = time.time()
        r_times[i] = after-before

        x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])
        pd_val["pred_" + model_names[i]] = preds_val
        pd_test["pred_" + model_names[i]] = preds_test
        modified = True
    
    # remove unwanted models from compositions
    comp_models = []
    for m in models:
        if type(m) not in skip_models_composit:
            comp_models.append(m)

    models = comp_models
    start_idx = len(val_keys)

    for idx in range(len(comp_names)):
        c_idx = comps[idx](models, lag)

        before = time.time()
        if isinstance(c_idx, BaseAdaptive):
            modified, pd_test = _run_and_save(modified, pd_test, c_idx, comp_names[idx], X_val)
        else:
            modified, pd_test = _run_and_save(modified, pd_test, c_idx, comp_names[idx], x_val_big)
        after = time.time()
        r_times[len(models) + len(skip_models_composit) + idx] = after-before

    if modified:
        pd_test.to_csv(test_path)
        pd_val.to_csv(val_path)
    return r_times

def main(lag, ds_names=None, override=None):
    if ds_names is not None:
        dataset_names = ds_names
    else:
        dataset_names = implemented_datasets.keys()

    if override is None:
        override = []
    
    model_names = single_models.keys()

    runtimes = np.zeros((len(dataset_names), len(test_keys)-1))

    idx_range = list(range(17))

    for d_ind, d_name in enumerate(dataset_names):
        if d_name.startswith("m4"):
            keys = ds.train_data.columns
            for idx in idx_range:
                designator = keys[idx]
                ds_full_name = "{}_{}".format(ds_name, designator)
                X_train, X_test = ds.get(designator)
                X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

                if len(X_val) < lag*lag:
                    continue
                x_val_big = _val_split(X_val, lag, lag*lag, use_torch=True)
                x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
                x_val_small = x_val_small.unsqueeze(1)

                X_test = X_test.float()
                X_val = X_val.float()

                model_a.load_state_dict(torch.load("models/{}_rnn.pth".format(ds_full_name)))
                model_b.load_state_dict(torch.load("models/{}_cnn.pth".format(ds_full_name)))
                model_c.load_state_dict(torch.load("models/{}_as01.pth".format(ds_full_name)))
                model_d.load_state_dict(torch.load("models/{}_as02.pth".format(ds_full_name)))

                models = [model_a, model_b, model_c, model_d]

                run_comparison(models, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)

        else:
            ds_full_name = d_name
            ds = implemented_datasets[d_name]['ds']()

            X = ds.torch()
            [x_train, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

            ts_length = lag # TODO: Is this correct?

            models = []
            for m_name in model_names:
                models.append(load_model(m_name, d_name, lag, ts_length))
        
            rtimes = run_comparison(models, list(model_names), override, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)
            runtimes[d_ind] = rtimes

    if len(runtimes) == len(implemented_datasets.keys()):
        print("bla")
        #np.save("results/runtimes.npy", runtimes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store", help="choose datasets to use for training", nargs="+")
    parser.add_argument("--override", action="store", help="", nargs="+")
    parser.add_argument("--lag", action="store", help="choose lag to use for training", default=5, type=int)
    args = parser.parse_args()
    main(args.lag, ds_names=args.datasets, override=args.override)
