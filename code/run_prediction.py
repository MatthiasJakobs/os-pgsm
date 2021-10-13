import numpy as np
import argparse
import glob

from datasets import M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import OS_PGSM
from datasets.utils import windowing, train_test_split, sliding_split
from os.path import exists
from pathlib import Path
from experiments import single_models, implemented_datasets, lag_mapping, load_model, skip_models_composit, m4_data_path, ospgsm_experiment_configurations
from utils import smape

def evaluate_test(model, x_test, lag=5, loss=smape):
    predictions = np.zeros_like(x_test)

    x = x_test[:lag]
    predictions[:lag] = x

    for x_i in range(lag, len(x_test)):
        x = x_test[x_i-lag:x_i].unsqueeze(0)
        predictions[x_i] = np.squeeze(model.predict(x.unsqueeze(0)))
        
    error = loss(x_test.numpy(), predictions)
    return predictions, error

def run_single_models(models, model_names, X_val, X_test, ds_name, lag, loss=smape, verbose=False):

    # Create folders (if not exist)
    if not exists(f"results/lag{lag}/{ds_name}"):
        path = Path(f"results/lag{lag}/{ds_name}")
        path.mkdir(parents=True, exist_ok=True)

    # Save ground truth time-series as well
    np.savetxt(f"results/lag{lag}/{ds_name}/y_test.csv", X_test)
    np.savetxt(f"results/lag{lag}/{ds_name}/y_val.csv", X_val)

    # Evaluate single models
    for i, m in enumerate(models):
        model_name = model_names[i]
        test_result_path = f"results/lag{lag}/{ds_name}/{model_name}_test.csv"
        val_result_path = f"results/lag{lag}/{ds_name}/{model_name}_val.csv"

        if exists(test_result_path) or exists(val_result_path):
            print(f"Skipping evaluation of {model_name} on {ds_name} (lag {lag}) because it exits...")
            continue

        if verbose:
            print(f"Evaluate {model_name} (lag {lag}) on {ds_name}")

        # TODO: Removed multiple running. Is this okay?
        preds_test, _ = evaluate_test(m, X_test, lag=lag, loss=loss)

        x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])

        np.savetxt(test_result_path, preds_test)
        np.savetxt(val_result_path, preds_val)
    

def run_comparison(models, X_val, X_test, ds_name, lag, verbose=False):
    # remove unwanted models from compositions
    comp_models = []
    for m in models:
        if type(m) not in skip_models_composit:
            comp_models.append(m)

    models = comp_models

    # Run all configurations of our ospgsm algorithm
    for ospgsm_exp_name, ospgsm_exp_config in ospgsm_experiment_configurations.items():
        comp_test_result_path = f"results/lag{lag}/{ds_name}/{ospgsm_exp_name}_test.csv"

        if exists(comp_test_result_path):
            print(f"Skipping evaluation of {ospgsm_exp_name} on {ds_name} (lag {lag}) because it exits...")
            continue

        if verbose:
            print(f"Evaluate {ospgsm_exp_name} (lag {lag}) on {ds_name}")

        compositor = OS_PGSM(models, ospgsm_exp_config(lag)) 
        preds = compositor.run(X_val, X_test)

        np.savetxt(comp_test_result_path, preds)
        

    # TODO: Here, we would test the other methods we want to compare against


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

def main(lag, ds_names=None, override=None, verbose=None):
    if ds_names is not None:
        dataset_names = ds_names
    else:
        dataset_names = implemented_datasets.keys()

    if override is None:
        override = []

    verbose = verbose is not None
    
    model_names = single_models.keys()

    for d_name in dataset_names:

        # assume: m4_hourly, m4_weekly, m4_quaterly, m4_monthly etc.
        if d_name.startswith("m4"):

            if d_name.startswith("m4_quaterly"):
                idx_range = list(range(5, 21))
            else:
                idx_range = list(range(1, 21))
            ds = _get_m4_ds(d_name)(path=m4_data_path)
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

                x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
                x_val_small = x_val_small.unsqueeze(1)

                X_test = X_test.float()
                X_val = X_val.float()
                run_comparison(models, X_val, X_test, ds_full_name, lag)

        else:
            ds_full_name = d_name
            ds = implemented_datasets[d_name]['ds']()

            X = ds.torch()
            [_, x_val_small], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

            models = []
            for m_name in model_names:
                models.append(load_model(m_name, d_name, lag, lag))
        
            run_single_models(models, list(model_names), X_val, X_test, ds_full_name, lag, verbose=verbose)
            run_comparison(models, X_val, X_test, ds_full_name, lag, verbose=verbose)

    #runtimes.to_csv("results/runtimes_{}.csv".format(lag))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store", help="choose datasets to use for training", nargs="+")
    parser.add_argument("--override", action="store", help="", nargs="+")
    parser.add_argument("--lag", action="store", help="choose lag to use for training", default=5, type=int)
    parser.add_argument("--verbose", help="print out all kinds of information", nargs="+")
    args = parser.parse_args()
    main(args.lag, ds_names=args.datasets, override=args.override, verbose=args.verbose)
