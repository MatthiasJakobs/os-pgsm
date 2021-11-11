import numpy as np
import argparse
import glob
import re

from datasets import M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import OS_PGSM
from datasets.utils import get_all_M4, windowing, train_test_split, sliding_split
from os.path import exists
from pathlib import Path
from experiments import single_models, implemented_datasets, lag_mapping, load_model, skip_models_composit, m4_data_path, ospgsm_experiment_configurations, min_distance_drifts
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

    n_omegas = [25, 30, 40, 50]
    n_ensembles = [10, 15, 20]
    thetas = [0.1, 0.5]

    skipped_models = []

    # Run all configurations of our ospgsm algorithm
    for n_omega in n_omegas:
        for n_ensemble in n_ensembles:
            for theta in thetas:
                for ospgsm_exp_name, ospgsm_exp_config in ospgsm_experiment_configurations.items():
                    conf = ospgsm_exp_config()
                    conf["n_omega"] = n_omega
                    conf["nr_clusters_ensemble"] = n_ensemble
                    conf["topm"] = min(2*n_ensemble, 30)
                    conf["smoothing_threshold"] = theta
                    conf["z"] = n_omega

                    #comp_test_result_path = f"results/lag{lag}/{ds_name}/{ospgsm_exp_name}_test.csv"
                    comp_test_result_path = f"results/lag{lag}/{ds_name}/{ospgsm_exp_name}_{n_omega}_{n_ensemble}_{theta}_test.csv"
                    print(conf)

                    if exists(comp_test_result_path):
                        print(f"Skipping evaluation of {ospgsm_exp_name} on {ds_name} (lag {lag}) because it exits...")
                        continue

                    if verbose:
                        print(f"Evaluate {ospgsm_exp_name} (lag {lag}) on {ds_name}")

                    compositor = OS_PGSM(models, conf) 
                    try:
                        preds = compositor.run(X_val, X_test)
                    except Exception:
                        skipped_models.append([ds_name, n_omega, n_ensemble, theta, ospgsm_exp_name])
                        continue

                    if np.any(np.isnan(preds)):
                        skipped_models.append([ds_name, n_omega, n_ensemble, theta, ospgsm_exp_name])
                        continue

                    np.savetxt(comp_test_result_path, preds)
        
                min_distance_config = min_distance_drifts()
                min_distance_config["n_omega"] = n_omega
                min_distance_config["nr_clusters_ensemble"] = n_ensemble
                min_distance_config["topm"] = 2*n_ensemble
                min_distance_config["smoothing_threshold"] = theta

                #comp_test_result_path = f"results/lag{lag}/{ds_name}/{ospgsm_exp_name}_test.csv"
                comp_test_result_path = f"results/lag{lag}/{ds_name}/min_distance_{n_omega}_{n_ensemble}_{theta}_test.csv"
                print(min_distance_config)

                if exists(comp_test_result_path):
                    print(f"Skipping evaluation of min_distance on {ds_name} (lag {lag}) because it exits...")
                    continue

                if verbose:
                    print(f"Evaluate min_distance (lag {lag}) on {ds_name}")

                compositor = OS_PGSM(models, min_distance_config) 
                try:
                    preds = compositor.run(X_val, X_test)
                except Exception:
                    skipped_models.append([ds_name, n_omega, n_ensemble, theta, "min_distance"])
                    continue
                if np.any(np.isnan(preds)):
                    skipped_models.append([ds_name, n_omega, n_ensemble, theta, "min_distance"])
                    continue

                np.savetxt(comp_test_result_path, preds)

    np.savetxt("skipped_models.csv", skipped_models)


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

def main(lag, is_m4, ds_names=None, override=None, verbose=None):
    if ds_names is not None:
        dataset_names = ds_names
    else:
        if is_m4:
            dataset_names = get_all_M4(lag)

            # Preload all M4 datasets to memory
            m4_hourly = M4_Hourly(path=m4_data_path)
            m4_daily = M4_Daily(path=m4_data_path)
            m4_monthly = M4_Monthly(path=m4_data_path)
            m4_quaterly = M4_Quaterly(path=m4_data_path)
            m4_weekly = M4_Weekly(path=m4_data_path)
        else:
            dataset_names = implemented_datasets.keys()

    if override is None:
        override = []

    verbose = verbose is not None
    
    model_names = single_models.keys()

    for d_name in dataset_names:

        if is_m4:
            if "hourly" in d_name:
                ds = m4_hourly
            elif "daily" in d_name:
                ds = m4_daily
            elif "monthly" in d_name:
                ds = m4_monthly
            elif "quaterly" in d_name:
                ds = m4_quaterly
            elif "weekly" in d_name:
                ds = m4_weekly
            else:
                raise NotImplementedError("Unknown M4 dataset", d_name)
            
            designator = d_name.split("_")[-1]

            models = []
            for m_name in model_names:
                models.append(load_model(m_name, d_name, lag, lag))

            X_train, X_test = ds.get(designator)
            X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

            X_test = X_test.float()
            X_val = X_val.float()

            run_single_models(models, list(model_names), X_val, X_test, d_name, lag, verbose=verbose)
            run_comparison(models, X_val, X_test, d_name, lag, verbose=verbose)

        else:
            ds_full_name = d_name
            ds = implemented_datasets[d_name]['ds']()

            X = ds.torch()
            [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

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
    parser.add_argument("--M4", action="store_const", const=True, default=False)
    args = parser.parse_args()
    main(args.lag, args.M4, ds_names=args.datasets, override=args.override, verbose=args.verbose)
