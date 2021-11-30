import traceback
import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import torch
import sys
import argparse
from warnings import simplefilter

from compositors import OS_PGSM
from datasets.utils import windowing, sliding_split
from datasets.monash_forecasting import load_dataset
from os.path import exists
from pathlib import Path
from experiments import implemented_datasets, load_models, ospgsm_experiment_configurations, min_distance_drifts
from utils import euclidean, smape
from sklearn.metrics import mean_squared_error
from evaluate_performance import calc_average_ranks

def evaluate_test(model, x_test, lag=5, loss=smape):
    predictions = np.zeros_like(x_test)

    x = x_test[:lag]
    predictions[:lag] = x

    for x_i in range(lag, len(x_test)):
        x = x_test[x_i-lag:x_i].unsqueeze(0)
        predictions[x_i] = np.squeeze(model.predict(x.unsqueeze(0)))
        
    error = loss(x_test.numpy(), predictions)
    return predictions, error

def run_single_models(models, model_names, X_val, X_test, ds_name, ds_index, loss=smape, dry_run=False, verbose=True):

    # Create folders (if not exist)
    if not exists(f"results/{ds_name}"):
        path = Path(f"results/{ds_name}")
        path.mkdir(parents=True, exist_ok=True)

    # Save ground truth time-series as well
    np.savetxt(f"results/{ds_name}/{ds_index}_y_test.csv", X_test)
    np.savetxt(f"results/{ds_name}/{ds_index}_y_val.csv", X_val)

    # Evaluate single models
    for i, m in enumerate(models):
        model_name = model_names[i]
        test_result_path = f"results/{ds_name}/{ds_index}_{model_name}_test.csv"
        val_result_path = f"results/{ds_name}/{ds_index}_{model_name}_val.csv"

        if exists(test_result_path) or exists(val_result_path):
            print(f"Skipping evaluation of {model_name} on {ds_name} (#{ds_index}) because it exits...")
            continue

        if verbose:
            print(f"Evaluate {model_name} on {ds_name} (#{ds_index})")

        # TODO: Removed multiple running. Is this okay?
        preds_test, _ = evaluate_test(m, X_test, loss=loss)

        x_val_small, _ = sliding_split(X_val, 5, use_torch=True)
        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])

        np.savetxt(test_result_path, preds_test)
        np.savetxt(val_result_path, preds_val)

def run_comparison(models, X_val, X_test, ds_name, ds_index, dry_run=False):
    n_omegas = [6, 10, 15, 20, 25, 30, 40, 50]
    n_ensembles = [3, 5, 10, 15]

    skipped_models = []

    # Run all configurations of our ospgsm algorithm
    for n_omega in n_omegas:
        for n_ensemble in n_ensembles:
            for ospgsm_exp_name, ospgsm_exp_config in ospgsm_experiment_configurations.items():
                conf = ospgsm_exp_config()
                conf["n_omega"] = n_omega
                conf["nr_clusters_ensemble"] = n_ensemble
                conf["topm"] = min(2*n_ensemble, 30)
                conf["z"] = n_omega

                config_name = f"{ospgsm_exp_name}_{n_omega}_{n_ensemble}"

                comp_test_result_path = f"results/{ds_name}/{ds_index}_{config_name}_test.csv"
                model_save_path = f"models/{ds_name}/{ds_index}_{config_name}.json"

                if exists(comp_test_result_path) or exists(model_save_path):
                    print(f"Skipping evaluation of {config_name} on {ds_name} because it exits...")
                    continue

                compositor = OS_PGSM(models, conf) 
                try:
                    preds = compositor.run(X_val, X_test)
                except Exception as e:
                    print(f"Exception during run for {config_name} on {ds_name}(#{ds_index})")
                    skipped_models.append([ds_name, n_omega, n_ensemble, ospgsm_exp_name])
                    #traceback.print_exc()
                    continue

                if np.any(np.isnan(preds)):
                    print(f"NaN found in preds for {config_name} on {ds_name}(#{ds_index})")
                    skipped_models.append([ds_name, n_omega, n_ensemble, ospgsm_exp_name])
                    continue

                print(f"Evaluated {config_name} on {ds_name}(#{ds_index})")

                if not dry_run:
                    np.savetxt(comp_test_result_path, preds)
                    compositor.save(model_save_path)

    #return skipped_models
    #topms = [5, 10, 15, 20, 25, 30]
    topms = [5, 15, 20, 25, 30]
    for n_omega in n_omegas:
        for topm in topms:
            config_name = f"min_distance_iterative_{n_omega}_{topm}"
            conf = min_distance_drifts()
            conf["n_omega"] = n_omega
            conf["topm"] = topm
            conf["nr_clusters_ensemble"] = None # iterative
            conf["z"] = n_omega

            comp_test_result_path = f"results/{ds_name}/{ds_index}_{config_name}_test.csv"
            model_save_path = f"models/{ds_name}/{ds_index}_{config_name}.json"

            if exists(comp_test_result_path) or exists(model_save_path):
                print(f"Skipping evaluation of {config_name} on {ds_name} because it exits...")
                continue

            compositor = OS_PGSM(models, conf) 
            try:
                preds = compositor.run(X_val, X_test)
            except Exception as e:
                print(f"Exception during run for {config_name} on {ds_name}(#{ds_index})")
                skipped_models.append([ds_name, n_omega, n_ensemble, ospgsm_exp_name])
                #traceback.print_exc()
                continue

            if np.any(np.isnan(preds)):
                print(f"NaN found in preds for {config_name} on {ds_name}(#{ds_index})")
                skipped_models.append([ds_name, n_omega, n_ensemble, ospgsm_exp_name])
                continue

            print(f"Evaluated {config_name} on {ds_name}(#{ds_index})")

            if not dry_run:
                np.savetxt(comp_test_result_path, preds)
                compositor.save(model_save_path)

    print(skipped_models)


def run_grid_search():
    ensemble_sizes = [5, 7, 10, 13, 15, 17, 20]
    evaluation_log = pd.DataFrame(columns=["classifier_name", "dataset_name", "accuracy", "k", "topm"])
    for ds_name, ds_index in implemented_datasets:

        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
        models, model_names = load_models(ds_name, ds_index, return_names=True)

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)
        for k in ensemble_sizes:
            topms = np.arange(start=1, stop=k, step=2)
            for topm in topms:
                config = min_distance_drifts()
                config["n_omega"] = 40
                config["nr_clusters_ensemble"] = k
                config["topm"] = topm
                config["distance_measure"] = "euclidean"

                comp = OS_PGSM(models, config, random_state=0)
                try:
                    preds = comp.run(X_val, X_test)
                except AssertionError as ae:
                    print(f"Skipping k={k} topm={topm}: {ae.args[0]}")
                    continue

                loss = mean_squared_error(X_test.squeeze(), preds.squeeze())

                assert not np.any(np.isnan(preds))
                evaluation_log = evaluation_log.append({
                    "dataset_name": f"{ds_name}_{ds_index}",
                    "classifier_name": f"k={k}-topm={topm}",
                    "accuracy": loss,
                    "k": k,
                    "topm": topm,
                }, ignore_index=True)
                evaluation_log.to_csv("results/large_experiment.csv")

def main(dry_run):

    simplefilter(action="ignore", category=UserWarning)

    # Run to fix n_omega by comparing different ambiguities on model subsets
    model_subsets = [
        ('electricity_hourly', 0),
        ('electricity_hourly', 1),
        ('electricity_hourly', 2),
        ('electricity_hourly', 3),
        ('electricity_hourly', 4),
        ('weather', 0),
        ('weather', 1),
        ('weather', 2),
        ('weather', 3),
        ('weather', 4),
        ('kdd_cup_2018', 0),
        ('kdd_cup_2018', 1),
        ('kdd_cup_2018', 2),
        ('kdd_cup_2018', 3),
        ('kdd_cup_2018', 4),
        ('solar_10_minutes', 0),
        ('solar_10_minutes', 1),
        ('solar_10_minutes', 2),
        ('solar_10_minutes', 3),
        ('solar_10_minutes', 4),
        ('pedestrian_counts', 0),
        ('pedestrian_counts', 1),
        ('pedestrian_counts', 2),
        ('pedestrian_counts', 3),
        ('pedestrian_counts', 4),
    ]
    n_omegas = [40, 50, 60]
    ensemble_sizes = [15, 20, 25]
    distance_measures = ["euclidean"]

    number_of_configurations = len(n_omegas) * len(ensemble_sizes) * len(model_subsets) * len(distance_measures)
    print(f"Number of configurations: {number_of_configurations}")

    evaluation_log = pd.DataFrame(columns=["classifier_name", "dataset_name", "accuracy", "k", "n_omega", "distance_fn"])
    for ds_name, ds_index in model_subsets:
        print(ds_name, ds_index)
        print("-"*30)
        for ensemble_size in ensemble_sizes:
            for n_omega in n_omegas:
                for distance_measure in distance_measures:
                    config = min_distance_drifts()
                    config["n_omega"] = n_omega
                    config["nr_clusters_ensemble"] = ensemble_size
                    config["topm"] = None
                    config["distance_measure"] = distance_measure

                    X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
                    models, model_names = load_models(ds_name, ds_index, return_names=True)

                    [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)

                    comp = OS_PGSM(models, config, random_state=0)
                    try:
                        preds = comp.run(X_val, X_test)
                    except AssertionError as ae:
                        print(f"Skipping n_omega={n_omega}-k={ensemble_size}-distance_fn={distance_measure}: {ae.args[0]}")
                        continue

                    loss = mean_squared_error(X_test.squeeze(), preds.squeeze())

                    assert not np.any(np.isnan(preds))
                    print(f"Finished n_omega={n_omega}-k={ensemble_size}-distance_fn={distance_measure}")
                    evaluation_log = evaluation_log.append({
                        "dataset_name": f"{ds_name}_{ds_index}",
                        "classifier_name": f"k={ensemble_size}-n_omega={n_omega}-distance_fn={distance_measure}",
                        "accuracy": loss,
                        "k": ensemble_size,
                        "n_omega": n_omega,
                        "distance_fn": distance_measure,
                    }, ignore_index=True)
                    evaluation_log.to_csv("results/large_experiment.csv")
        print("-"*30)

    calc_average_ranks(evaluation_log)
        
    return

    for ds_name, ds_index in implemented_datasets:

        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
        models, model_names = load_models(ds_name, ds_index, return_names=True)

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)
    
        run_single_models(models, list(model_names), X_val, X_test, ds_name, ds_index, dry_run=dry_run)
        run_comparison(models, X_val, X_test, ds_name, ds_index, dry_run=dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands that would be executed")
    args = parser.parse_args()
    
    main(dry_run=args.dry_run)
