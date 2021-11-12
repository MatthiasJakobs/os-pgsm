import numpy as np
import torch

from compositors import OS_PGSM
from datasets.utils import windowing, sliding_split
from datasets.monash_forecasting import load_dataset
from os.path import exists
from pathlib import Path
from experiments import implemented_datasets, load_models, ospgsm_experiment_configurations 
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

def run_single_models(models, model_names, X_val, X_test, ds_name, ds_index, loss=smape, verbose=True):

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
    

def run_comparison(models, X_val, X_test, ds_name, ds_index):
    n_omegas = [25, 30, 40, 50]
    n_ensembles = [10, 15, 20]

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


def main():

    for ds_name, ds_index in implemented_datasets:

        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
        models, model_names = load_models(ds_name, ds_index, return_names=True)

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)
    
        run_single_models(models, list(model_names), X_val, X_test, ds_name, ds_index)
        #run_comparison(models, X_val, X_test, ds_name, ds_index, verbose=verbose)

if __name__ == "__main__":
    main()
