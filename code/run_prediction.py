import skorch
import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import torch
import argparse
from warnings import simplefilter
from single_models import Simple_LSTM

from compositors import OS_PGSM, RandomSubsetEnsemble, SimpleLSTMBaseline
from ncl import NegCorLearning
from datasets.utils import windowing, sliding_split
from os.path import exists
from pathlib import Path
from experiments import load_models, all_experiments, min_distance_drifts, ospgsm_st_original, single_models_with_lstm
from utils import euclidean, smape
from sklearn.metrics import mean_squared_error
from evaluate_performance import calc_average_ranks
from datasets.dataloading import implemented_datasets, load_dataset

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

def save_test_forecasters(comp, path):
    test_forecasters = comp.test_forecasters
    x_length = len(test_forecasters)

    binary_list = np.zeros((x_length, len(comp.models)), dtype=np.int8)

    for i in range(x_length):
        for forecaster in test_forecasters[i]:
            binary_list[i][forecaster] = 1

    np.savetxt(path, binary_list, header=",".join([f"model_{i}" for i in range(len(comp.models))]), comments="", delimiter=",")

def run_comparison(models, model_names, X_val, X_test, ds_name, ds_index, dry_run=False, override=False):

    test_result_path = f"results/main_experiments/test_{ds_name}_#{ds_index}.csv"
    val_result_path = f"results/main_experiments/val_{ds_name}_#{ds_index}.csv"

    try:
        df_val = pd.read_csv(val_result_path)
    except Exception:
        df_val = pd.DataFrame()
        df_val["y"] = X_val
    try:
        df_test = pd.read_csv(test_result_path)
    except Exception:
        df_test = pd.DataFrame()
        df_test["y"] = X_test

    x_val_small, _ = sliding_split(X_val, 5, use_torch=True)

    # Predictions for single models
    for m, m_name in zip(models, model_names):
        preds, _ = evaluate_test(m, X_test)
        df_test[m_name] = preds

        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])
        df_val[m_name] = preds_val
    
    if not dry_run:
        df_val.to_csv(val_result_path, index=False)

    # Run all configurations of our ospgsm algorithm
    for comp_class, exp_config in all_experiments:
        config_name = exp_config["name"]

        if config_name in df_test.columns and not override:
            print(f"Skipping {config_name} on {ds_name}(#{ds_index})")
            continue

        compositor = comp_class(models, exp_config) 
        if isinstance(compositor, NegCorLearning):
            used_models = []
            for m in torch.load(f"models/{ds_name}/{ds_index}_ncl.pth"):
                m.use_device = 'cpu'
                used_models.append(m)
            compositor.models = used_models
            compositor.eval()
        if isinstance(compositor, SimpleLSTMBaseline):
            lstm_config = single_models_with_lstm["simplelstm"]
            nr_filters = lstm_config["nr_filters"]
            hidden_states = lstm_config["hidden_states"]
            save_path = f"models/{ds_name}/{ds_index}_simplelstm.pth"
            model = skorch.NeuralNetRegressor(
                Simple_LSTM, 
                module__nr_filters=nr_filters, 
                module__hidden_states=hidden_states, 
                module__ts_length=5)

            model.initialize()
            model.load_params(f_params=save_path)
            compositor = SimpleLSTMBaseline([model.module_], exp_config)

        print(f"Start {config_name} on {ds_name}(#{ds_index})")
        preds = compositor.run(X_val, X_test)

        if np.any(np.isnan(preds)):
            print(f"Error in config {config_name}: Prediction contains NaN values")
            continue

        df_test[config_name] = preds

        print(f"Evaluated {config_name} on {ds_name}(#{ds_index})")

        if not dry_run:
            df_test.to_csv(test_result_path, index=False)
            if not (isinstance(compositor, NegCorLearning) or isinstance(compositor, SimpleLSTMBaseline)):
                save_test_forecasters(compositor, f"results/test_forecasters/{ds_name}_{ds_index}_{config_name}.csv")

def remove_compositors(ds_name, ds_index, config_names):
    test_result_path = f"results/main_experiments/test_{ds_name}_#{ds_index}.csv"
    df = pd.read_csv(test_result_path)
    df = df.drop(columns=config_names, errors="ignore")
    df.to_csv(test_result_path, index=False)

def main(dry_run, override):

    simplefilter(action="ignore", category=UserWarning)

    for ds_name, ds_index in implemented_datasets:
        #remove_compositors(ds_name, ds_index, ["OEP-ROC-Euc", "OEP-ROC-5-topm", "OEP-ROC-10-topm", "OEP-ROC-15-topm", "OEP-ROC-20-topm", "OEP-ROC-15", "OEP-ROC-5", "OEP-ROC-10", "OEP-ROC-20", "OEP-ROC-I", "OEP-ROC-II", "OEP-ROC-ST", "OEP-ROC-Per", "OEP-ROC-C"])
        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
        models, model_names = load_models(ds_name, ds_index, return_names=True, device='cpu')

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)
        run_comparison(models, list(model_names), X_val, X_test, ds_name, ds_index, dry_run=dry_run, override=override)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, override=args.override)
