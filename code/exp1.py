import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.utils import windowing, _apply_window
from compositors import OS_PGSM
from experiments import implemented_datasets, ospgsm_int_original, load_models, ospgsm_st_original
from datasets.monash_forecasting import load_dataset
from sklearn.metrics import mean_squared_error
from test_barplots import grouped_barplot
from os.path import exists

''' Count the number of empty rocs '''
warnings.filterwarnings("ignore")

def create_dataset_experiment():
    path = "results/nr_zeros.csv"
    if exists(path):
        print(f"{path} exists, skipping...")
        return
    log = pd.DataFrame(columns=["method", "dataset", "inverted_relu", "roc_only_best", "percentage_empty_rocs", "MSE"])
    for ds_name, ds_index in implemented_datasets:
        if ds_name == "weather" and ds_index == 3:
            continue
        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)

        models = load_models(ds_name, ds_index)

        for roc_take_only_best in [True, False]:
            for inverted_relu in [True, False]:
                ospgsm_int_config = ospgsm_int_original()
                ospgsm_int_config["smoothing_threshold"] = 0.1
                ospgsm_int_config["roc_take_only_best"] = roc_take_only_best
                ospgsm_int_config["invert_relu"] = inverted_relu
                torch.manual_seed(0)
                new_comp = OS_PGSM(models, ospgsm_int_config, random_state=1010)
                preds = new_comp.run(X_val, X_test)
                try:
                    loss = mean_squared_error(X_test, preds)
                    roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

                    zero_length_rocs = np.sum(roc_lengths == 0)
                    other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]
                    print(f"OSPGSM-Int on {ds_name} #{ds_index} percentage of empty RoCs {zero_length_rocs / len(new_comp.rocs):.3f}")
                    log = log.append({
                        "method": "OS-PGSM-Int",
                        "dataset": f"{ds_name}_{ds_index}",
                        "inverted_relu": inverted_relu,
                        "roc_only_best": roc_take_only_best,
                        "percentage_empty_rocs": zero_length_rocs / len(new_comp.rocs),
                        "MSE": loss
                    }, ignore_index=True)
                except:
                    print(f"OSPGSM-Int on {ds_name} #{ds_index} ALL ROCS EMPTY")

                ospgsm_st_config = ospgsm_st_original()
                ospgsm_st_config["smoothing_threshold"] = 0.1
                ospgsm_st_config["roc_take_only_best"] = roc_take_only_best
                ospgsm_st_config["invert_relu"] = inverted_relu
                torch.manual_seed(0)
                new_comp = OS_PGSM(models, ospgsm_st_config, random_state=1010)
                preds = new_comp.run(X_val, X_test)
                try:
                    loss = mean_squared_error(X_test, preds)
                    roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

                    zero_length_rocs = np.sum(roc_lengths == 0)
                    other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]
                    print(f"OSPGSM-St on {ds_name} #{ds_index} percentage of empty RoCs {zero_length_rocs / len(new_comp.rocs):.3f}")
                    log = log.append({
                        "method": "OS-PGSM-St",
                        "dataset": f"{ds_name}_{ds_index}",
                        "inverted_relu": inverted_relu,
                        "roc_only_best": roc_take_only_best,
                        "percentage_empty_rocs": zero_length_rocs / len(new_comp.rocs),
                        "MSE": loss
                    }, ignore_index=True)
                except:
                    print(f"OSPGSM-St on {ds_name} #{ds_index} ALL ROCS EMPTY")
    log.to_csv(path)

def plot_nonempty_rocs():
    path = "results/nr_zeros.csv"
    df = pd.read_csv(path)

    padding = 0.1
    colors = ["red", "blue", "green", "orange"]

    for model in ["OS-PGSM-St", "OS-PGSM-Int"]:

        fig, ax = plt.subplots(figsize=(12,6))

        plt.suptitle(model)

        subset = df[(df["method"] == model)] 
        subset = subset.drop(columns=["MSE", "method"])
        # Drop first column of dataframe
        subset = subset.iloc[: , 1:]

        # First bar: Original (normal relu, only_best_model)
        s = subset[(subset["inverted_relu"] == False) & (subset["roc_only_best"] == True)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        X = np.arange(len(s))
        first_empty = 1 - s["percentage_empty_rocs"].to_numpy()

        # Second bar: inverted_relu , only_best_model
        s = subset[(subset["inverted_relu"] == True) & (subset["roc_only_best"] == True)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        second_empty = 1 - s["percentage_empty_rocs"].to_numpy()

        # Third bar: normal relu, all_models
        s = subset[(subset["inverted_relu"] == False) & (subset["roc_only_best"] == False)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        third_empty = 1 - s["percentage_empty_rocs"].to_numpy()

        # Third bar: inverted relu, all_models
        s = subset[(subset["inverted_relu"] == True) & (subset["roc_only_best"] == False)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        fourth_empty = 1 - s["percentage_empty_rocs"].to_numpy()

        ys = np.array([first_empty, second_empty, third_empty, fourth_empty]).T

        labels = ["Original", "Inverted ReLU", "RoCs from all models", "Inverted ReLu and RoCs from all models"]

        grouped_barplot(ax, X, ys, padding, colors, labels=labels)

        ax.get_xaxis().set_ticks(np.arange(len(X)))
        ax.get_xaxis().set_ticklabels([])
        ax.set_ylabel("Percentage of non-empty RoCs (more is better)")
        ax.set_xlabel("Datasets")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"plots/empty_rocs_{model}.png")
        plt.close()

def plot_dataset_experiment():
    path = "results/nr_zeros.csv"
    df = pd.read_csv(path)

    padding = 0.1
    colors = ["red", "blue", "green", "orange"]

    for model in ["OS-PGSM-St", "OS-PGSM-Int"]:

        fig, ax = plt.subplots(figsize=(12,6))

        plt.suptitle(model)

        subset = df[(df["method"] == model)] 
        subset = subset.drop(columns=["percentage_empty_rocs", "method"])
        # Drop first column of dataframe
        subset = subset.iloc[: , 1:]

        # First bar: Original (normal relu, only_best_model)
        s = subset[(subset["inverted_relu"] == False) & (subset["roc_only_best"] == True)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        X = np.arange(len(s))
        first_y = s["MSE"].to_numpy()

        # Second bar: inverted_relu , only_best_model
        s = subset[(subset["inverted_relu"] == True) & (subset["roc_only_best"] == True)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        second_y = s["MSE"].to_numpy()

        # Third bar: normal relu, all_models
        s = subset[(subset["inverted_relu"] == False) & (subset["roc_only_best"] == False)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        third_y = s["MSE"].to_numpy()

        # Third bar: inverted relu, all_models
        s = subset[(subset["inverted_relu"] == True) & (subset["roc_only_best"] == False)]
        s = s.drop(columns=["inverted_relu", "roc_only_best"])
        fourth_y = s["MSE"].to_numpy()

        ys = np.array([first_y, second_y, third_y, fourth_y]).T

        labels = ["Original", "Inverted ReLU", "RoCs from all models", "Inverted ReLu and RoCs from all models"]

        grouped_barplot(ax, X, ys, padding, colors, labels=labels)

        ax.get_xaxis().set_ticks(np.arange(len(X)))
        ax.get_xaxis().set_ticklabels([])
        ax.set_ylabel("MSE on test dataset (lower is better)")
        ax.set_xlabel("Datasets")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"plots/inverted_relu_{model}.png")
        plt.close()



if __name__ == "__main__":
    create_dataset_experiment()
    plot_dataset_experiment()
    plot_nonempty_rocs()
