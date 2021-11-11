from os import error
from os.path import exists
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.utils import windowing
from compositors import OS_PGSM, Inv_OS_PGSM
from experiments import ospgsm_original, ospgsm_per_original, ospgsm_st_original, implemented_datasets, load_model, single_models, skip_models_composit, ospgsm_int_original
from sklearn.metrics import mean_squared_error
from critical_difference import draw_cd_diagram

path = "results/test_model_subset.csv"

''' See how nr of models with RoCs changes as we add more models '''
warnings.filterwarnings("ignore")

def plot_data():
    data = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    datasets = data["ds_name"].unique()

    ax1.set_xlabel("Nr of models")
    ax1.set_xticks(np.arange(13, 34, 1))
    ax1.set_ylabel("Percentage of RoCs empty")

    ax2.set_xlabel("Nr of models")
    ax2.set_xticks(np.arange(13, 34, 1))
    ax2.set_ylabel("Percentage of RoCs empty")

    for ds_name in datasets:
        subset = data[data["ds_name"] == ds_name]

        ax1.plot(subset["nr_used_models"], subset["nr_zero_rocs_st"]/subset["nr_used_models"], label=f"{ds_name}")
        ax2.plot(subset["nr_used_models"], subset["nr_zero_rocs_int"]/subset["nr_used_models"], label=f"{ds_name}")
        #ax.plot(subset["nr_used_models"], subset["nr_zero_rocs_int"]/subset["nr_used_models"], label=f"{ds_name} - Int")
        #ax.plot(subset["nr_used_models"], subset["nr_zero_rocs_st"]/subset["nr_used_models"], label=f"{ds_name} - St")

    ax1.legend()
    ax2.legend()

    ax1.set_title("OSPGSM-St")
    ax2.set_title("OSPGSM-Int")

    ax1.hlines(0.95, 13, 33, linestyles="dashed", colors="black")
    ax2.hlines(0.95, 13, 33, linestyles="dashed", colors="black")

    plt.suptitle("Percentage of RoCs empty as we add more models")
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    plt.savefig("plots/model_subsets.png")

def create_data():
    if exists(path):
        print(f"{path} exists, skip create_data")
        return
    
    original_models = [0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17]
    new_models = [i for i in list(range(len(single_models)-2)) if i not in original_models]

    lag = 5
    ts_length = lag 

    ds_names = ["AbnormalHeartbeat", "SNP500", "Rock", "Mallat", "EOGHorizontalSignal", "Cricket"]

    results = pd.DataFrame(columns=["ds_name", "nr_used_models", "nr_zero_rocs_int", "nr_zero_rocs_st"])

    for ds_name in ds_names:
        ds = implemented_datasets[ds_name]['ds']()
        
        for i in range(1, len(new_models)+1):
            used_models = original_models + new_models[:i]

            model_names = [m_name for idx, m_name in enumerate(single_models.keys()) if idx in used_models]

            X = ds.torch()
            [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=25, use_torch=True)

            models = []
            for m_name in model_names:
                m = load_model(m_name, ds_name, lag, ts_length)
                if type(m) not in skip_models_composit:
                    models.append(m)

            print("-"*30)
            try:
                ################ Original - Int ###############3
                ospgsm_int_config = ospgsm_int_original()
                torch.manual_seed(0)
                new_comp = OS_PGSM(models, ospgsm_int_config, random_state=1010)
                original_preds = new_comp.run(X_val, X_test)
                error_comp = mean_squared_error(X_test, original_preds)
                roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

                int_zero_length_rocs = np.sum(roc_lengths == 0)
                other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

                print(f"--- {ds_name} ---")
                print(f"Original-int zero rocs: {int_zero_length_rocs / len(new_comp.rocs)}")
                print(f"Original-int non-zero rocs: {other_length_rocs}")
                print(f"Original-int MSE {error_comp}")


                ################ Original - St ###############3
                ospgsm_config = ospgsm_st_original()
                torch.manual_seed(0)
                new_comp = OS_PGSM(models, ospgsm_config, random_state=1010)
                original_preds = new_comp.run(X_val, X_test)
                error_comp = mean_squared_error(X_test, original_preds)
                roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

                st_zero_length_rocs = np.sum(roc_lengths == 0)
                other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

                print(f"Original-st zero rocs: {st_zero_length_rocs / len(new_comp.rocs)}")
                print(f"Original-st non-zero rocs: {other_length_rocs}")
                print(f"Original-st MSE {error_comp}")

                results = results.append({
                    "ds_name": ds_name,
                    "nr_used_models": len(used_models),
                    "nr_zero_rocs_int": int_zero_length_rocs,
                    "nr_zero_rocs_st": st_zero_length_rocs,
                }, ignore_index=True)

            except Exception:
                continue

    print(results)
    results.to_csv(path)

if __name__ == "__main__":
    create_data()
    plot_data()