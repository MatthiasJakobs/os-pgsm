from os import error
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

''' Compare performance between traditional and inverted method '''
warnings.filterwarnings("ignore")

def main():
    df = pd.DataFrame(columns=["classifier_name", "dataset_name", "accuracy"])
    for ds_name in list(implemented_datasets.keys()):
        lag = 5
        ts_length = lag 

        ds = implemented_datasets[ds_name]['ds']()
        model_names = single_models.keys()

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

            zero_length_rocs = np.sum(roc_lengths == 0)
            other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

            df = df.append({"classifier_name": "original-int", "dataset_name": ds_name, "accuracy": error_comp}, ignore_index=True)

            print(f"--- {ds_name} ---")
            print(f"Original-int zero rocs: {zero_length_rocs / len(new_comp.rocs)}")
            print(f"Original-int non-zero rocs: {other_length_rocs}")
            print(f"Original-int MSE {error_comp}")

            ################ Inverted - Int ###############3
            ospgsm_int_config = ospgsm_int_original()
            torch.manual_seed(0)
            new_comp = Inv_OS_PGSM(models, ospgsm_int_config, random_state=1010)
            preds = new_comp.run(X_val, X_test)
            error_comp = mean_squared_error(X_test, preds)
            roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

            zero_length_rocs = np.sum(roc_lengths == 0)
            other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

            df = df.append({"classifier_name": "inverted-int", "dataset_name": ds_name, "accuracy": error_comp}, ignore_index=True)

            print(f"Inverted-int zero rocs: {zero_length_rocs / len(new_comp.rocs)}")
            print(f"Inverted-int non-zero rocs: {other_length_rocs}")
            print(f"Inverted-int MSE {error_comp}")

            ################ Original - St ###############3
            ospgsm_config = ospgsm_st_original()
            torch.manual_seed(0)
            new_comp = OS_PGSM(models, ospgsm_config, random_state=1010)
            original_preds = new_comp.run(X_val, X_test)
            error_comp = mean_squared_error(X_test, original_preds)
            roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

            zero_length_rocs = np.sum(roc_lengths == 0)
            other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

            df = df.append({"classifier_name": "original-st", "dataset_name": ds_name, "accuracy": error_comp}, ignore_index=True)

            print(f"Original-st zero rocs: {zero_length_rocs / len(new_comp.rocs)}")
            print(f"Original-st non-zero rocs: {other_length_rocs}")
            print(f"Original-st MSE {error_comp}")

            ################ Inverted - st ###############3
            ospgsm_config = ospgsm_st_original()
            torch.manual_seed(0)
            new_comp = Inv_OS_PGSM(models, ospgsm_config, random_state=1010)
            preds = new_comp.run(X_val, X_test)
            error_comp = mean_squared_error(X_test, preds)
            roc_lengths = np.array([len(roc) for roc in new_comp.rocs]).squeeze()

            zero_length_rocs = np.sum(roc_lengths == 0)
            other_length_rocs = roc_lengths[np.nonzero(roc_lengths)]

            df = df.append({"classifier_name": "inverted-st", "dataset_name": ds_name, "accuracy": error_comp}, ignore_index=True)

            print(f"Inverted-st zero rocs: {zero_length_rocs / len(new_comp.rocs)}")
            print(f"Inverted-st non-zero rocs: {other_length_rocs}")
            print(f"Inverted-st MSE {error_comp}")
        except Exception:
            continue

    print(df)
    df.to_csv("results/cd-inverted.csv")
    draw_cd_diagram("test-ci.png", df_perf=df, title="Test")

if __name__ == "__main__":
    main()