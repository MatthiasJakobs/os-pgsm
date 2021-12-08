from typing_extensions import runtime
import torch
import pickle
import numpy as np

from datasets.dataloading import load_dataset
from datasets.utils import windowing
from experiments import load_models, min_distance_drifts
from compositors import OS_PGSM
from os.path import exists
from warnings import simplefilter

simplefilter(action="ignore", category=UserWarning)
used_datasets = [
    ("electricity_hourly", 0),
    ("Phoneme", 0),
    ("kdd_cup_2018", 0),
]

used_experiments = [
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-5", nr_clusters_ensemble=5)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-15")),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-10", nr_clusters_ensemble=10)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-20", nr_clusters_ensemble=20)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-C", skip_topm=True)),
    (OS_PGSM, min_distance_drifts(name="OEP-ROC-TOP", skip_clustering=True)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-II", skip_type1=True)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-I", skip_type2=True)),
    (OS_PGSM, min_distance_drifts(name="OEP-ROC-ST", skip_drift_detection=True)),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-Per", concept_drift_detection="periodic")),
    # (OS_PGSM, min_distance_drifts(name="OEP-ROC-15-topm-10", nr_select=10, skip_topm=True)),
]

def measure_runtime():
    runtime_dict = dict()
    for ds_name, ds_index in used_datasets:
        X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
        models, model_names = load_models(ds_name, ds_index, return_names=True)

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)

        for comp_class, exp_config in used_experiments:
            config_name = exp_config["name"]
            print(ds_name, ds_index, config_name)
            compositor = comp_class(models, exp_config)
            _, runtime = compositor.run(X_val, X_test, measure_runtime=True)

            try:
                runtime_dict[config_name].append(runtime)
            except Exception:
                runtime_dict[config_name] = [runtime]
    return runtime_dict

if __name__ == "__main__":

    if exists("results/runtimes.p"):
        runtime_dict = pickle.load(open("results/runtimes.p", "rb"))
    else:
        runtime_dict = measure_runtime()
        pickle.dump(runtime_dict, open("results/runtimes.p", "wb"))

    for config_name, runtimes in runtime_dict.items():
        print(f"{config_name}\t{np.mean(runtimes):.2f}\t{np.std(runtimes):.2f}")
