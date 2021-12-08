import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from os.path import exists

from utils import euclidean
from datasets.dataloading import load_dataset
from datasets.utils import windowing
from experiments import load_models, min_distance_drifts
from compositors import OS_PGSM
from warnings import simplefilter

simplefilter(action="ignore", category=UserWarning)

colors = {
    "roc_our": "b",
    "roc_random": "g",
    "topm_selection": "r",
    "data": "k",
    "forecast": "r",
}

def ambiguity(arr):
    mean = np.mean(arr, axis=0)
    return np.sum((arr - mean)**2)

# Helper function for plot designing
def generate_synthetic_rocs(n, return_y=False, random_state=0):
    rng = np.random.RandomState(random_state)
    length = 5
    roc = np.zeros((n, 5))
    ys = np.zeros((n, 1))
    start_values = rng.uniform(-0.5, 0.5, size=n)
    for i in range(n):
        # Random walk for example data
        r = np.zeros((length))
        r[0] = start_values[i]
        for j in range(1, length):
            r[j] = r[j-1] + rng.uniform(-0.2, 0.2, size=1)

        ys[i] = r[-1] + rng.uniform(-0.2, 0.2, size=1)
        roc[i] = r

    if return_y:
        return roc, ys
    return roc

def get_data_plot_2():
    # Needed:
    #     x (1, 5)
    #     y (1, 1)
    #     rocs from our method (k, 5)
    ds_name = "AbnormalHeartbeat"
    ds_index = 0
    X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
    models, _ = load_models(ds_name, ds_index, return_names=True)

    [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)

    # Load data for plotting. Save intermediate results to speed up replotting
    if exists("results/plot_1_our_data.p"):
        our_data = pickle.load(open("results/plot_1_our_data.p", "rb"))
    else:
        compositor = OS_PGSM(models, min_distance_drifts(nr_clusters_ensemble=10)) 
        preds = compositor.run(X_val, X_test)

        our_data = {
            "clustered_rocs": compositor.clustered_rocs,
            "topm_selection": compositor.topm_selection,
            "preds": preds
        }
        pickle.dump(our_data, open("results/plot_1_our_data.p", "wb"))

    # Which timestep to visualize
    start_index = 50
    x = X_test[start_index:(start_index+5)].numpy()
    y = X_test[(start_index+5)].numpy()

    # Find used clusters and selections
    for (iteration, topm_selection, topm_rocs) in our_data["topm_selection"]:
        if iteration <= start_index:
            our_selection_rocs = topm_rocs

    our_rocs = torch.cat([r.reshape(1, -1) for r in our_selection_rocs], axis=0).numpy()
    prediction = our_data["preds"][(start_index+5)]
    return x.reshape(1, -1), y.reshape(1, -1), prediction.reshape(1, -1), our_rocs

def get_data_plot_1():
    # Needed:  
    #     RoCs from our method (k, 5)
    #     RoCs from random cluster (k, 5)
    #     TopM selection from our method (binary) (k,1)
    #     TopM selection from random (binary) (k,1)
    ds_name = "AbnormalHeartbeat"
    ds_index = 0
    X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()
    models, _ = load_models(ds_name, ds_index, return_names=True)

    [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)

    # Load data for plotting. Save intermediate results to speed up replotting
    if exists("results/plot_1_our_data.p"):
        our_data = pickle.load(open("results/plot_1_our_data.p", "rb"))
    else:
        compositor = OS_PGSM(models, min_distance_drifts(nr_clusters_ensemble=10)) 
        preds = compositor.run(X_val, X_test)

        our_data = {
            "clustered_rocs": compositor.clustered_rocs,
            "topm_selection": compositor.topm_selection,
            "preds": preds
        }
        pickle.dump(our_data, open("results/plot_1_our_data.p", "wb"))

    if exists("results/plot_1_random_data.p"):
        random_data = pickle.load(open("results/plot_1_random_data.p", "rb"))
    else:
        compositor = OS_PGSM(models, min_distance_drifts(nr_clusters_ensemble=10, skip_topm=True, nr_select=6)) 
        preds = compositor.run(X_val, X_test)

        random_data = {
            "clustered_rocs": compositor.clustered_rocs,
            "topm_selection": compositor.topm_selection,
            "preds": preds
        }
        pickle.dump(random_data, open("results/plot_1_random_data.p", "wb"))

    # Which timestep to visualize
    start_index = 50
    x = X_test[start_index:(start_index+5)].numpy()

    # Find used clusters and selections
    for (iteration, c_selection, c_rocs) in our_data["clustered_rocs"]:
        if iteration <= start_index:
            our_clusters_selection = c_selection
            our_clusters_rocs = c_rocs
    for (iteration, topm_selection, topm_rocs) in our_data["topm_selection"]:
        if iteration <= start_index:
            our_selection_models = topm_selection
            our_selection_rocs = topm_rocs

    for (iteration, c_selection, c_rocs) in random_data["clustered_rocs"]:
        if iteration <= start_index:
            random_clusters_selection = c_selection
            random_clusters_rocs = c_rocs
    for (iteration, topm_selection, topm_rocs) in random_data["topm_selection"]:
        if iteration <= start_index:
            random_selection_models = topm_selection
            random_selection_rocs = topm_rocs

    our_selection = torch.cat([r.reshape(1, -1) for r in our_clusters_rocs], axis=0).numpy()
    our_selection_mask = np.zeros((10), dtype=np.bool8)
    for i in range(10):
        our_selection_mask[i] = (our_clusters_selection[i] in our_selection_models)

    random_selection = torch.cat([r.reshape(1, -1) for r in random_clusters_rocs], axis=0).numpy()
    random_selection_mask = np.zeros((10), dtype=np.bool8)
    for i in range(10):
        random_selection_mask[i] = (random_clusters_selection[i] in random_selection_models)

    return x, our_selection, random_selection, our_selection_mask.squeeze(), random_selection_mask.squeeze()

def plot_1(x, our_selection, random_selection, our_selection_mask, random_selection_mask, size=(2,10)):
    our_ambiguity = ambiguity(our_selection)
    random_ambiguity = ambiguity(random_selection)

    rows = size[0]
    cols = size[1]

    fig, axs = plt.subplots(rows, cols+1, figsize=(12, 3))
    plt.suptitle(f"OEP-ROC-10 (top row): $amb={our_ambiguity:.2f}$ \t OEP-ROC-10-topm-6 (bottom row) $amb={random_ambiguity:.2f}$")
    our_i = 0
    random_i = 0
    for row in range(rows):
        for col in range(cols):

            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
            if row < rows//2:
                axs[row][col].plot(our_selection[our_i], c=colors["roc_our"])
                d = euclidean(x, our_selection[our_i])
                if our_selection_mask[our_i]:
                    for spine in axs[row][col].spines.values():
                        spine.set_edgecolor(colors["topm_selection"])
                        spine.set_linewidth(2)
                our_i += 1
            if row >= rows//2:
                axs[row][col].plot(random_selection[random_i], c=colors["roc_random"])
                d = euclidean(x, random_selection[random_i])
                if random_selection_mask[random_i]:
                    for spine in axs[row][col].spines.values():
                        spine.set_edgecolor(colors["topm_selection"])
                        spine.set_linewidth(2)
                random_i += 1
            axs[row][col].set_title(f"$d={d:.2f}$")
        axs[row][cols].set_xticks([])
        axs[row][cols].set_yticks([])
        axs[row][cols].set_title(f"input pattern")
        axs[row][cols].plot(x, c=colors["data"])

    plt.tight_layout()
    plt.savefig("plots/plot_1.pdf")

def plot_2(x, y, pred, rocs):
    complete = np.concatenate([x, y], axis=1).squeeze()
    fig, axs = plt.subplots(1, 2, figsize=(8, 2))
    axs[0].plot(complete, c=colors["forecast"])
    axs[0].plot(np.concatenate([x, pred], axis=1).squeeze(), c="green")
    x = x.squeeze()
    axs[0].plot(x, c=colors["data"])
    axs[0].set_xticks([])
    for r in rocs:
        axs[1].plot(r, c=colors["roc_our"], alpha=0.3)
    mean_roc = np.mean(rocs, axis=0)
    axs[1].plot(mean_roc, c=colors["roc_our"])
    axs[1].set_xticks([])

    plt.tight_layout()
    plt.savefig("plots/plot_2.pdf")

if __name__ == "__main__":
    plot_1(*get_data_plot_1())
    plot_2(*get_data_plot_2())