import numpy as np
import matplotlib.pyplot as plt

from utils import euclidean

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

def plot_1(x, our_selection, random_selection, our_selection_mask, random_selection_mask, size=(2,10)):
    our_ambiguity = ambiguity(our_selection)
    random_ambiguity = ambiguity(random_selection)

    rows = size[0]
    cols = size[1]

    fig, axs = plt.subplots(rows, cols+1, figsize=(12, 3))
    plt.suptitle(f"OEP-ROC (blue): $amb={our_ambiguity:.2f}$ \t Random selection (red) $amb={random_ambiguity:.2f}$")
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
    plt.savefig("plots/test1.png")

def plot_2(x, y, rocs):
    complete = np.concatenate([x, y], axis=1).squeeze()
    x = x.squeeze()
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(complete, c=colors["forecast"])
    axs[0].plot(x, c=colors["data"])
    for r in rocs:
        axs[1].plot(r, c=colors["roc_our"], alpha=0.3)
    mean_roc = np.mean(rocs, axis=0)
    print(mean_roc.shape)
    axs[1].plot(mean_roc, c=colors["roc_our"])

    plt.tight_layout()
    plt.savefig("plots/test2.png")


if __name__ == "__main__":
    n_clusters = 10
    our_method = generate_synthetic_rocs(n_clusters, random_state=1)
    random_cluster = generate_synthetic_rocs(n_clusters, random_state=2)
    x = generate_synthetic_rocs(1, random_state=3).squeeze()
    plot_1(x, our_method, random_cluster, np.array([True, False]*5), np.array([False, True]*5))

    x, y = generate_synthetic_rocs(1, return_y=True, random_state=51234)
    plot_2(x, y, our_method)