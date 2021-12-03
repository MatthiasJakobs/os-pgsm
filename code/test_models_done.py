import itertools
import numpy as np
import pandas as pd

df = pd.read_csv("results/large_experiment_save.csv", index_col=0)

df["k"] = df["classifier_name"].apply(lambda x: x[2:4]).to_numpy().astype(np.int8)
df["n_omega"] = df["classifier_name"].apply(lambda x: x[13:15]).to_numpy().astype(np.int8)
df["distances"] = df["classifier_name"].apply(lambda x: x[28:]).to_numpy()

# check which configs are done
distances = ["euclidean", "dtw"]
n_omegas = [30, 40, 50, 60]
ks = [13, 15, 17]

max_number_configs = len(ks) * len(n_omegas) * len(distances) * 25

for d, n_omega, k in list(itertools.product(distances, n_omegas, ks)):
    completed_experiments = df[(df.k == k) & (df.n_omega == n_omega) & (df.distances == d)]
    if len(completed_experiments) == max_number_configs:
        print(f"k={k} n_omega={n_omega} distance_fn={d}")