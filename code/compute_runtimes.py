import pandas as pd
import numpy as np

from meassure_runtime import ds_subset
from experiments import comp_names


lag = 15

total = np.zeros((7, 10))
per_step = np.zeros((7, 10))

for i, d_name in enumerate(ds_subset):
    df = pd.read_csv("results/runtimes/{}_lag{}.csv".format(d_name, lag), index_col=0)
    for j, c_name in enumerate(comp_names):
        total[j, i] = df.loc[c_name]['total_runtime']
        per_step[j, i] = df.loc[c_name]['per_mean']

total_mean = np.mean(total, axis=1)
total_std = np.std(total, axis=1)
per_mean = np.mean(per_step, axis=1)
per_std = np.std(per_step, axis=1)

rounding=2
print(np.round(total_mean, rounding))
print(np.round(total_std, rounding))
print(np.round(per_mean, 2*rounding))
print(np.round(per_std, 2*rounding))