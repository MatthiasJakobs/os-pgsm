import numpy as np
import pandas as pd

from experiments import * 
from compositors import *
from datasets.utils import windowing, train_test_split, _apply_window, sliding_split, _val_split
from run_prediction import _get_m4_ds

ds_subset = [
    "AbnormalHeartbeat",
    "bike_total_rents",
    "CatsDogs",
    "NYSE",
    "Rock",
    "CloudCoverage",
    "m4_hourly_H1",
    "m4_hourly_H10",
    "m4_hourly_H12",
    "m4_hourly_H3",
]

def main():
    for lag in [5, 10, 15]:
        for ds_full_name in ds_subset:
            models = []
            for m_name in single_models.keys():
                m = load_model(m_name, ds_full_name, lag, lag)
                if type(m) not in skip_models_composit:
                    models.append(m)

            if ds_full_name.startswith("m4"):
                ds = _get_m4_ds(ds_full_name)()
                designator = ds_full_name.split("_")[-1]
                X_train, X_test = ds.get(designator)
                X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

                if len(X_val) <= lag_mapping[str(lag)]:
                    print(designator, "is to short, skipping")
                    #continue

                x_val_big = _val_split(X_val, lag, lag_mapping[str(lag)], use_torch=True)
                x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
                x_val_small = x_val_small.unsqueeze(1)

                X_test = X_test.float()
                X_val = X_val.float()
            else:
                ds = implemented_datasets[ds_full_name]['ds']()

                X = ds.torch()
                [x_train, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

            df = pd.DataFrame(index=comp_names, columns=["total_runtime", "per_mean", "per_std"])

            for c_name, G in zip(comp_names, comps):
                gc = G(models, lag, lag_mapping[str(lag)])
                if isinstance(gc, BaseAdaptive):
                    _, total_runtime, runtimes = gc.run(X_val, X_test, report_runtime=True)
                else:
                    _, total_runtime, runtimes = gc.run(x_val_big, X_test, report_runtime=True)

                per_mean = np.mean(runtimes)
                per_std = np.std(runtimes)
                df.loc[c_name]['total_runtime'] = total_runtime
                df.loc[c_name]['per_mean'] = per_mean
                df.loc[c_name]['per_std'] = per_std

            result_name = "results/runtimes/{}_lag{}.csv".format(ds_full_name, lag)
            df.to_csv(result_name, index_label="comp")

if __name__ == "__main__":
    main()