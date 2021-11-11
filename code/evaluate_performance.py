from matplotlib.pyplot import draw
import numpy as np
import pandas as pd
import glob

from os.path import exists
from sklearn.metrics import mean_squared_error

from experiments import implemented_datasets
from critical_difference import draw_cd_diagram

def main():

    performance_df = pd.DataFrame(columns=["classifier_name", "dataset_name", "accuracy"])
    loss = mean_squared_error
    dataset_names = list(implemented_datasets.keys())

    for d_name in dataset_names:
        if not exists(f"results/lag5/{d_name}/y_test.csv"):
            continue
        y_test = np.genfromtxt(f"results/lag5/{d_name}/y_test.csv").squeeze()
        results_path = f"results/lag5/{d_name}/"
        if not exists(results_path):
            #print(f"{d_name} folder empty")
            continue

        for full_file_name in glob.glob(results_path + "*_test.csv"):
            if "y_test" in full_file_name:
                continue

            if not "ospgsm" in full_file_name or "min_distance" in full_file_name:
                continue

            file_name = full_file_name.split("/")[-1]
            model_name = file_name.replace("_test.csv", "")

            preds = np.genfromtxt(full_file_name).squeeze()
            prediction_loss = loss(y_test, preds)

            performance_df = performance_df.append({
                "classifier_name": model_name,
                "dataset_name": d_name,
                "accuracy": prediction_loss
            }, ignore_index=True)

    print(performance_df)
    draw_cd_diagram("plots/evaluate_performance.png", df_perf=performance_df)

if __name__ == "__main__":
    main()