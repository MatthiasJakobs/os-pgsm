from matplotlib.pyplot import draw
import numpy as np
import pandas as pd
import glob

from os.path import exists
from sklearn.metrics import mean_squared_error

from critical_difference import draw_cd_diagram
from datasets.dataloading import implemented_datasets

def calc_average_ranks(df):
    models = df["classifier_name"].unique()
    datasets = df["dataset_name"].unique()

    ranks = np.zeros((len(models)))
    avg_counter = 0

    for dataset in datasets:
        subset = df[df["dataset_name"] == dataset]

        if len(subset) < len(models):
            print(f"{dataset} only contains {len(subset)} models instead of {len(models)}")
            continue
        
        # Lower rank is better
        ranks += subset["accuracy"].rank().to_numpy()
        avg_counter += 1

    ranks = ranks / avg_counter

    sorted_by_rank = pd.DataFrame(data={'name': models, 'avg_rank': ranks})
    print("-"*30)
    print("AVG Ranks - lower is better")
    print(sorted_by_rank.sort_values(by=['avg_rank'], ascending=False).to_string(index=False))

def main():

    performance_df = pd.DataFrame(columns=["classifier_name", "dataset_name", "accuracy"])
    loss = mean_squared_error

    for d_name, idx in implemented_datasets:
        if not exists(f"results/{d_name}/{idx}_y_test.csv"):
            continue
        y_test = np.genfromtxt(f"results/{d_name}/{idx}_y_test.csv").squeeze()
        results_path = f"results/{d_name}/"
        if not exists(results_path):
            #print(f"{d_name} folder empty")
            continue

        for full_file_name in glob.glob(results_path + f"{idx}_*_test.csv"):
            if "y_test" in full_file_name:
                continue

            if not "ospgsm" in full_file_name and not "min_distance" in full_file_name:
                continue

            file_name = full_file_name.split("/")[-1]
            model_name = file_name.replace("_test.csv", "")[2:]

            preds = np.genfromtxt(full_file_name).squeeze()
            prediction_loss = loss(y_test, preds)

            performance_df = performance_df.append({
                "classifier_name": model_name,
                "dataset_name": d_name + "_" + str(idx),
                "accuracy": prediction_loss
            }, ignore_index=True)

    print(performance_df)
    draw_cd_diagram("plots/evaluate_performance.png", df_perf=performance_df)
    calc_average_ranks(performance_df)

if __name__ == "__main__":
    main()
