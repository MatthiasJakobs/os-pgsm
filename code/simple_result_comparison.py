import argparse
import numpy as np

from experiments import test_keys, val_keys
from tsx.metrics import smape
from sklearn.metrics import mean_squared_error

def main(lag, ds_name, metric="smape"):
    best_results = np.genfromtxt("results/{}_lag{}_best_test.csv".format(ds_name, lag), delimiter=",")
    avg_results = np.genfromtxt("results/{}_lag{}_avg_test.csv".format(ds_name, lag), delimiter=",")

    offset = len(val_keys)

    model_names = test_keys[offset:]

    y = best_results[:, 0]

    if metric == "smape":
        metric = smape
    else:
        metric = lambda x1, x2: mean_squared_error(x1, x2, squared=False)

    qualities_best = np.zeros(len(model_names))
    qualities_avg = np.zeros(len(model_names))

    for i in range(len(model_names)):
        a = avg_results[:, i+offset]
        b = best_results[:, i+offset]

        qualities_avg[i] = metric(a, y)
        qualities_best[i] = metric(b, y)

    smallest_error_best = model_names[np.argmin(qualities_best)]
    smallest_error_avg = model_names[np.argmin(qualities_avg)]

    print("--- {} ---".format(ds_name))
    print("model | " + " | ".join([w[5:] for w in model_names]))
    print("-"*60)
    print("best  | {}".format(" | ".join([str(np.round(m, 4)) for m in qualities_best])))
    print("avg   | {}".format(" | ".join([str(np.round(s, 4)) for s in qualities_avg])))
    print("-"*60)
    print("smallest error best", smallest_error_best)
    print("smallest error avg", smallest_error_avg)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", help="", type=str)
    parser.add_argument("--lag", action="store", help="", type=int)
    parser.add_argument("--metric", action="store", help="", default="smape", type=str)
    args = parser.parse_args()
    main(args.lag, args.dataset, metric=args.metric)