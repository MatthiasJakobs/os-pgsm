import numpy as np
import glob
from tsx.metrics import smape
from experiments import test_keys, val_keys

def simple_test_ranking(path):
    results = np.genfromtxt(path, delimiter=",")

    baseline_index = len(val_keys)+1
    model_indices = list(range(baseline_index, len(test_keys)))
    baseline = np.squeeze(results[:, baseline_index])
    y = np.squeeze(results[:, 0])

    baseline_smape = smape(baseline, y)

    print("---- {} -----".format(path))
    print("Baseline", baseline_smape)
    better_than_baseline = np.zeros(len(model_indices))
    for idx in model_indices:
        pred = np.squeeze(results[:, idx])
        pred_smape = smape(pred, y)
        if pred_smape < baseline_smape:
            print("-- BETTER --",test_keys[idx], pred_smape)
            better_than_baseline[idx-model_indices[0]] = 1
        else:
            print(test_keys[idx], pred_smape)

    return better_than_baseline.astype(np.bool8)

def main():
    # Run all "_test.csv" files and see if one of our methods is better than the baseline

    paths = glob.glob("results/*_test.csv")
    better = []

    for p in paths:
        #print(p)
        pred = simple_test_ranking(p)
        better.append(pred)
        print()

    better = np.array(better)
    print(np.mean(better, axis=0))

    #print("Better: {} ({}%)".format(better, float(better)/len(paths)*100.0))

if __name__ == "__main__":
    main()