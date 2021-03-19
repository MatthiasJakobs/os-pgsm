from datasets.m4 import M4_Monthly
from compositors import *
from datasets import Bike_Total_Rents
from datasets.utils import *
from datasets.utils import _val_split, sliding_split, train_test_split
from tsx.models.forecaster import *
from tsx.metrics import smape
from matplotlib import cm

from os.path import exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    lambs = np.arange(0.05, 1, 0.05)
    deltas = np.arange(0.05, 1, 0.05)

    path = "results/page-hinkley-gridsearch.csv"
    if exists(path):
        results = pd.read_csv(path, delimiter=',')
        plt.figure(figsize=(10, 6))
        for lamb in lambs:
            subset = results[results['lambda'] == lamb]
            x = subset['delta']
            y = subset['smape']
            plt.plot(x,y, color=plt.get_cmap('winter')(int(255.0*lamb)))
        plt.title("Gridsearch Page-Hinkley m4_monthly_M4")
        plt.xlabel("delta")
        plt.ylabel("smape")
        #plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ph_gridsearch/smape.pdf")

        plt.figure(figsize=(10,6))
        for lamb in lambs:
            subset = results[results['lambda'] == lamb]
            x = subset['delta']
            y = subset['nr_changes']
            plt.plot(x,y, label="lambda={}".format(lamb))
        plt.title("Gridsearch Page-Hinkley m4_monthly_M4")
        plt.xlabel("delta")
        plt.ylabel("nr_changes")
        #plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ph_gridsearch/changes.pdf")

    else:
        ds = M4_Monthly()
        ds_full_name = "m4_monthly_M12"

        lag = 5

        model_a = Shallow_CNN_RNN(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
        model_b = Shallow_FCN(batch_size=50, ts_length=lag, epochs=None)
        model_c = AS_LSTM_01(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
        model_d = AS_LSTM_02(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)

        # X = ds.torch()
        # [_, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)
        X_train, X_test = ds.get("M12")
        X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

        x_val_big = _val_split(X_val, lag, lag*lag, use_torch=True)
        x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
        x_val_small = x_val_small.unsqueeze(1)

        X_test = X_test.float()
        X_val = X_val.float()

        model_a.load_state_dict(torch.load("models/{}_rnn.pth".format(ds_full_name)))
        model_b.load_state_dict(torch.load("models/{}_cnn.pth".format(ds_full_name)))
        model_c.load_state_dict(torch.load("models/{}_as01.pth".format(ds_full_name)))
        model_d.load_state_dict(torch.load("models/{}_as02.pth".format(ds_full_name)))

        models = [model_a, model_b, model_c, model_d]

        baseline_comp = BaselineCompositor(models, lag)
        preds_base, smape_base = baseline_comp.run(x_val_big, X_test)
        print("Baseline", smape_base)

        results = np.zeros((len(lambs)*len(deltas), 4))
        idx = 0
        for lamb in lambs:
            for delta in deltas:
                our_comp = GC_Big_Adaptive_PageHinkley(models, lag, lamb=lamb, delta=delta)
                preds_o = our_comp.run(X_val, X_test)
                smp = smape(np.squeeze(preds_o), np.squeeze(X_test.numpy()))
                drifts = len(our_comp.drifts_detected)

                results[idx, 0] = lamb
                results[idx, 1] = delta
                results[idx, 2] = smp
                results[idx, 3] = drifts
                print(idx, len(lambs) * len(deltas))
                idx += 1

        np.savetxt("results/page-hinkley-gridsearch.csv", results, comments='', delimiter=',', header="lambda,delta,smape,nr_changes")
