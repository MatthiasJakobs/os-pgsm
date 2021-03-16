import torch
import numpy as np
import argparse

from main import evaluate_test
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Temperature, Bike_Registered, M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import GCCompositor, BaselineCompositor, GC_EvenCompositor, GC_EuclidianComparison
from datasets.utils import windowing, train_test_split, _apply_window, sliding_split, _val_split
from collections import defaultdict
from csv import DictReader

from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02

val_keys = ['y','pred_rnn','pred_cnn','pred_as01','pred_as02']
test_keys = ['y','pred_rnn','pred_cnn','pred_as01','pred_as02', 'pred_baseline', 'pred_gradcam_small', 'pred_gradcam_large', 'pred_gradcam_euclid']


def run_comparison(models, x_val_small, x_val_big, X_val, X_test, ds_name, lag):
    val_npy = np.zeros((len(X_val), len(val_keys)))
    test_npy = np.zeros((len(X_test), len(test_keys)))

    val_npy[:, 0] = np.squeeze(X_val.numpy())
    test_npy[:, 0] = np.squeeze(X_test.numpy())

    for i, m in enumerate(models):
        preds_test, _ = evaluate_test(m, X_test, reuse_predictions=False, lag=lag)
        x_val_small, _ = sliding_split(X_val, lag, use_torch=True)
        preds_val = m.predict(x_val_small.unsqueeze(1).float())
        preds_val = np.concatenate([np.squeeze(x_val_small[0]), preds_val])
        val_npy[:, i+1] = preds_val
        test_npy[:, i+1] = preds_test

    baseline_comp = BaselineCompositor(models, lag)
    preds_base, _ = baseline_comp.run(x_val_big, X_test)

    our_comp = GCCompositor(models, lag)
    preds_o, _ = our_comp.run(x_val_big, X_test)

    our_comp = GC_EvenCompositor(models, lag)
    preds_even, _ = our_comp.run(x_val_big, X_test)

    our_comp = GC_EuclidianComparison(models, lag)
    preds_euclid, _ = our_comp.run(x_val_big, X_test)

    test_npy[:, 5] = np.squeeze(preds_base)
    test_npy[:, 6] = np.squeeze(preds_o)
    test_npy[:, 7] = np.squeeze(preds_even)
    test_npy[:, 8] = np.squeeze(preds_euclid)

    np.savetxt("results/{}_test.csv".format(ds_name), test_npy, header="y,pred_rnn,pred_cnn,pred_as01,pred_as02,pred_baseline,pred_gradcam_large,pred_gradcam_small,pred_gradcam_euclidian", delimiter=",")
    np.savetxt("results/{}_val.csv".format(ds_name), val_npy, header="y,pred_rnn,pred_cnn,pred_as01,pred_as02", delimiter=",")

# Took out Jena (because it takes ages)
# datasets = [ M4_Hourly(), M4_Weekly(), M4_Quaterly(), M4_Daily(), M4_Monthly(), Jena_Climate(), Bike_Total_Rents(), Bike_Temperature(), Bike_Registered() ]
# dataset_names = [ 'm4_hourly', 'm4_weekly', 'm4_quaterly', 'm4_daily', 'm4_monthly', 'jena', 'bike_total_rents', 'bike_temperature', 'bike_registered' ]

def main():
    datasets = [ M4_Hourly(), M4_Weekly(), M4_Quaterly(), M4_Daily(), M4_Monthly(), Bike_Total_Rents(), Bike_Temperature(), Bike_Registered() ]
    dataset_names = [ 'm4_hourly', 'm4_weekly', 'm4_quaterly', 'm4_daily', 'm4_monthly', 'bike_total_rents', 'bike_temperature', 'bike_registered' ]
    comps = [BaselineCompositor, GCCompositor, GC_EvenCompositor, GC_EuclidianComparison]
    comp_names = ['baseline', 'gradcam_large', 'gradcam_small', 'gradcam_euclidian']
    model_names = ['rnn', 'cnn', 'as01', 'as02']

    idx_range = list(range(17))
    lag = 5

    model_a = Shallow_CNN_RNN(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_b = Shallow_FCN(batch_size=50, ts_length=lag, epochs=None)
    model_c = AS_LSTM_01(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_d = AS_LSTM_02(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)

    for ds_name, ds in zip(dataset_names, datasets):
        if ds_name.startswith("m4"):
            keys = ds.train_data.columns
            for idx in idx_range:
                designator = keys[idx]
                ds_full_name = "{}_{}".format(ds_name, designator)
                X_train, X_test = ds.get(designator)
                X_train, X_val = train_test_split(X_train, split_percentages=(2.0/3.0, 1.0/3.0))

                if len(X_val) < lag*lag:
                    continue
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

                run_comparison(models, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)
                # try:
                #     run_comparison(models, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)
                # except Exception as e:
                #     print(e)
                #     continue

        else:
            ds_full_name = ds_name

            X = ds.torch()
            [_, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)

            model_a.load_state_dict(torch.load("models/{}_rnn.pth".format(ds_full_name)))
            model_b.load_state_dict(torch.load("models/{}_cnn.pth".format(ds_full_name)))
            model_c.load_state_dict(torch.load("models/{}_as01.pth".format(ds_full_name)))
            model_d.load_state_dict(torch.load("models/{}_as02.pth".format(ds_full_name)))

            models = [model_a, model_b, model_c, model_d]
        
            run_comparison(models, x_val_small, x_val_big, X_val, X_test, ds_full_name, lag)

if __name__ == "__main__":
    main()
