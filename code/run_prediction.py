import torch
import numpy as np

from main import evaluate_test
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Temperature, Bike_Registered, M4_Daily, M4_Hourly, M4_Monthly, M4_Quaterly, M4_Quaterly, M4_Weekly
from compositors import GCCompositor, BaselineCompositor, GC_EvenCompositor
from datasets.utils import windowing, train_test_split, _apply_window

from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02

def run_comparison(models, x_val_big, x_test, ds_name):

    result_string_base = "results/pred_{}_".format(ds_name)

    baseline_comp = BaselineCompositor(models, 5)
    preds_base, _ = baseline_comp.run(x_val_big, x_test)
    np.savetxt("{}{}.csv".format(result_string_base, "baseline"), np.squeeze(preds_base), delimiter=",")

    our_comp = GCCompositor(models, 5)
    preds_o, _ = our_comp.run(x_val_big, x_test)
    np.savetxt("{}{}.csv".format(result_string_base, "our_large"), np.squeeze(preds_o), delimiter=",")

    our_comp = GC_EvenCompositor(models, 5)
    preds_even, _ = our_comp.run(x_val_big, x_test)
    np.savetxt("{}{}.csv".format(result_string_base, "our_small"), np.squeeze(preds_even), delimiter=",")

datasets = [ M4_Hourly(), M4_Weekly(), M4_Quaterly(), M4_Daily(), M4_Monthly(), Jena_Climate(), Bike_Total_Rents(), Bike_Temperature(), Bike_Registered() ]
dataset_names = [ 'm4_hourly', 'm4_weekly', 'm4_quaterly', 'm4_daily', 'm4_monthly', 'jena', 'bike_total_rents', 'bike_temperature', 'bike_registered' ]
comps = [BaselineCompositor, GCCompositor, GC_EvenCompositor]
comp_names = ['baseline', 'gradcam_large', 'gradcam_small']

idx_range = list(range(17))
lag = 5

for ds_name, ds in zip(dataset_names, datasets):
    model_a = Shallow_CNN_RNN(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_b = Shallow_FCN(batch_size=50, ts_length=lag, epochs=None)
    model_c = AS_LSTM_01(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_d = AS_LSTM_02(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)

    if ds_name.startswith("m4"):
        keys = ds.train_data.columns
        for idx in idx_range:
            designator = keys[idx]
            model_a.load_state_dict(torch.load("models/{}_{}_rnn.pth".format(ds_name, designator)))
            model_b.load_state_dict(torch.load("models/{}_{}_cnn.pth".format(ds_name, designator)))
            model_c.load_state_dict(torch.load("models/{}_{}_as01.pth".format(ds_name, designator)))
            model_d.load_state_dict(torch.load("models/{}_{}_as02.pth".format(ds_name, designator)))

            models = [model_a, model_b, model_c, model_d]

            X_train, X_test = ds.get(designator)
            X_train, X_val = train_test_split(X_train, split_percentages=(0.75, 0.25))
            # TODO: Make this correct
            try:
                x_val = []

                for i in range(0, len(X_val), lag*lag):
                    x_val.append(X_val[i:(i+lag*lag)].unsqueeze(0))

                x_val = torch.cat(x_val[:-1], 0)

                X_test = X_test.float()
                x_val = x_val.float()
                np.savetxt("results/y_val_{}_{}.csv".format(ds_name, designator), np.squeeze(X_test.numpy()), delimiter=",")
                for m, m_name in zip(models, ['rnn', 'cnn', 'as01', 'as02']):
                    preds_a, _ = evaluate_test(m, X_test, reuse_predictions=False, lag=lag)
                    np.savetxt("results/val_{}_{}_{}.csv".format(ds_name, designator, m_name), np.squeeze(preds_a), delimiter=",")

                np.savetxt("results/y_{}_{}.csv".format(ds_name, designator), np.squeeze(X_test.numpy()), delimiter=",")
                run_comparison(models, x_val, X_test, "{}_{}".format(ds_name, designator))
            except Exception as e:
                print(e)
                continue
    else:
        model_a.load_state_dict(torch.load("models/{}_rnn.pth".format(ds_name)))
        model_b.load_state_dict(torch.load("models/{}_cnn.pth".format(ds_name)))
        model_c.load_state_dict(torch.load("models/{}_as01.pth".format(ds_name)))
        model_d.load_state_dict(torch.load("models/{}_as02.pth".format(ds_name)))
        models = [model_a, model_b, model_c, model_d]
        X = ds.torch()
        [x_train, x_val], [y_train, y_val], x_val_big, x_test = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)
        np.savetxt("results/y_val_{}.csv".format(ds_name), np.squeeze(x_test), delimiter=",")
        for m, m_name in zip(models, ['rnn', 'cnn', 'as01', 'as02']):
            preds_a, _ = evaluate_test(m, x_test, reuse_predictions=False, lag=lag)
            np.savetxt("results/val_{}_{}.csv".format(ds_name, m_name), np.squeeze(preds_a), delimiter=",")

        np.savetxt("results/y_{}.csv".format(ds_name), np.squeeze(x_test.numpy()), delimiter=",")
        run_comparison(models, x_val_big, x_test, ds_name)
