import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.utils import windowing
from compositors import OS_PGSM
from experiments import ospgsm_original, ospgsm_per_original, ospgsm_st_original, implemented_datasets, load_model, single_models, skip_models_composit, ospgsm_int_original
from sklearn.metrics import mean_squared_error

''' Checking the thesis that good indiviual models do not provide as much RoC information because of ReLU in GC '''

#ds_name = "AbnormalHeartbeat"
collector = []
for ds_name in ["AbnormalHeartbeat", "bike_registered", "bike_temperature", "CatsDogs", "Cricket"]:
    lag = 5
    ts_length = lag 

    ds = implemented_datasets[ds_name]['ds']()
    model_names = single_models.keys()

    X = ds.torch()
    [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=25, use_torch=True)

    models = []
    new_model_names = []
    individual_predictions = np.zeros((len(model_names)-len(skip_models_composit), len(X_test)))
    i = 0
    for m_name in model_names:
        m = load_model(m_name, ds_name, lag, ts_length)
        if type(m) not in skip_models_composit:
            models.append(m)
            new_model_names.append(m_name)
            path = f"results/lag5/{ds_name}/{m_name}_test.csv"
            preds = np.loadtxt(path)
            individual_predictions[i] = preds
            i += 1

    print("Int-original")
    ospgsm_int_config = ospgsm_int_original()
    ospgsm_int_config["n_omega"] = 40
    ospgsm_int_config["smoothing_threshold"] = 0.5
    ospgsm_int_config["n_clusters_ensemble"] = 5
    ospgsm_int_config["topm"] = 10
    torch.manual_seed(0)
    new_comp = OS_PGSM(models, ospgsm_int_config, random_state=1010)
    original_preds = new_comp.run(X_val, X_test)
    error_comp = mean_squared_error(X_test, original_preds)

    rocs = new_comp.rocs
    for model_idx, model_name in enumerate(new_model_names):
        length_of_rocs = len(rocs[model_idx])
        mse_individual = mean_squared_error(X_test, individual_predictions[model_idx].squeeze())
        print(model_name, length_of_rocs, mse_individual)
        collector.append([mse_individual - error_comp, length_of_rocs])

    print("MSE original", error_comp)
    print("-"*20)
    # Plot error vs nr_rocs


    # print("Int-new")
    # ospgsm_int_config = ospgsm_int_original()
    # ospgsm_int_config["invert_loss"] = True
    # torch.manual_seed(0)
    # new_comp = OS_PGSM(models, ospgsm_int_config, random_state=1010)
    # inverted_preds = new_comp.run(X_val, X_test)

    #print("MSE inverted", mean_squared_error(X_test, inverted_preds))

collector = np.array(collector)
plt.figure()
idx_zero = np.where(collector[:, 1] == 0)[0]
idx_nonzero = np.where(collector[:, 1] != 0)[0]
plt.plot(collector[:, 1][idx_zero], collector[:, 0][idx_zero], "g+")
plt.plot(collector[:, 1][idx_nonzero], collector[:, 0][idx_nonzero], "r+")
plt.xlabel("Length of RoCs")
plt.ylabel("Difference to ensemble prediction")
plt.hlines(0, 0, np.max(collector[:, 1]))
plt.savefig("test.png")