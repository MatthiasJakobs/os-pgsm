import torch
import numpy as np
from datasets.utils import windowing
from compositors import OS_PGSM as new_OSPGSM
from compositors import OS_PGSM_Faster as new_OSPGSM_fast
from legacy_composition import OS_PGSM
from experiments import ospgsm_original, single_models, load_models
from datasets.dataloading import load_dataset, implemented_datasets

from sklearn.metrics import mean_squared_error as mse

ds_name = "electricity_hourly"
lag = 5
ts_length = lag 

big_lag_mapping = {
     "5": 25,
    "10": 40,
    "15": 60,
}   

big_lag = big_lag_mapping[str(lag)]

names = ['new', 'new_fast', 'old']
best_implementation = []
ospgsm_original_config = ospgsm_original()

old_losses = []
new_losses = []
new_fast_losses = []

for ds_name, ds_index in implemented_datasets:
    try:
        X = load_dataset(ds_name, ds_index)
        model_names = single_models.keys()

        [_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=big_lag, use_torch=True)

        X_val = torch.from_numpy(X_val).float()
        X_test = X_test.squeeze().float()

        models = load_models(ds_name, ds_index)

        #print("OSPGSM New")
        torch.manual_seed(0)
        new_comp = new_OSPGSM(models, ospgsm_original_config, random_state=1010)
        new_preds = new_comp.run(X_val, X_test)

        #print("OSPGSM Old")
        torch.manual_seed(0)
        old_comp = OS_PGSM(models, lag,  big_lag)
        old_preds = old_comp.run(X_val, X_test, verbose=False, random_state=1010)

        #print("OSPGSM New (fast)")
        torch.manual_seed(0)
        new_fast_comp = new_OSPGSM_fast(models, ospgsm_original_config, random_state=1010)
        new_fast_preds = new_fast_comp.run(X_val, X_test)

        X_test = X_test.numpy()
        mse_old = mse(old_preds, X_test)
        mse_new_fast = mse(new_fast_preds, X_test)
        mse_new = mse(new_preds, X_test)

    except Exception:
        continue

    old_losses.append(mse_old)
    new_losses.append(mse_new)
    new_fast_losses.append(mse_new_fast)

    best_implementation.append(names[np.argmin([mse_new, mse_new_fast, mse_old])])

print(best_implementation)
print(np.unique(best_implementation, return_counts=True))

print('old')
print(old_losses)
print('---'*10)
print('new')
print(new_losses)
print('---'*10)
print('new_fast')
print(new_fast_losses)

# print("Percent preds correct (old vs new)", np.sum(np.isclose(old_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(old_preds))
# print("Percent preds correct (new vs new_fast)", np.sum(np.isclose(new_fast_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(new_preds))
# print("Nr Test forecasters different (old vs new):", np.sum(new_comp.test_forecasters != old_comp.test_forecasters))
# print("Nr Test forecasters different (new vs new_fast):", np.sum(new_comp.test_forecasters != new_fast_comp.test_forecasters))
