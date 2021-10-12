import torch
import numpy as np
from datasets.utils import windowing
from compositors import OS_PGSM as new_OSPGSM
from legacy_composition import OS_PGSM, OS_PGSM_Int, OS_PGSM_Per, OS_PGSM_St
from experiments import ospgsm_original, ospgsm_per_original, ospgsm_st_original, implemented_datasets, load_model, single_models, skip_models_composit, ospgsm_int_original

ds_name = "AbnormalHeartbeat"
ospgsm_st_original_config = ospgsm_st_original(5)
lag = ospgsm_st_original_config["k"]
ts_length = lag 

big_lag_mapping = {
     "5": 25,
    "10": 40,
    "15": 60,
}   

big_lag = big_lag_mapping[str(lag)]

ds = implemented_datasets[ds_name]['ds']()
model_names = single_models.keys()

X = ds.torch()
[_, _], [_, _], _, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=big_lag, use_torch=True)

models = []
for m_name in model_names:
    m = load_model(m_name, ds_name, lag, ts_length)
    if type(m) not in skip_models_composit:
        models.append(m)

models = models[:11]

print("OSPGSM-Int Old")
torch.manual_seed(0)
old_comp = OS_PGSM_Int(models, lag,  big_lag)
old_preds = old_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("OSPGSM-Int New")
ospgsm_int_config = ospgsm_int_original(5)
torch.manual_seed(0)
new_comp = new_OSPGSM(models, ospgsm_int_config)
new_preds = new_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("Percent preds correct", np.sum(np.isclose(old_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(old_preds))
print("Nr Test forecasters different:", np.sum(new_comp.test_forecasters != old_comp.test_forecasters))

print("-"*50)

print("OSPGSM-ST Old")
torch.manual_seed(0)
old_comp = OS_PGSM_St(models, lag,  big_lag)
old_preds = old_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("OSPGSM-ST New")
torch.manual_seed(0)
new_comp = new_OSPGSM(models, ospgsm_st_original_config)
new_preds = new_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("Percent preds correct", np.sum(np.isclose(old_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(old_preds))
print("Nr Test forecasters different:", np.sum(new_comp.test_forecasters != old_comp.test_forecasters))

print("-"*50)

ospgsm_original_config = ospgsm_original(5)

print("OSPGSM Old")
torch.manual_seed(0)
old_comp = OS_PGSM(models, lag,  big_lag)
old_preds = old_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("OSPGSM New")
torch.manual_seed(0)
new_comp = new_OSPGSM(models, ospgsm_original_config)
new_preds = new_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("Percent preds correct", np.sum(np.isclose(old_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(old_preds))
print("Nr Test forecasters different:", np.sum(new_comp.test_forecasters != old_comp.test_forecasters))

print("-"*50)

ospgsm_per_original_config = ospgsm_per_original(5)

print("OSPGSM-Per Old")
torch.manual_seed(0)
old_comp = OS_PGSM_Per(models, lag,  big_lag)
old_preds = old_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("OSPGSM-Per New")
torch.manual_seed(0)
new_comp = new_OSPGSM(models, ospgsm_per_original_config)
new_preds = new_comp.run(X_val, X_test, verbose=False, random_state=1010)

print("Percent preds correct", np.sum(np.isclose(old_preds - new_preds, np.zeros_like(new_preds), atol=1e-4)) / len(old_preds))
print("Nr Test forecasters different:", np.sum(new_comp.test_forecasters != old_comp.test_forecasters))