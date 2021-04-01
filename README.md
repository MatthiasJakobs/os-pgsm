# Explainable Online Deep Neural Network Selection using Adaptive Saliency Maps for Time Series Forecasting

Code accompanying our ECML2021 research track submission.

## How to use the code
- `requirements.txt` contains all used Python packages, including their version number.
    - Sometimes PyTorch crashes while installing when using a `requirements.txt` so you might need to remove it from `requirements.txt` and install separately
- `code/experiments.py` specifies the used single models and datasets, as well as global parameters.
- `code/train_models.py` trains the single models
    - The status of which models have already been trained can be monitored using `check_train_complete.py`
- `code/run_prediction.py` runs **PGSM** using the single models on all datasets
    - The results of `code/run_prediction.py` are stored in `results/`
    - The status of which results have already been created can be monitored using `check_pred_complete.py`
- Using the R files in `EA-DRL-rep.zip`, the predictions can be compared to state-of-the-art methods as presented in the experiment section of our paper
