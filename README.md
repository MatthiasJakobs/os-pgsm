# Explainable Online Deep Neural Network Selection using Adaptive Saliency Maps for Time Series Forecasting

Code accompanying the ECML2021 research track submission 599 titled "Explainable Online Deep Neural Network Selection using Adaptive Saliency Maps for Time Series Forecasting"

## Install requirements

`pip install git+https://github.com/MatthiasJakobs/tsx.git@ecml2021`

`pip install torch==1.7 tqdm matplotlib sklearn fastdtw numpy pandas`

## How to use the code
- `code/experiments.py` contains the used single models and datasets, as well as global parameters.
- `code/train_models.py` trains the single models
    - The status of which models have already been trained can be monitored using `check_train_complete.py`
    - Example: `python code/train_models.py --dataset AbnormalHeartbeat --lag 10 --model rnn_a`
- `code/run_prediction.py` runs **PGSM** using the single models on all datasets
    - The results of `code/run_prediction.py` are stored in `results/`
    - The status of which results have already been created can be monitored using `check_pred_complete.py`
    - Example: `python code/run_prediction.py --lag 10`
- `code/measure_runtime.py` creates the runtime experiment and prints the results to the terminal.
- To generate all plots used in the paper, run `code/explain.py` after all experiments are finished.
- Using the R files in `EA-DRL-rep.zip`, the predictions can be compared to state-of-the-art methods as presented in the experiment section of our paper
