# OS-PGSM and OEP-ROC
Code accompanying the papers *Explainable Online Ensemble of Deep Neural Networks Pruning for Time Series Forecasting* and *Explainable Online Deep Neural Network Selection Using Adaptive Saliency Maps for Time Series Forecasting*.

## Datasets
Some datasets need to be downloaded and placed in `code/datasets/monash_ts` for the experiments to work.
Specifically, you need to download the `Electricity (Hourly)`, `KDD Cup 2018`, `Pedestrian Counts`, `Solar (10 minutes)` and `Weather` datasets from https://forecastingdata.org/.
If available, choose the datasets without missing values.
Also, you need to rename them to `electricity_hourly.tsf`, `kdd_cup_2018.tsf`, `pedestrian_counts.tsf`, `solar_10_minutes.tsf` and `weather.tsf`.

## Installing dependencies
We use `python` version `3.7.10`. Install all dependencies via
```
pip install -r requirements.txt
```

## Overview of the repository

- `code/compositors.py` contains the code for **OEP-ROC** and for **OS-PGSM**.
- `code/experiments.py` contains all experiment parameters and configurations.
