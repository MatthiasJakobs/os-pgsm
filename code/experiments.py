import torch
from datasets import *
from single_models import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02, AS_LSTM_03, Simple_LSTM, OneResidualFCN, TwoResidualFCN
from compositors import OS_PGSM
from adaptive_mixtures import AdaptiveMixForecaster

skip_models_composit = [Simple_LSTM, AdaptiveMixForecaster]

# comps =                [KNN_ROC,    OS_PGSM_St,      OS_PGSM_Int,     OS_PGSM_Euc,               OS_PGSM_Int_Euc,           OS_PGSM,                    OS_PGSM_Per]
# comp_names =           ['baseline', 'gradcam_large', 'gradcam_small', 'gradcam_large_euclidian', 'gradcam_small_euclidian', 'large_adaptive_hoeffding', 'large_adaptive_periodic']

m4_data_path = "/data/M4"

def load_model(m_name, d_name, lag, ts_length):
    m_obj = single_models[m_name]

    if d_name.startswith("m4"):
        batch_size = 500
        epochs = 3000
        lr = 1e-3
    else:
        d_obj = implemented_datasets[d_name]

        try:
            batch_size = d_obj["batch_size"]
        except:
            batch_size = 500
        try:
            epochs = d_obj["epochs"]
        except:
            epochs = 3000
        try:
            lr = d_obj["lr"]
        except:
            lr = 1e-3

    nr_filters = m_obj["nr_filters"]
    hidden_states = m_obj["hidden_states"]

    if "lstm" in m_name or "adaptive_mixture" in m_name:
        m = m_obj['obj'](lag, batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=ts_length, hidden_states=hidden_states, learning_rate=lr)
    elif "cnn" in m_name or "residual" in m_name:
        m = m_obj['obj'](batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=ts_length, learning_rate=lr)
    else:
        m = m_obj['obj'](batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=ts_length, hidden_states=hidden_states, learning_rate=lr)

    m.load_state_dict(torch.load("models/{}/{}_lag{}.pth".format(m_name, d_name, lag)))

    return m

single_models = {
    "rnn_a" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "rnn_b" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "rnn_c" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "rnn_d" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 32,
        "hidden_states": 100
    },
    "rnn_e" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 64,
        "hidden_states": 100
    },
    "rnn_f" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 128,
        "hidden_states": 100
    },
    "cnn_a" : {
        "obj": Shallow_FCN,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "cnn_b" : {
        "obj": Shallow_FCN,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "cnn_c" : {
        "obj": Shallow_FCN,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "as01_a" : {
        "obj": AS_LSTM_01,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "as01_b" : {
        "obj": AS_LSTM_01,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "as01_c" : {
        "obj": AS_LSTM_01,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "as01_d" : {
        "obj": AS_LSTM_01,
        "nr_filters": 32,
        "hidden_states": 100
    },
    "as01_e" : {
        "obj": AS_LSTM_01,
        "nr_filters": 64,
        "hidden_states": 100
    },
    "as01_f" : {
        "obj": AS_LSTM_01,
        "nr_filters": 128,
        "hidden_states": 100
    },
    "as02_a" : {
        "obj": AS_LSTM_02,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "as02_b" : {
        "obj": AS_LSTM_02,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "as02_c" : {
        "obj": AS_LSTM_02,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "as02_d" : {
        "obj": AS_LSTM_02,
        "nr_filters": 32,
        "hidden_states": 100
    },
    "as02_e" : {
        "obj": AS_LSTM_02,
        "nr_filters": 64,
        "hidden_states": 100
    },
    "as02_f" : {
        "obj": AS_LSTM_02,
        "nr_filters": 128,
        "hidden_states": 100
    },
    "as03_a" : {
        "obj": AS_LSTM_03,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "as03_b" : {
        "obj": AS_LSTM_03,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "as03_c" : {
        "obj": AS_LSTM_03,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "as03_d" : {
        "obj": AS_LSTM_03,
        "nr_filters": 32,
        "hidden_states": 100
    },
    "as03_e" : {
        "obj": AS_LSTM_03,
        "nr_filters": 64,
        "hidden_states": 100
    },
    "as03_f" : {
        "obj": AS_LSTM_03,
        "nr_filters": 128,
        "hidden_states": 100
    },
    "one_residual_a" : {
        "obj": OneResidualFCN,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "one_residual_b" : {
        "obj": OneResidualFCN,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "one_residual_c" : {
        "obj": OneResidualFCN,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "two_residual_a" : {
        "obj": TwoResidualFCN,
        "nr_filters": 32,
        "hidden_states": 10
    },
    "two_residual_b" : {
        "obj": TwoResidualFCN,
        "nr_filters": 64,
        "hidden_states": 10
    },
    "two_residual_c" : {
        "obj": TwoResidualFCN,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "lstm_a" : {
        "obj": Simple_LSTM,
        "nr_filters": 128,
        "hidden_states": 10
    },
    "adaptive_mixture": {
        "obj": AdaptiveMixForecaster,
        "nr_filters": None,
        "hidden_states": None
    }
}

implemented_datasets = {
    "bike_total_rents": {
        "ds": Bike_Total_Rents,
        "epochs": 1500,
        "batch_size": 50,
        "lr": 1e-4,
    },
    "bike_registered": {
        "ds": Bike_Registered,
        "epochs": 1500,
        "batch_size": 50,
        "lr": 1e-4,
    },
    "bike_temperature": {
        "ds": Bike_Temperature,
        "epochs": 2500,
        "batch_size": 50,
        "lr": 1e-4,
    },
    "AbnormalHeartbeat": {
        "ds": AbnormalHeartbeat,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "CatsDogs": {
        "ds": CatsDogs,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Cricket": {
        "ds": Cricket,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "EOGHorizontalSignal": {
        "ds": EOGHorizontalSignal,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "EthanolConcentration": {
        "ds": EthanolConcentration,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Mallat": {
        "ds": Mallat,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Phoneme": {
        "ds": Phoneme,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "PigAirwayPressure": {
        "ds": PigAirwayPressure,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Rock": {
        "ds": Rock,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "SNP500": {
        "ds": SNP500,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "NASDAQ": {
        "ds": NASDAQ,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "DJI": {
        "ds": DJI,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "NYSE": {
        "ds": NYSE,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "RUSSELL": {
        "ds": RUSSELL,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Energy_RH1": {
        "ds": Energy_RH1,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Energy_RH2": {
        "ds": Energy_RH2,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Energy_T4": {
        "ds": Energy_T4,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "Energy_T5": {
        "ds": Energy_T5,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
    "CloudCoverage": {
        "ds": CloudCoverage,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
}

lag_mapping = {
     "5": 25,
    "10": 40,
    "15": 60,
}

###
# k: Number of lagged values
# omega: Size of validation set (in percent)
# n_omega: Length of RoC
# z: Number of steps
# delta: Hoeffding bound threshold
# topm: Choose the best m models for ensembling
# nr_clusters_single: Number of desired clusters inside each single model
# nr_clusters_ensemble: Number of desired clusters over all ensembles
# concept_drift_detection: ["periodic", "hoeffding", None]
###
def ospgsm_original(lag):
    return dict(
            k=lag, 
            omega=0.25, 
            n_omega=lag_mapping[str(lag)], 
            z=lag_mapping[str(lag)], 
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            smoothing_threshold=0.5,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection="hoeffding",
    )

def ospgsm_per_original(lag):
    return dict(
            k=lag, 
            omega=0.25, 
            n_omega=lag_mapping[str(lag)], 
            z=lag_mapping[str(lag)], 
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            smoothing_threshold=0.5,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection="periodic",
    )

def ospgsm_st_original(lag):
    return dict(
            k=lag, 
            omega=0.25, 
            n_omega=lag_mapping[str(lag)], 
            z=lag_mapping[str(lag)], 
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            smoothing_threshold=0.5,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection=None,
    )

def ospgsm_int_original(lag):
    return dict(
            k=lag, 
            omega=0.25, 
            n_omega=lag_mapping[str(lag)], 
            z=lag_mapping[str(lag)], 
            small_z=lag,
            roc_mean=False,
            delta=0.95,
            topm=1,
            smoothing_threshold=0.5,
            nr_clusters_single=1,
            nr_clusters_ensemble=1, 
            concept_drift_detection=None,
    )

# All configurations used for experiments
ospgsm_experiment_configurations = {
    "ospgsm": ospgsm_original,
    "ospgsm_st": ospgsm_st_original,
    "ospgsm_int": ospgsm_int_original,
}

val_keys = ['y'] + ['pred_' + w for w in single_models.keys()]
test_keys = val_keys + ['pred_' + w for w in ospgsm_experiment_configurations.keys()]