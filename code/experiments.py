import torch
from datasets import *
from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02, Simple_LSTM
from compositors import *
from adaptive_mixtures import AdaptiveMixForecaster

skip_models_composit = [Simple_LSTM, AdaptiveMixForecaster]

comps =                [KNN_ROC,    OS_PGSM_St,      OS_PGSM_Int,     OS_PGSM_Euc,               OS_PGSM_Int_Euc,           OS_PGSM,                    OS_PGSM_Per]
comp_names =           ['baseline', 'gradcam_large', 'gradcam_small', 'gradcam_large_euclidian', 'gradcam_small_euclidian', 'large_adaptive_hoeffding', 'large_adaptive_periodic']

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
    elif "cnn" in m_name:
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

val_keys = ['y'] + ['pred_' + w for w in single_models.keys()]
test_keys = val_keys + ['pred_' + w for w in comp_names]