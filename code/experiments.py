from datasets import *
from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02

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
    # "CinCECGTorso": {
    #     "ds": CinCECGTorso(),
    #     "epochs": 2500,
    #     "batch_size": 100,
    #     "lr": 1e-4,
    # },
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
    # "MixedShapes": {
    #     "ds": MixedShapes(),
    #     "epochs": 2500,
    #     "batch_size": 100,
    #     "lr": 1e-4,
    # },
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
    # "SharePriceIncrease": {
    #     "ds": SharePriceIncrease(),
    #     "epochs": 2500,
    #     "batch_size": 100,
    #     "lr": 1e-4,
    # },
    "SNP500": {
        "ds": SNP500,
        "epochs": 2500,
        "batch_size": 100,
        "lr": 1e-4,
    },
}
