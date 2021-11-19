import skorch
from datasets.monash_forecasting import _get_ds_names
from single_models import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02, AS_LSTM_03, Simple_LSTM, OneResidualFCN, TwoResidualFCN
from compositors import OS_PGSM
from itertools import product

ds_names =  _get_ds_names()
ds_indices = list(range(5))

implemented_datasets = product(ds_names, ds_indices)

def load_models(ds_name, ds_index, return_names=False):
    all_models = []
    all_model_names = []
    model_names = [(name, m_obj) for (name, m_obj) in single_models.items() if not "lstm" in name and not "adaptive" in name]
    for model_name, model_obj in model_names:
        save_path = f"models/{ds_name}/{ds_index}_{model_name}.pth"
        nr_filters = model_obj["nr_filters"]
        hidden_states = model_obj["hidden_states"]
        model = skorch.NeuralNetRegressor(
                model_obj["obj"], 
                module__nr_filters=nr_filters, 
                module__hidden_states=hidden_states, 
                module__ts_length=5) # ?

        model.initialize()
        model.load_params(f_params=save_path)
        all_models.append(model.module_)
        all_model_names.append(model_name)

    if return_names:
        return (all_models, all_model_names)
    return all_models
        
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
        "hidden_states": 30
    },
    "rnn_e" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 64,
        "hidden_states": 30
    },
    "rnn_f" : {
        "obj": Shallow_CNN_RNN,
        "nr_filters": 128,
        "hidden_states": 30
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
        "hidden_states": 30
    },
    "as01_e" : {
        "obj": AS_LSTM_01,
        "nr_filters": 64,
        "hidden_states": 30
    },
    "as01_f" : {
        "obj": AS_LSTM_01,
        "nr_filters": 128,
        "hidden_states": 30
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
        "hidden_states": 30
    },
    "as02_e" : {
        "obj": AS_LSTM_02,
        "nr_filters": 64,
        "hidden_states": 30
    },
    "as02_f" : {
        "obj": AS_LSTM_02,
        "nr_filters": 128,
        "hidden_states": 30
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
        "hidden_states": 30
    },
    "as03_e" : {
        "obj": AS_LSTM_03,
        "nr_filters": 64,
        "hidden_states": 30
    },
    "as03_f" : {
        "obj": AS_LSTM_03,
        "nr_filters": 128,
        "hidden_states": 30
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
def min_distance_drifts():
    return dict(
            k=5, 
            n_omega=50, 
            z=25,
            small_z=1,
            roc_mean = True,
            delta=0.95,
            topm=10,
            invert_relu=False,
            roc_take_only_best=False,
            smoothing_threshold=0.1,
            nr_clusters_single=1,
            nr_clusters_ensemble=5,
            drift_type="min_distance_change",
            concept_drift_detection="hoeffding",
    )

def ospgsm_original():
    return dict(
            k=5, 
            n_omega=25,
            z=25,
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            invert_relu=False,
            roc_take_only_best=False,
            smoothing_threshold=0.1,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection="hoeffding",
            drift_type="ospgsm"
    )

def ospgsm_per_original():
    return dict(
            k=5, 
            n_omega=25,
            z=25,
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            invert_relu=False,
            roc_take_only_best=False,
            smoothing_threshold=0.1,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection="periodic",
            drift_type="ospgsm",
    )

def ospgsm_st_original():
    return dict(
            k=5, 
            n_omega=25,
            z=25,
            small_z=1,
            roc_mean = False,
            delta=0.95,
            topm=1,
            invert_relu=False,
            roc_take_only_best=False,
            smoothing_threshold=0.1,
            nr_clusters_single=1,
            nr_clusters_ensemble=1,
            concept_drift_detection=None,
            drift_type="ospgsm",
    )

def ospgsm_int_original():
    return dict(
            k=5, 
            omega=0.25, 
            z=25,
            small_z=5,
            roc_mean=False,
            delta=0.95,
            topm=1,
            invert_relu=False,
            roc_take_only_best=False,
            smoothing_threshold=0.1,
            nr_clusters_single=1,
            nr_clusters_ensemble=1, 
            concept_drift_detection=None,
            drift_type="ospgsm"
    )

# All configurations used for experiments
ospgsm_experiment_configurations = {
    "min_distance": min_distance_drifts,
    "ospgsm": ospgsm_original,
    "ospgsm_st": ospgsm_st_original,
    "ospgsm_int": ospgsm_int_original,
}

val_keys = ['y'] + ['pred_' + w for w in single_models.keys()]
test_keys = val_keys + ['pred_' + w for w in ospgsm_experiment_configurations.keys()]
