# code from https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py

from datetime import datetime
from numpy import distutils
from os.path import exists
import matplotlib.pyplot as plt
import hashlib
import distutils
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ts_path = "code/datasets/monash_complete.npy"

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with = 'NaN', value_column_name = "series_value"):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"): # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (len(line_content) != 3):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(distutils.util.strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(distutils.util.strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if(len(series) == 0):
                            raise Exception("A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if (numeric_series.count(replace_missing_vals_with) == len(numeric_series)):
                            raise Exception("All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                            else:
                                raise Exception("Invalid attribute type.") # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if(att_val == None):
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length

def extract_subseries(df, amount=15, length=500, random_state=0):
    extracted_timeseries = np.zeros((amount, length))
    n_rows = len(df)

    rng = np.random.RandomState(random_state)
    row_subset = rng.choice(np.arange(n_rows), size=n_rows, replace=False)
    used_series = 0
    for idx, row in enumerate(row_subset):
        subseries = df.iloc[row]['series_value'].to_numpy().squeeze()
        if len(subseries) < length:
            continue
        i = 1
        while np.mean(subseries) == subseries[0]:
            subseries = df.iloc[row+i]['series_value'].to_numpy().squeeze()
            i+=1
        assert len(subseries) >= length
        padding = len(subseries) - length
        start_idx = rng.randint(0, high=padding-1)

        series = subseries[start_idx:(start_idx+length)]
        scaler = StandardScaler()
        series = scaler.fit_transform(series.reshape(-1, 1))
        extracted_timeseries[used_series] = series.squeeze()

        used_series += 1

        if used_series >= amount:
            break

    print(extracted_timeseries[:, :5])
    return extracted_timeseries

def calculate_dataset_seed(ds_name):
    return int(hashlib.md5((ds_name).encode("utf-8")).hexdigest(), 16) & 0xffffffff

def describe_dataset(used_dataset_names, dataset):
    print("Used datasets")
    for name in used_dataset_names:
        print(name)
    print(f"Dataset size {dataset.shape}")

def _get_ds_names():
    used_datasets = sorted(glob.glob("code/datasets/monash_ts/*.tsf"))
    ds = []
    for path in used_datasets:
        dataset_name = path.split("/")[-1].split(".")[0]
        ds.append(dataset_name)
    return ds
    
def load_dataset(name, idx):
    ds_names = _get_ds_names()
    ds_idx = None
    for i, dn in enumerate(ds_names):
        if dn == name:
            ds_idx = i
            break
    if ds_idx is None:
        raise Exception(f"Unknown dataset {name}, choose one from {ds_names}")

    ds = np.load(ts_path)
    return ds[ds_idx + idx].squeeze()

def plot_datasets():
    full_ds = np.load(ts_path) 
    for idx in range(25):
        ds = full_ds[idx].squeeze()

        plt.figure(figsize=(10, 4))
        plt.plot(ds)
        plt.savefig(f"plots/ds_{idx}.png")
        plt.close()

if __name__ == "__main__":
    used_datasets = sorted(glob.glob("code/datasets/monash_ts/*.tsf"))

    if exists(ts_path):
        dataset = np.load(ts_path)
        describe_dataset(used_datasets, dataset)
        exit()

    ts_agg = []
    for path in used_datasets:
        dataset_name = path.split("/")[-1].split(".")[0]
        dataset_seed = calculate_dataset_seed(dataset_name)
        print(dataset_name, dataset_seed)
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(path)
        ts = extract_subseries(loaded_data, random_state=dataset_seed)
        ts_agg.append(ts)

    dataset = np.concatenate(ts_agg)
    np.save(ts_path, dataset)
    describe_dataset(used_datasets, dataset)

