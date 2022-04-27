import torch
import pandas as pd
import numpy as np

from scipy.io import arff
from sktime.datasets import load_UCR_UEA_dataset
from tsx.datasets.ucr import UCR_UEA_Dataset

class BaseDataset:

    def __init__(self, normalize=True):
        self.X = self.load_ds()
        if normalize:
            self.normalize()

    def load_ds(self):
        raise NotImplementedError()

    def torch(self):
        return torch.from_numpy(self.X).float()

    def normalize(self):
        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X - mean) / std

class BaseUEA(BaseDataset):

    def load_ds(self, name):
        ds = UCR_UEA_Dataset(name)
        print(ds.x_train.shape)
        # path = f"code/datasets/legacy_ts/{name}.arff"
        # self.X = arff.loadarff(path)[0]
        # print(self.X)
        # print(self.X.shape)
        # print(type(self.X))

class BaseSimple(BaseDataset):

    def load_ds(self, path):
        return pd.read_csv(path, delimiter=",").to_numpy().squeeze()

class BaseStock(BaseDataset):

    def load_ds(self, path):
        df = pd.read_csv(path, delimiter=",")
        x = df['Close'].to_numpy()
        return x

class NASDAQ(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/NASDAQ.csv")

class DJI(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/DJI.csv")

class RUSSELL(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/RUSSELL.csv")

class NYSE(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/NYSE.csv")

class SNP500(BaseDataset):

    def load_ds(self):
        df = pd.read_csv("code/datasets/legacy_ts/SNP500.csv", delimiter=",")
        x = df['SP500'].to_numpy().squeeze()
        return x

class BikeRegistered(BaseSimple):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/BikeRegistered.csv")

class BikeRents(BaseSimple):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/BikeRents.csv")

class BikeTemperature(BaseSimple):

    def load_ds(self):
        return super().load_ds("code/datasets/legacy_ts/BikeTemperature.csv")

class CloudCoverage(BaseDataset):

    def load_ds(self):
        df = pd.read_csv("code/datasets/legacy_ts/CloudCoverage.csv")
        x = df["CLOUDCOVER"][5:]
        x[x == -1] = np.nan
        x = x.fillna(method="ffill")
        return x.to_numpy()

class UCR_To_Forecasting(UCR_UEA_Dataset):

    # Take first datapoint from supplied dataset and return that as time-series
    # We made sure that they are sufficiently long
    def torch(self, train=True):
        mean = np.mean(self.X)
        var = np.std(self.X)
        self.X = (self.X - mean) / var
        return torch.from_numpy(self.X).float()

class AbnormalHeartbeat(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("AbnormalHeartbeat")

    def parseData(self, path):
        features = []
        labels = []
        data = pd.DataFrame(arff.loadarff(path)[0])
        self.X = data.drop(columns=["target"]).to_numpy()[0]

        return None, None


class CatsDogs(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("CatsDogs")

    def parseData(self, path):
        data = pd.DataFrame(arff.loadarff(path)[0])
        self.X = data.iloc[0][:-1].to_numpy().astype(np.float32)
        return None, None

class Cricket(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("Cricket")

    def parseData(self, path):
        data = pd.DataFrame(arff.loadarff(path)[0])
        self.X = np.squeeze(np.array(data.iloc[0]["relationalAtt"][0].tolist()))
        return None, None

class EOGHorizontalSignal(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("EOGHorizontalSignal")

    def parseData(self, path):
        features = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                feature = np.array(tokens[1:]).astype(np.float32)
                features.append(pd.Series(feature))

        self.X = np.squeeze(features[0].to_numpy())
        return None, None

class EthanolConcentration(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("EthanolConcentration")

    def parseData(self, path):
        data = pd.DataFrame(arff.loadarff(path)[0])
        self.X = np.squeeze(np.array(data.iloc[0]["relationalAtt"][0].tolist()))
        return None, None

class Mallat(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("Mallat")

    def parseData(self, path):
        features = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                feature = np.array(tokens[1:]).astype(np.float32)
                features.append(pd.Series(feature))

        self.X = np.squeeze(features[0].to_numpy())
        return None, None

class Phoneme(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("Phoneme")

    def parseData(self, path):
        features = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                feature = np.array(tokens[1:]).astype(np.float32)
                features.append(pd.Series(feature))

        self.X = np.squeeze(features[0].to_numpy())
        return None, None

class PigAirwayPressure(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("PigAirwayPressure")

    def parseData(self, path):
        features = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                feature = np.array(tokens[1:]).astype(np.float32)
                features.append(pd.Series(feature))

        self.X = np.squeeze(features[0].to_numpy())
        return None, None

class Rock(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("Rock")

    def parseData(self, path):
        features = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                feature = np.array(tokens[1:]).astype(np.float32)
                features.append(pd.Series(feature))

        self.X = np.squeeze(features[0].to_numpy())
        return None, None

class SharePriceIncrease(UCR_To_Forecasting):

    def __init__(self):
        super().__init__("SharePriceIncrease")

    def parseData(self, path):
        data = pd.DataFrame(arff.loadarff(path)[0])
        self.X = np.squeeze(np.array(data.iloc[0]["relationalAtt"][0].tolist()))
        return None, None

all_legacy_datasets = [
    BikeRegistered, 
    SNP500, 
    NYSE, 
    DJI, 
    RUSSELL, 
    AbnormalHeartbeat, 
    CatsDogs, 
    Cricket, 
    EOGHorizontalSignal, 
    EthanolConcentration, 
    Phoneme, 
    Rock
]
all_legacy_names = [
    "BikeRegistered", 
    "SNP500", 
    "NYSE", 
    "DJI", 
    "RUSSELL", 
    "AbnormalHeartbeat", 
    "CatsDogs", 
    "Cricket", 
    "EOGHorizontalSignal", 
    "EthanolConcentration", 
    "Phoneme", 
    "Rock"
]

def load_dataset(name, idx):
    ds_idx = None
    for i, dn in enumerate(all_legacy_names):
        if dn == name:
            ds_idx = i
            break
    if ds_idx is None:
        raise Exception(f"Unknown dataset {name}, choose one from {all_legacy_names}")

    ds = np.load("code/datasets/legacy_complete.npy")
    return ds[ds_idx].squeeze()

if __name__ == "__main__":
    # Cut legacy datasets to 500
    rng = np.random.RandomState(0)
    acc = np.zeros((len(all_legacy_datasets), 500))
    for i, ds_class in enumerate(all_legacy_datasets):
        X = ds_class().X

        start = len(X) - 500 - 1
        subset = X[start:(start+500)]
        subset = (subset - np.mean(subset)) / np.std(subset)
        acc[i] = subset
    
    np.save("code/datasets/legacy_complete.npy", acc)