import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tsx.datasets import UCR_UEA_Dataset
from scipy.io import arff

ds_list = [
    "AbnormalHeartbeat",
    "CatsDogs",
    "CinCECGtorso",
    "Cricket",
    "EOGHorizontalSignal",
    "EthanolConcentration",
    "Mallat",
    "MixedShapes",
    "Phoneme",
    "PigAirwayPressure",
    "Rock",
    "SharePriceIncrease",
]

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
