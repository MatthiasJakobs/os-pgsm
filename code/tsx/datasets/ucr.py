import pandas as pd
import numpy as np
import zipfile
import tempfile
import torch

from os.path import join, basename, dirname, exists
from urllib.request import urlretrieve
from shutil import rmtree as remove_dir
from tsx.utils import prepare_for_pytorch
from scipy.io import arff

class UCR_UEA_Dataset:

    def __init__(self, name, path=None, download=True, transforms=None, remap_labels=True):
        self.name = name
        self.transforms = transforms
        self.download = download
        self.path = path
        self.remap_labels = remap_labels

        if self.path is None and self.download == False:
            raise ValueError("If you do not want to download the dataset, you need to provide a path!")

        self.download_or_load()

        self.x_train, self.y_train = self.parseData(self.train_path)
        self.x_test, self.y_test = self.parseData(self.test_path)

        if transforms is not None:
            for t in transforms:
                self.x_train = t(self.x_train)
                self.x_test = t(self.x_test)

    def download_or_load(self):
        # based on code from https://github.com/alan-turing-institute/sktime/blob/master/sktime/datasets/base.py
        if self.path is None:
            self.path = join(dirname(__file__), "data", self.name)

        if self.download:
            if not exists(self.path):
                url = "http://timeseriesclassification.com/Downloads/{}.zip".format(self.name)
                dl_dir = tempfile.mkdtemp()
                zip_file_name = join(dl_dir, basename(url))
                urlretrieve(url, zip_file_name)

                print(self.name)
                zipfile.ZipFile(zip_file_name, "r").extractall(self.path)
                remove_dir(dl_dir)

        if exists(join(self.path, self.name + "_TRAIN.txt")):
            self.train_path = join(self.path, self.name + "_TRAIN.txt")
            self.test_path = join(self.path, self.name + "_TEST.txt")
        elif exists(join(self.path, self.name + "_TRAIN.arff")):
            self.train_path = join(self.path, self.name + "_TRAIN.arff")
            self.test_path = join(self.path, self.name + "_TEST.arff")
        elif exists(join(self.path, self.name, self.name + "_TRAIN.arff")):
            self.train_path = join(self.path, self.name, self.name + "_TRAIN.arff")
            self.test_path = join(self.path, self.name, self.name + "_TEST.arff")
        else:
            raise NotImplementedError("No .arff or .txt files found for dataset {}".format(self.name))

    def parseData(self, path):
        features = []
        labels = []
        with open(path, "r") as fp:
            for line in fp:
                tokens = [s for s in line.split(" ") if s != ""]
                label = tokens[0]
                feature = np.array(tokens[1:]).astype(np.float32)

                features.append(pd.Series(feature))
                labels.append(label)

        self.same_length = len(np.unique([len(x) for x in features])) == 1
        if self.same_length:
            features = np.array(features)
        else:
            features = np.array(features, dtype=object)

        labels = np.array(labels).astype(float).astype(int)

        # make sure all labels are enumerated, starting from 0
        if self.remap_labels:
            old_labels = np.unique(labels)
            new_labels = np.arange(len(old_labels))

            for i, old_label in enumerate(old_labels):
                labels[labels == old_label] = new_labels[i]

        return np.array(features), np.array(labels).astype(float).astype(int)

    def torch(self, train=True):
        if self.same_length:
            if train:
                #return prepare_for_pytorch(self.x_train).float(), torch.from_numpy(self.y_train).long()
                #return torch.tensor(self.x_train).float(), torch.tensor(self.y_train).long() 
                x_1, y_1 = prepare_for_pytorch(self.x_train).float(), torch.from_numpy(self.y_train).long()
                x_2, y_2 = torch.tensor(self.x_train).float(), torch.tensor(self.y_train).long() 
                return x_1, y_1
            else:
                #return prepare_for_pytorch(self.x_test).float(), torch.from_numpy(self.y_test).long()
                #return torch.tensor(self.x_test).float(), torch.tensor(self.y_test).long() 
                x_1, y_1 = prepare_for_pytorch(self.x_test).float(), torch.from_numpy(self.y_test).long()
                x_2, y_2 = torch.tensor(self.x_test).float(), torch.tensor(self.y_test).long() 
                return x_1, y_1
        else:
            raise ValueError("Dataset {} contains time-series data with different length. Conversion to pytorch failed".format(self.name))
            if train:
                return self.x_train, self.y_train
            else:
                return self.x_test, self.y_test 

def load_adiac(**kwargs):
    name = "Adiac"
    return UCR_UEA_Dataset(name, **kwargs)

def load_arrowhead(**kwargs):
    name = "ArrowHead"
    return UCR_UEA_Dataset(name, **kwargs)

def load_beef(**kwargs):
    name = "Beef"
    return UCR_UEA_Dataset(name, **kwargs)

def load_beetlefly(**kwargs):
    name = "BeetleFly"
    return UCR_UEA_Dataset(name, **kwargs)

def load_birdchicken(**kwargs):
    name = "BirdChicken"
    return UCR_UEA_Dataset(name, **kwargs)

def load_cbf(**kwargs):
    name = "CBF"
    return UCR_UEA_Dataset(name, **kwargs)

def load_car(**kwargs):
    name = "Car"
    return UCR_UEA_Dataset(name, **kwargs)

def load_chlcon(**kwargs):
    name = "ChlorineConcentration"
    return UCR_UEA_Dataset(name, **kwargs)

def load_cincecgtorso(**kwargs):
    name = "CinCECGTorso"
    return UCR_UEA_Dataset(name, **kwargs)

def load_coffee(**kwargs):
    name = "Coffee"
    return UCR_UEA_Dataset(name, **kwargs)

def load_computers(**kwargs):
    name = "Computers"
    return UCR_UEA_Dataset(name, **kwargs)

def load_cricketx(**kwargs):
    name = "CricketX"
    return UCR_UEA_Dataset(name, **kwargs)

def load_crickety(**kwargs):
    name = "CricketY"
    return UCR_UEA_Dataset(name, **kwargs)

def load_cricketz(**kwargs):
    name = "CricketZ"
    return UCR_UEA_Dataset(name, **kwargs)

def load_diasizred(**kwargs):
    name = "DiatomSizeReduction"
    return UCR_UEA_Dataset(name, **kwargs)

def load_disphaoutagegro(**kwargs):
    name = "DistalPhalanxOutlineAgeGroup"
    return UCR_UEA_Dataset(name, **kwargs)

def load_disphaoutcor(**kwargs):
    name = "DistalPhalanxOutlineCorrect"
    return UCR_UEA_Dataset(name, **kwargs)

def load_disphatw(**kwargs):
    name = "DistalPhalanxTW"
    return UCR_UEA_Dataset(name, **kwargs)

def load_ecg200(**kwargs):
    name = "ECG200"
    return UCR_UEA_Dataset(name, **kwargs)

def load_ecg5000(**kwargs):
    name = "ECG5000"
    return UCR_UEA_Dataset(name, **kwargs)

def load_ecgfivedays(**kwargs):
    name = "ECGFiveDays"
    return UCR_UEA_Dataset(name, **kwargs)

def load_earthquakes(**kwargs):
    name = "Earthquakes"
    return UCR_UEA_Dataset(name, **kwargs)

def load_electricdevices(**kwargs):
    name = "ElectricDevices"
    return UCR_UEA_Dataset(name, **kwargs)

def load_faceall(**kwargs):
    name = "FaceAll"
    return UCR_UEA_Dataset(name, **kwargs)

def load_facefour(**kwargs):
    name = "FaceFour"
    return UCR_UEA_Dataset(name, **kwargs)

def load_facesucr(**kwargs):
    name = "FacesUCR"
    return UCR_UEA_Dataset(name, **kwargs)

def load_fiftywords(**kwargs):
    name = "FiftyWords"
    return UCR_UEA_Dataset(name, **kwargs)

def load_fish(**kwargs):
    name = "Fish"
    return UCR_UEA_Dataset(name, **kwargs)

def load_forda(**kwargs):
    name = "FordA"
    return UCR_UEA_Dataset(name, **kwargs)

def load_fordb(**kwargs):
    name = "FordB"
    return UCR_UEA_Dataset(name, **kwargs)

def load_gunpoint(**kwargs):
    name = "GunPoint"
    return UCR_UEA_Dataset(name, **kwargs)

def load_ham(**kwargs):
    name = "Ham"
    return UCR_UEA_Dataset(name, **kwargs)

def load_handoutlines(**kwargs):
    name = "HandOutlines"
    return UCR_UEA_Dataset(name, **kwargs)

def load_haptics(**kwargs):
    name = "Haptics"
    return UCR_UEA_Dataset(name, **kwargs)

def load_herring(**kwargs):
    name = "Herring"
    return UCR_UEA_Dataset(name, **kwargs)

def load_inlineskate(**kwargs):
    name = "InlineSkate"
    return UCR_UEA_Dataset(name, **kwargs)

# def load_inswinsou(**kwargs): # TODO: Is this correct?
#     name = "InsWinSou"
#     return UCR_UEA_Dataset(name, **kwargs)

def load_itapowdem(**kwargs):
    name = "ItalyPowerDemand"
    return UCR_UEA_Dataset(name, **kwargs)

def load_larkitapp(**kwargs):
    name = "LargeKitchenAppliances"
    return UCR_UEA_Dataset(name, **kwargs)

def load_lightning2(**kwargs):
    name = "Lightning2"
    return UCR_UEA_Dataset(name, **kwargs)

def load_lightning7(**kwargs):
    name = "Lightning7"
    return UCR_UEA_Dataset(name, **kwargs)

def load_mallat(**kwargs):
    name = "Mallat"
    return UCR_UEA_Dataset(name, **kwargs)

def load_meat(**kwargs):
    name = "Meat"
    return UCR_UEA_Dataset(name, **kwargs)

def load_medicalimages(**kwargs):
    name = "MedicalImages"
    return UCR_UEA_Dataset(name, **kwargs)

def load_midphaoutagegro(**kwargs):
    name = "MiddlePhalanxOutlineAgeGroup"
    return UCR_UEA_Dataset(name, **kwargs)

def load_midphaoutcor(**kwargs):
    name = "MiddlePhalanxOutlineCorrect"
    return UCR_UEA_Dataset(name, **kwargs)

def load_middlephalanxtw(**kwargs):
    name = "MiddlePhalanxTW"
    return UCR_UEA_Dataset(name, **kwargs)

def load_motestrain(**kwargs):
    name = "MoteStrain"
    return UCR_UEA_Dataset(name, **kwargs)

def load_noninvfetecgtho1(**kwargs):
    name = "NonInvasiveFetalECGThorax1"
    return UCR_UEA_Dataset(name, **kwargs)

def load_noninvfetecgtho2(**kwargs):
    name = "NonInvasiveFetalECGThorax2"
    return UCR_UEA_Dataset(name, **kwargs)

def load_osuleaf(**kwargs):
    name = "OSULeaf"
    return UCR_UEA_Dataset(name, **kwargs)

def load_oliveoil(**kwargs):
    name = "OliveOil"
    return UCR_UEA_Dataset(name, **kwargs)

def load_phaoutcor(**kwargs):
    name = "PhalangesOutlinesCorrect"
    return UCR_UEA_Dataset(name, **kwargs)

def load_phoneme(**kwargs):
    name = "Phoneme"
    return UCR_UEA_Dataset(name, **kwargs)

def load_plane(**kwargs):
    name = "Plane"
    return UCR_UEA_Dataset(name, **kwargs)

def load_prophaoutagegro(**kwargs):
    name = "ProximalPhalanxOutlineAgeGroup"
    return UCR_UEA_Dataset(name, **kwargs)

def load_prophaoutcor(**kwargs):
    name = "ProximalPhalanxOutlineCorrect"
    return UCR_UEA_Dataset(name, **kwargs)

def load_prophatw(**kwargs):
    name = "ProximalPhalanxTW"
    return UCR_UEA_Dataset(name, **kwargs)

def load_refdev(**kwargs):
    name = "RefrigerationDevices"
    return UCR_UEA_Dataset(name, **kwargs)

def load_screentype(**kwargs):
    name = "ScreenType"
    return UCR_UEA_Dataset(name, **kwargs)

def load_shapeletsim(**kwargs):
    name = "ShapeletSim"
    return UCR_UEA_Dataset(name, **kwargs)

def load_shapesall(**kwargs):
    name = "ShapesAll"
    return UCR_UEA_Dataset(name, **kwargs)

def load_smakitapp(**kwargs):
    name = "SmallKitchenAppliances"
    return UCR_UEA_Dataset(name, **kwargs)

def load_sonaiborobsur1(**kwargs):
    name = "SonyAIBORobotSurface1"
    return UCR_UEA_Dataset(name, **kwargs)

def load_sonaiborobsur2(**kwargs):
    name = "SonyAIBORobotSurface2"
    return UCR_UEA_Dataset(name, **kwargs)

def load_starlightcurves(**kwargs):
    name = "StarLightCurves"
    return UCR_UEA_Dataset(name, **kwargs)

def load_strawberry(**kwargs):
    name = "Strawberry"
    return UCR_UEA_Dataset(name, **kwargs)

def load_swedishleaf(**kwargs):
    name = "SwedishLeaf"
    return UCR_UEA_Dataset(name, **kwargs)

def load_symbols(**kwargs):
    name = "Symbols"
    return UCR_UEA_Dataset(name, **kwargs)

def load_syncon(**kwargs):
    name = "SyntheticControl"
    return UCR_UEA_Dataset(name, **kwargs)

def load_toeseg1(**kwargs):
    name = "ToeSegmentation1"
    return UCR_UEA_Dataset(name, **kwargs)

def load_toeseg2(**kwargs):
    name = "ToeSegmentation2"
    return UCR_UEA_Dataset(name, **kwargs)

def load_trace(**kwargs):
    name = "Trace"
    return UCR_UEA_Dataset(name, **kwargs)

def load_twoleadecg(**kwargs):
    name = "TwoLeadECG"
    return UCR_UEA_Dataset(name, **kwargs)

def load_twopatterns(**kwargs):
    name = "TwoPatterns"
    return UCR_UEA_Dataset(name, **kwargs)

def load_uwavgesliball(**kwargs):
    name = "UWaveGestureLibraryAll"
    return UCR_UEA_Dataset(name, **kwargs)

def load_uwavgeslibx(**kwargs):
    name = "UWaveGestureLibraryX"
    return UCR_UEA_Dataset(name, **kwargs)

def load_uwavgesliby(**kwargs):
    name = "UWaveGestureLibraryY"
    return UCR_UEA_Dataset(name, **kwargs)

def load_uwavgeslibz(**kwargs):
    name = "UWaveGestureLibraryZ"
    return UCR_UEA_Dataset(name, **kwargs)

def load_wafer(**kwargs):
    name = "Wafer"
    return UCR_UEA_Dataset(name, **kwargs)

def load_wine(**kwargs):
    name = "Wine"
    return UCR_UEA_Dataset(name, **kwargs)

def load_wordsynonyms(**kwargs):
    name = "WordSynonyms"
    return UCR_UEA_Dataset(name, **kwargs)

def load_worms(**kwargs):
    name = "Worms"
    return UCR_UEA_Dataset(name, **kwargs)

def load_wormstwoclass(**kwargs):
    name = "WormsTwoClass"
    return UCR_UEA_Dataset(name, **kwargs)

def load_yoga(**kwargs):
    name = "Yoga"
    return UCR_UEA_Dataset(name, **kwargs)

bake_off = [
    load_adiac,
    load_arrowhead,
    load_beef,
    load_beetlefly,
    load_birdchicken,
    load_cbf,
    load_chlcon,
    load_cincecgtorso,
    load_coffee,
    load_computers,
    load_cricketx,
    load_crickety,
    load_cricketz,
    load_diasizred,
    load_disphaoutagegro,
    load_disphaoutcor,
    load_disphatw,
    load_ecg200,
    load_ecg5000,
    load_ecgfivedays,
    load_earthquakes,
    load_electricdevices,
    load_faceall,
    load_facefour,
    load_facesucr,
    load_fiftywords,
    load_fish,
    load_forda,
    load_fordb,
    load_gunpoint,
    load_ham,
    load_handoutlines,
    load_haptics,
    load_herring,
    load_inlineskate,
    #load_inswinsou,
    load_itapowdem,
    load_larkitapp,
    load_lightning2,
    load_lightning7,
    load_mallat,
    load_meat,
    load_medicalimages,
    load_midphaoutagegro,
    load_midphaoutcor,
    load_middlephalanxtw,
    load_motestrain,
    load_noninvfetecgtho1,
    load_noninvfetecgtho2,
    load_osuleaf,
    load_oliveoil,
    load_phaoutcor,
    load_phoneme,
    load_plane,
    load_prophaoutagegro,
    load_prophaoutcor,
    load_prophatw,
    load_refdev,
    load_screentype,
    load_shapeletsim,
    load_shapesall,
    load_smakitapp,
    load_sonaiborobsur1,
    load_sonaiborobsur2,
    load_starlightcurves,
    load_strawberry,
    load_swedishleaf,
    load_symbols,
    load_syncon,
    load_toeseg1,
    load_toeseg2,
    load_trace,
    load_twoleadecg,
    load_twopatterns,
    load_uwavgesliball,
    load_uwavgeslibx,
    load_uwavgesliby,
    load_uwavgeslibz,
    load_wafer,
    load_wine,
    load_wordsynonyms,
    load_worms,
    load_wormstwoclass,
    load_yoga
]