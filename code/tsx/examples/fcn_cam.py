import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset

from tsx.attribution import ClassActivationMaps
from tsx.models.classifier import TimeSeries1DNet
from tsx.datasets import load_ecg200
from tsx.visualizations import plot_cam
from tsx.utils import prepare_for_pytorch

ds = load_ecg200()
x_train, y_train = ds.torch(train=True)
x_test, y_test = ds.torch(train=False)

model = TimeSeries1DNet(n_classes=2, epochs=10)
model.fit(x_train, y_train, X_test=x_test, y_test=y_test)

example_x = x_test[10:13]
example_prediction = model.predict(example_x)

attr = ClassActivationMaps(example_x, example_prediction, model)

plot_cam(example_x, attr)