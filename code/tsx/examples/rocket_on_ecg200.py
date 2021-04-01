from tsx.models.classifier import ROCKET
from tsx.datasets import load_ecg200
from tsx.datasets.utils import normalize
import torch

ds = load_ecg200(transforms=[normalize])
X_train, y_train = ds.torch(train=True)
X_test, y_test = ds.torch(train=False)

# Train model with Ridge regression
config_ridge = {
    "n_classes": 2,
    "ridge": True,
    "input_length": X_train.shape[-1]
}

model = ROCKET(**config_ridge)
model.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# Train model with logistic regression
config_logistic = {
    "n_classes": 2,
    "ridge": False,
    "learning_rate": 1e-5,
    "epochs": 100,
    "input_length": X_train.shape[-1]
}

model = ROCKET(**config_logistic)
model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
