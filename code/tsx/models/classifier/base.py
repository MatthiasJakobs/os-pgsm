import torch
import pickle
import torch.nn as nn
import numpy as np

from os.path import join

class BaseClassifier:

    def fit(self, X, y):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def inform(self, string):
        if self.verbose:
            print(string)

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        raise NotImplementedError()

    
class BasePyTorchClassifier(nn.Module, BaseClassifier):

    def __init__(self, n_classes=10, epochs=5, batch_size=10, verbose=False, optimizer=torch.optim.Adam, loss=nn.CrossEntropyLoss, learning_rate=1e-3):
        super(BasePyTorchClassifier, self).__init__()
        self.classifier = True
        self.forecaster = False
        self.loss = loss
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        self.fitted = False


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        # Expects X, y to be Pytorch tensors 
        X_train, y_train, X_test, y_test = self.preprocessing(X_train, y_train, X_test=X_test, y_test=y_test)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.loss()
        optim = self.optimizer(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print_epoch = epoch + 1
            epoch_loss = 0.0
            for i, (X, y) in enumerate(dl):
                optim.zero_grad()
                prediction = self.forward(X)
                loss = loss_fn(prediction, y)
                loss.backward()
                epoch_loss += loss.item()
                optim.step()

            train_accuracy = self.accuracy(X_train, y_train)
            if X_test is not None and y_test is not None:
                test_accuracy = self.accuracy(X_test, y_test)
                print("Epoch {} train_loss {} train_accuracy {} test_accuracy {}".format(print_epoch, epoch_loss, train_accuracy, test_accuracy))
            else:
                print("Epoch {} train_loss {} train_accuracy {}".format(print_epoch, epoch_loss, train_accuracy))

        self.fitted = True

    # def predict(self, X):
    #     # Expects X to be Pytorch tensors 
    #     return self.forward(self.transform(X))

    def accuracy(self, X, y, batch_size=None):
        # Expects X, y to be Pytorch tensors
        number_y = len(y)
        if batch_size is None:
            batch_size = self.batch_size

        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        running_correct = 0
        for i, (X, y) in enumerate(dl):
            prediction = self.forward(X)
            prediction = torch.argmax(prediction, dim=-1)
            running_correct += torch.sum((prediction == y).float())

        return running_correct / number_y
