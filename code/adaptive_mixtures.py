import torch
import torch.nn as nn
import numpy as np

from datasets import Bike_Total_Rents
from datasets.utils import train_test_split, sliding_split

class AdaptiveMixForecaster(nn.Module):

    def __init__(self, lag=5, nr_filters=None, nr_experts=8, ts_length=None, hidden_states=None, learning_rate=1e-3, batch_size=50, epochs=300):
        super(AdaptiveMixForecaster, self).__init__()

        self.lag = lag
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.experts = []
        self.nr_experts = nr_experts
        for i in range(nr_experts):
            self.experts.append(self.generate_expert())

        L = [
            nn.Conv1d(1, len(self.experts), 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.lag * len(self.experts), len(self.experts)),
            nn.Softmax(dim=-1)
        ]
        self.gating_network = nn.Sequential(*L)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x):
        if x.shape[0] == 1:
            expert_preds = [expert.forward(x) for expert in self.experts]
        else:
            expert_preds = [expert.forward(x).squeeze().unsqueeze(0) for expert in self.experts]

        p_i = self.gating_network.forward(x)

        return torch.cat(expert_preds, dim=0).permute(1,0), p_i

    def predict(self, x):
        with torch.no_grad():
            predictions = []
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            expert_preds, p_i = self.forward(x)
            predicted_experts = torch.argmax(p_i, axis=-1)
            for i, expert_index in enumerate(predicted_experts):
                predictions.append(expert_preds[i][expert_index])

            return np.array(predictions)


    def generate_expert(self):
        #return nn.Linear(self.lag, 1)
        return nn.Sequential(nn.Conv1d(1, 3, 1, padding=0), nn.ReLU(), nn.Flatten(), nn.Linear(3*self.lag, 1))

    def adaptive_loss(self, y, expert_preds, p_i):
        y = y.reshape(-1, 1).repeat(1, expert_preds.shape[-1])
        #t = torch.exp(-1/2 * torch.linalg.norm(y - expert_preds, 1, dim=0) ** 2)
        t = -torch.log(torch.sum(p_i * torch.exp(-1/2*torch.abs(y-expert_preds)**2), axis=1))
        return t

    def fit(self, x_train, y_train, X_val=None, y_val=None, model_save_path=None, verbose=True):
        ds = torch.utils.data.TensorDataset(x_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.adaptive_loss
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-1)

        if model_save_path is not None:
            best_validation = 1e12
            best_epoch = 0

        logs = {
            "val_mse": []
        }

        for epoch in range(self.epochs):
            train_predictions = []
            train_labels = []
            print_epoch = epoch + 1
            epoch_loss = 0.0
            epoch_correct = 0
            for i, (X, y) in enumerate(dl):
                self.train()
                optim.zero_grad()
                expert_preds, p_i = self.forward(X)
                loss = loss_fn(y, expert_preds, p_i)

                summed_loss = loss.sum()
                summed_loss.backward()
                epoch_loss += summed_loss.item()
                optim.step()

                # see how accurate p_i is
                gaiting_decision = torch.argmax(p_i, axis=1).detach()
                epoch_correct += torch.sum(torch.argmin(torch.abs(y.reshape(-1, 1).detach().repeat(1, len(self.experts)) - expert_preds.detach()), axis=1) == gaiting_decision)

            if X_val is not None:
                with torch.no_grad():
                    self.eval()
                    expert_preds, p_i = self.forward(X_val)
                    val_preds = []
                    for i, best_predictor in enumerate(torch.argmax(p_i, axis=-1)):
                        val_preds.append(expert_preds[i][best_predictor])

                    val_preds = torch.tensor(val_preds)
                    mse_score = nn.MSELoss()(val_preds, y_val.squeeze())
                    logs["val_mse"].append(mse_score)
                    if model_save_path is not None:
                        if mse_score < best_validation:
                            best_validation = mse_score
                            best_epoch = epoch
                    if verbose:
                        print(epoch_loss, epoch_correct / len(ds), mse_score)
            else:
                if verbose:
                    print(epoch_loss, epoch_correct / len(ds))

        return logs, best_epoch
                

if __name__ == "__main__":
    X = Bike_Total_Rents().torch()
    X_train, X_test = train_test_split(X, split_percentages=(0.75, 0.25))
    X_train, X_val = train_test_split(X_train, split_percentages=(0.66, 0.33))
    x_train, y_train = sliding_split(X_train, 5, use_torch=True)
    x_val, y_val = sliding_split(X_val, 5, use_torch=True)
    net = AdaptiveMixForecaster(5)
    net.fit(x_train, y_train, X_val=x_val, y_val=y_val)
