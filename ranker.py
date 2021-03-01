import torch
import torch.nn as nn

class Ranker(nn.Module):

    def __init__(self, models):
        super(Ranker, self).__init__()
        self.models = models

    def forward(self, x):
        feature_maps = []
        predictions = []

        for m in self.models:
            intermediates = m.forward(x, return_intermediate=True)
            predictions.append(intermediates['logits'])
            feature_maps.append(intermediates['feats'])

        return feature_maps, predictions


        

    