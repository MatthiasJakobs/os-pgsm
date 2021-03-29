import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import *
from compositors import *
from datasets.utils import windowing
from viz import plot_compositor_results

def viz_rocs(comp):
    fig = plt.figure(figsize=(10,6))

    for i in range(12):
        ax = fig.add_subplot(3, 4, i+1)
        rocs = comp1.rocs[i]
        if len(rocs) == 0:
            continue
        mu = torch.mean(torch.cat(rocs))
        s = torch.std(torch.cat(rocs))
        max_length = max([len(w) for w in rocs])
        for r in rocs:
            #ax.plot((r-mu)/s)
            ax.plot(r)

    fig.tight_layout()
    fig.savefig("plots/explainability/abnormal_viz_rocs.png")
    plt.close(fig)

def train_comp(models, comp, x_val, X_test, lag):
    if len(x_val.shape) == 1:
        preds = comp.run(x_val, X_test, big_lag=lag_mapping[str(lag)])
    else:
        preds, _ = comp.run(x_val, X_test)

    return comp, preds

models = []
lag = 10
d_name = "AbnormalHeartbeat"
ds = implemented_datasets[d_name]['ds']()
X = ds.torch()
[x_train, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

single_names = []
for m_name in single_models.keys():
    if m_name not in ['lstm_a', 'adaptive_mixture']:
        single_names.append(m_name)
        models.append(load_model(m_name, d_name, lag, lag))

comp1 = GC_Large(models, lag, lag_mapping[str(lag)])
comp1, preds = train_comp(models, comp1, x_val_big, X_test, lag)

viz_rocs(comp1)
#plot_compositor_results(comp1, [100, 150], preds, X_test, single_names, d_name, "GC_Large", lag)
