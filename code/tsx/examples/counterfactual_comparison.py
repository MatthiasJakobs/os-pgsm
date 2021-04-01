import numpy as np
import matplotlib.pyplot as plt
from tsx.datasets import load_itapowdem
from tsx.datasets.utils import normalize
from tsx.models.classifier import ROCKET
from tsx.counterfactuals import MOC, NativeGuide
from tsx.distances import dtw

ds = load_itapowdem(transforms=[normalize])
x_train, y_train = ds.torch(train=True)
x_test, y_test = ds.torch(train=False)

model = ROCKET(input_length=len(x_train[0]), batch_size=100, n_classes=len(np.unique(y_train)))
model.fit(x_train, y_train, x_test, y_test)

moc_cf = MOC(model, x_train, y_train, generations=150, log_generations=False)

x_star = x_test[10].unsqueeze(0)
y_star = y_test[10]
target = np.array([ 0 if y_star.item() == 1 else 1])

fitness, moc_xs = moc_cf.generate(x_star, target=target)
if len(fitness) != 0:
    print("MOC Evaluation")
    print("-"*10)
    print(moc_cf._apply_functions(moc_xs))

    nativeguide_cf = NativeGuide(model, x_train, y_train)
    ng_xs = nativeguide_cf.generate(x_star, y_star.unsqueeze(0))
    ng_xs = np.expand_dims(ng_xs[0][0][1], 0)

    print("NativeGuide Evaluation")
    print("-"*10)
    print(moc_cf._apply_functions(ng_xs))
    
    max_cols = 2
    nr_plots = len(ng_xs) + len(moc_xs)
    rows = int(np.ceil(nr_plots / max_cols))
    plt.figure()

    i = 1
    while i <= len(moc_xs):
        plt.subplot(rows, max_cols, i)
        plt.vlines(np.arange(len(moc_xs[0])), -5, 5, colors="black", alpha=0.3)
        plt.plot(x_star.squeeze().numpy(), color="black")
        plt.plot(moc_xs[i-1], color="green")
        i += 1

    while i <= nr_plots:
        plt.subplot(rows, max_cols, i)
        plt.vlines(np.arange(len(moc_xs[0])), -5, 5, colors="black", alpha=0.3)
        plt.plot(x_star.squeeze().numpy(), color="black")
        plt.plot(ng_xs[i - 1 - len(moc_xs)], color="blue")
        i += 1

    plt.show()
else:
    print("No solution found in MOC")
