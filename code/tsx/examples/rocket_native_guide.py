import numpy as np
import torch
import matplotlib.pyplot as plt
from tsx.models.classifier import ROCKET
from tsx.datasets import load_itapowdem
from tsx.counterfactuals import NativeGuide

ds = load_itapowdem()
x_train, y_train = ds.torch(train=True)
x_test, y_test = ds.torch(train=False)

model = ROCKET(input_length=x_train.shape[-1], batch_size=100, n_classes=len(np.unique(y_train)))
model.fit(x_train, y_train, x_test, y_test)

cf = NativeGuide(model, x_train, y_train, distance='euclidian', batch_size=1000)

print("Original classes of input: {}".format(y_test[0:2]))

# Get two counterfactuals for each datapoint
generated_cfs = cf.generate(x_test[0:2], y_test[0:2], n=2)

plt.figure()
for i in range(len(generated_cfs)):
    plt.subplot(1,2,i+1)
    plt.plot(x_test[i].squeeze(), color='green')
    print(generated_cfs[i][0][1].shape)
    plt.plot(generated_cfs[i][0][1], color='red')

plt.show()
