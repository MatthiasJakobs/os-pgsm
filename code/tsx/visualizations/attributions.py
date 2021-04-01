import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib
import numpy as np

from tsx.utils import to_numpy
from tsx.visualizations.utils import calc_optimal_grid

def plot_cam(data, attribution, title=None, save_to=None):
    data = np.squeeze(to_numpy(data))
    attribution = np.squeeze(to_numpy(attribution))
    if len(data.shape) == 1:
        nr_images = 1
    else:
        nr_images = data.shape[0]    

    xs = np.arange(data.shape[-1])
    colors = cm.get_cmap('viridis')
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=1) # TODO: Is this correct? 
    sm = plt.cm.ScalarMappable(cmap=colors)
    rows, cols = calc_optimal_grid(nr_images)

    fig, axes = plt.subplots(rows, cols)
    if title is not None:
        fig.suptitle(title)

    for i in range(nr_images):
        ax = axes.flat[i]

        if nr_images == 1:
            ittr = zip(data, attribution, xs)
            ax.plot(data, color="black", zorder=1)
        else:
            ittr = zip(data[i], attribution[i], xs)
            ax.plot(data[i], color="black", zorder=1)

        for y_i, c_i, x_i in ittr:
            ax.scatter(x_i, y_i, color=colors(c_i), zorder=2, picker=True)

        sm.set_array([])
        plt.colorbar(sm, ax=ax)

    if nr_images % 2 == 1:
        fig.delaxes(axes.flat[-1])

    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()




