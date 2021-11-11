import numpy as np
import matplotlib.pyplot as plt

def calculate_bar_width(nr_bars, desired_padding):
    space_after_padding = 1 - 2 * desired_padding
    width = space_after_padding / nr_bars

    return width

def calculate_x_positions(nr_bars, width):
    nr_one_side = nr_bars // 2
    odd_nr_bars = nr_bars % 2 == 1
    offsets = [i * width/2 for i in range(nr_one_side)]
    one_side = np.array([offsets[i-1] + i * width/2 + odd_nr_bars*width/2 for i in range(1, nr_one_side+1)])

    if odd_nr_bars:
        return np.concatenate([np.flip(-one_side), np.zeros((1)), one_side])
    else:
        return np.concatenate([np.flip(-one_side), one_side])

def grouped_barplot(ax, X, ys, padding, colors, labels=None):
    used_labels = set()
    for x, y in zip(X, ys):
        nr_bars = len(y)
        width = calculate_bar_width(nr_bars, padding)
        xs = calculate_x_positions(nr_bars, width) + x
        for i, (_x, _y) in enumerate(zip(xs,y)):
            if labels is not None and labels[i] not in used_labels:
                ax.bar(x=_x, height=_y, width=width, color=colors[i], label=labels[i])
            else:
                ax.bar(x=_x, height=_y, width=width, color=colors[i])

            used_labels.add(labels[i])


if __name__ == "__main__":
    x = np.arange(3)
    desired_padding = 0
    ys = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.4], [0.1, 0.2, 0.3]])
    colors = ["red", "blue", "coral"]

    fig, ax = plt.subplots(1,1)
    for i in x:
        grouped_barplot(ax, i, ys[i], desired_padding, colors)

    plt.savefig("plots/test.png")