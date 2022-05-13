import os
import pandas
import numpy as np

X_MAX = 2500000
SMOOTHING = 5

def find_cutoff(x):
    cutoff = -1
    for i in range(len(x)):
        if x[i+1] > X_MAX:
            cutoff = i
            break
    if cutoff == -1:
        raise ValueError

    return cutoff

def smooth_data(y):
    assert SMOOTHING%2 == 1
    offset = int((SMOOTHING-1)/2)

    for i in range(offset, len(y)-offset-1):
        summed = 0
        for j in np.arange(-1*offset, offset+1):
            summed += y[int(i+j)]
        average = summed/SMOOTHING
        y[i] = average

    return y


def get_fname_input(learner, expert):
    name = f"{learner}from{expert}"
    return name

def get_fname_output(learner, expert):
    fname = f"{learner}_from_{expert}"
    return fname

def get_figure_fpath(data_to_plot, fig_save_path, learner, expert):
    fname_base = get_fname_output(learner, expert)
    fname = f"{fname_base}"

    fpath = os.path.join(fig_save_path, fname)

    return fpath

def load_data(data_path, learner, expert):
    fname = get_fname_input(learner, expert) + ".csv"
    fpath = os.path.join(data_path, fname)
    data_in = pandas.read_csv(fpath)

    assert (data_in.shape[1] - 1) % 3 == 0

    return data_in


def compute_mean_std_from_xy_pairs(xy_pairs):
    ys = []
    for item in xy_pairs:
        ys.append(item[1])

    ys = np.stack(ys, axis=1)
    mean = np.mean(ys, axis=1)
    std = np.std(ys, axis=1)

    return mean, std


def get_xy_pairs(data):
    n_seeds = int((data.shape[1] - 1) / 6)

    x_y_pairs = []

    for seed in range(n_seeds):

        x = data.iloc[:, 0].values
        y = data.iloc[:, 4 + 6 * seed].values

        y = smooth_data(y)

        cutoff = find_cutoff(x)

        x = x[:cutoff]
        y = y[:cutoff]

        if len(x_y_pairs) > 0:
            assert x.shape == x_y_pairs[-1][0].shape
            assert y.shape == x_y_pairs[-1][1].shape

        x_y_pairs.append((x, y))

    return x_y_pairs, n_seeds


def process_data(data):

    xy_pairs, n_seeds = get_xy_pairs(data)

    mean, std = compute_mean_std_from_xy_pairs(xy_pairs)

    data_to_plot = dict(
        xy_pairs=xy_pairs,
        mean=mean,
        std=std,
        n_seeds=n_seeds
    )

    return data_to_plot

def plot_data(data_to_plot, ax):
    # for pair in data_to_plot["xy_pairs"]:
    #     ax.plot(pair[0], pair[1])
    if "std" in data_to_plot.keys():
        mean = data_to_plot["mean"]
        std = data_to_plot["std"]
        y_lower = mean - std
        y_upper = mean + std
        x = data_to_plot["xy_pairs"][0][0]
        ax.fill_between(x, y_lower, y_upper, facecolor='blue', alpha=0.1)

    if "mean" in data_to_plot.keys():
        x = data_to_plot["xy_pairs"][0][0]
        mean = data_to_plot["mean"]
        ax.plot(x, mean, color="blue")

def save_fig(plt, data_to_plot, fig_save_path, learner, expert):
    fpath_base = get_figure_fpath(data_to_plot, fig_save_path, learner, expert)
    plt.tight_layout()
    plt.savefig(f"{fpath_base}.png")
    plt.savefig(f"{fpath_base}.pgf")