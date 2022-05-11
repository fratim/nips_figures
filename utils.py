import os
import pandas
import numpy as np

def get_data_dir(DATA_PATH, PROJECT, LEARNER):
    return os.path.join(DATA_PATH, PROJECT, LEARNER)

def get_fname(LEARNER, DEM):
    fname = f"{LEARNER}_from_{DEM}"
    return fname

def get_figure_fpath(data_to_plot, DATA_PATH, PROJECT, LEARNER, DEM):
    data_dir = get_data_dir(DATA_PATH, PROJECT, LEARNER)
    fname_base = get_fname(LEARNER, DEM)
    fname = f"{fname_base}_nseeds{data_to_plot['n_seeds']}"

    fpath = os.path.join(data_dir, fname)

    return fpath

def load_data(DATA_PATH, PROJECT, LEARNER, DEM):

    data_dir = get_data_dir(DATA_PATH, PROJECT, LEARNER)
    fname = get_fname(LEARNER, DEM) + ".csv"

    fpath = os.path.join(data_dir, fname)

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

def save_fig(plt, data_to_plot, DATA_PATH, PROJECT, LEARNER, DEM):
    fpath_base = get_figure_fpath(data_to_plot, DATA_PATH, PROJECT, LEARNER, DEM)
    plt.savefig(f"{fpath_base}.png")
    plt.savefig(f"{fpath_base}.pgf")