import os
import pandas
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import sem
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

X_MAX = 2500000
SMOOTHING = 7

# EXPERT_DISTANCES = dict(
#     Hopper=19.31,
#     Walker=14.0,
#     HalfCheetah=158.67,
# )

def find_cutoff(x):

    # this means that data is from the XIRL experiments
    if x[-1] < 2000000:
        return len(x)

    cutoff = -1
    for i in range(len(x)):
        if x[i+1] > X_MAX:
            cutoff = i
            break
    if cutoff == -1:
        raise ValueError

    return cutoff

def smooth_data(y, smoothing_factor=None):


    if smoothing_factor is None:
        smoothing_factor = SMOOTHING

    assert smoothing_factor % 2 == 1
    offset = int((smoothing_factor-1)/2)

    y_padded = np.concatenate((y[0]*np.ones(offset), y, y[-1]*np.ones(offset)), axis=0)

    for i in range(offset, len(y_padded)-offset-1):
        accumulated = []
        for j in np.arange(-1*offset, offset+1):
            accumulated.append(y_padded[int(i+j)])
        average = np.nanmean(accumulated)
        y_padded[i] = average

    return y_padded[offset:-offset]


def get_fname_input(learner, expert):
    name = f"{learner}from{expert}"
    return name

def get_fname_output(learner, expert):
    fname = f"{learner}_from_{expert}"
    return fname

def get_figure_fpath(fig_save_path, learner, expert):
    fname_base = get_fname_output(learner, expert)
    fname = f"{fname_base}"

    fpath = os.path.join(fig_save_path, fname)

    return fpath

def load_data(data_path, learner, expert, algo="ours"):
    fname = get_fname_input(learner, expert)

    print(f"loading data {fname}")

    fname += ".csv"
    fpath = os.path.join(data_path, algo, fname)
    data_in = pandas.read_csv(fpath)
    data_in = data_in.fillna(method="backfill")

    assert (data_in.shape[1] - 1) % 3 == 0

    return data_in


def compute_mean_std_from_xy_pairs(xy_pairs):
    ys = []
    for item in xy_pairs:
        ys.append(item[1])

    ys = np.stack(ys, axis=1)
    mean = np.mean(ys, axis=1)
    max = np.max(ys, axis=1)
    std = sem(ys, axis=1, nan_policy="omit")
    # std = np.nanstd(ys, axis=1)

    return mean, std, max


def get_xy_pairs(data, agent_type):
    n_seeds = int((data.shape[1] - 1) / 6)
    col_size = 6
    # if data.iloc[:, 0].values[-1] < 500:
    #     n_seeds = int((data.shape[1] - 1) / 3)
    #     col_size = 3
    # else:
    #     n_seeds = int((data.shape[1] - 1) / 6)
    #     col_size = 6

    print(f"loading n seeds: {n_seeds}")

    x_y_pairs = []

    for seed in range(n_seeds):

        x = data.iloc[:, 0].values

        # if x[-1] < 500:
        #

        # y = data.iloc[:, 4 + col_size * seed].values

        if data.shape[1] >= 19:
            x = data.iloc[:, 0].values
            y = data.iloc[:, 4 + 6 * seed].values
        else:
            x = data.iloc[:, 0].values

            x = x*3000000/x[-1]
            y = data.iloc[:, 1 + 1 * seed].values

        y = smooth_data(y)

        cutoff = find_cutoff(x)

        x = x[:cutoff]
        y = y[:cutoff]

        if x[-1] > 1000000:
            for i in range(len(x)):
                x[i] = np.round(x[i]/50000)*50000

        # y = y/EXPERT_DISTANCES[agent_type]

        if len(x_y_pairs) > 0:
            assert x.shape == x_y_pairs[-1][0].shape
            assert y.shape == x_y_pairs[-1][1].shape

        x_y_pairs.append((x, y))

    return x_y_pairs, n_seeds


def process_data(data, agent_type):

    xy_pairs, n_seeds = get_xy_pairs(data, agent_type)

    y_margin_max = np.NINF
    for pair in xy_pairs:
        y_margin_max = (np.max(pair[1])-np.min(pair[1])) if (np.max(pair[1])-np.min(pair[1])) > y_margin_max else y_margin_max

    mean, std, max = compute_mean_std_from_xy_pairs(xy_pairs)

    data_to_plot = dict(
        xy_pairs=xy_pairs,
        mean=mean,
        std=std,
        n_seeds=n_seeds,
        y_margin_max=y_margin_max,
        max=max,
        original_data=data
    )

    return data_to_plot

def remove_double_entries(x, y):

    for i in range(len(x)-1):
        if x[i] + 20000 > x[i+1]:
            average = (y[i] + y[i+1])/2
            x[i+1] = x[i]
            y[i] = average
            y[i+1] = average

    return x, y


def plot_data(data_to_plot, ax, color, label, fig, white_line_dist=None, style="solid"):

    # for cur_run in data_to_plot["xy_pairs"]:
    #     ax.plot(cur_run[0], cur_run[1], color=color, linewidth=0.5, label=label, alpha=0.3)
    #     renderer = fig.canvas.renderer
    #     ax.draw(renderer)
    #     plt.draw()


    if "mean" in data_to_plot.keys():
        x = data_to_plot["xy_pairs"][0][0]
        mean = data_to_plot["mean"]

        # remove double entries only for Gym data
        # import pdb
        # pdb.set_trace()

        if x[-1] > 1000000:
            x, mean = remove_double_entries(x, mean)

        # mean = smooth_data(mean, 3)
        ax.plot(x, mean, color=color, linewidth=2, label=label, linestyle=style, markersize=10)


    if "std" in data_to_plot.keys():
        mean = data_to_plot["mean"]
        std = data_to_plot["std"]
        y_lower = mean - std
        y_upper = mean + std
        x = data_to_plot["xy_pairs"][0][0]
        ax.fill_between(x, y_lower, y_upper, facecolor=color, alpha=0.1)

        # white_line_points = get_white_line_points(x, y_lower, white_line_dist)

        # plot white lines
        # ax.plot(white_line_points[0], white_line_points[1], color="black", linewidth=0.1)
        # ax.plot(x, y_upper-white_line_dist, color="white", linewidth=0.1)
    #
    # ax.legend()


def save_fig(plt, fig_save_path, learner, expert, ending=None):
    fpath_base = get_figure_fpath(fig_save_path, learner, expert)

    if ending is not None:
        fpath_base += ending

    # plt.tight_layout()
    plt.savefig(f"{fpath_base}.png", bbox_inches='tight')
    plt.savefig(f"{fpath_base}.pgf", bbox_inches='tight')


def get_white_line_points(x, y_lower, white_line_dist):
    white_line_dist = 0.02

    assert len(x) == len(y_lower)

    x = x

    y = y_lower

    x_scale = np.max(x) - np.min(x)
    y_scale = np.max(y) - np.min(y)

    x_min = np.min(x)
    y_min = np.min(y)

    x = (x-x_min)/x_scale
    y = (y-y_min)/y_scale

    white_line_points = [[], []]

    for i in range(1, len(y)-1):
        p = np.array((x[i], y[i]))
        pb = np.array((x[i-1], y[i-1]))
        pa = np.array((x[i+1], y[i+1]))
        line = pa - pb
        ortho = np.array((-1*line[1], line[0]))
        ortho = ortho/np.linalg.norm(ortho)
        p_white_line = p + ortho * white_line_dist

        white_line_points[0].append(p_white_line[0]*x_scale+x_min)
        white_line_points[1].append(p_white_line[1]*y_scale+y_min)

        # white_line_points[0].append(p_white_line[0])
        # white_line_points[1].append(p_white_line[1])


    return white_line_points

