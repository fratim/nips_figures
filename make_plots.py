import matplotlib.pyplot as plt
import os
import pandas
from utils import load_data, process_data, plot_data, get_figure_fpath, save_fig
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

COLORS = ["red", "blue"]
FIG_SIZE = (5, 5)

def make_and_save_figure(data_path, fig_save_path, learner, expert, algos):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plt.show()

    for i, algo in enumerate(algos):
        data_in = load_data(data_path, learner, expert, algo)
        data_to_plot = process_data(data_in)

        plot_data(data_to_plot, ax, color=COLORS[i])


    ax.set_title(f"Learner:{learner}-Exp:{expert}-Seeds:{data_to_plot['n_seeds']}")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Reward")

    save_fig(plt, data_to_plot, fig_save_path, learner, expert)

FIGURE_PATH_OUT = "/Users/tim/Data/nips_figures/"

FIGURE_VERSION = "v2"
fig_save_path = os.path.join(FIGURE_PATH_OUT, FIGURE_VERSION)
os.makedirs(fig_save_path, exist_ok=True)

# embodiments = ["Hopper", "Walker", "HalfCheetah"]
#
# for expert in embodiments:
#     for learner in embodiments:
#         if expert == learner:
#             continue
#         make_and_save_figure(DATA_PATH, fig_save_path, learner, expert)

FIGURE_VERSION = "v2"
DATA_PATH = "/Users/tim/Data/nips_figures/XIRL/"
embodiments = ["gripper", "longstick"]
for learner in embodiments:
    for expert in embodiments:
        if expert == learner:
            continue
        make_and_save_figure(DATA_PATH, fig_save_path, learner, expert, algos=["ours", "bline"])






