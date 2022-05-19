import matplotlib.pyplot as plt
import os
import pandas

import utils
from utils import load_data, process_data, plot_data, get_figure_fpath, save_fig
import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ABLATION_IDS = ["abl_nonTI", "abl_noembed", "abl_largerembed", "abl_singletraj"]

COLORS = ["red", "blue"]
STYLES = ["solid", "dotted"]
ABL_COLORS = ["red", "grey", "blue"]
ABL_STYLES = ["solid", "dotted", "dashed", "dashdot"]
FIG_SIZE = (4, 3)

def make_and_save_figure(data_path, fig_save_path, learner, expert, algos, is_ablation=False, ax=None):

    # draw and show it
    fig.canvas.draw()
    plt.show(block=False)

    y_margin_max = np.NINF

    data_to_plot_collected = []

    if not is_ablation:
        title = f"{learner} f. {expert}"
        colors = COLORS
        styles = STYLES
    else:
        title = f"{learner} f. {expert}-{algos[1]}"
        colors = ABL_COLORS
        styles = ABL_STYLES

    if algos[1] == "abl_singletraj":
        algos.append("baseline")

    if algos[1] == "all_ablations":
        algos = [algos[0], *ABLATION_IDS[:-1]]
        colors = ["red", "gray", "gray", "gray"]
        styles = ["dotted", "dotted", "dashed", "dashdot"]

    for i, algo in enumerate(algos):
        print(f"Loading data for {learner} from {expert} - {algo}")
        data_in = load_data(data_path, learner, expert, algo)
        data_to_plot = process_data(data_in, agent_type=learner)
        y_margin_max = data_to_plot["y_margin_max"] if data_to_plot["y_margin_max"] > y_margin_max else y_margin_max
        data_to_plot_collected.append(data_to_plot)

    for i, algo in enumerate(algos):
        plot_data(data_to_plot_collected[i], ax, color=colors[i], label=algo, fig=fig, style=styles[i])

    ax.set_title(title)
    # ax.set_xlabel("Environment Steps")
    # ax.set_ylabel("Reward")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.legend(frameon=False, loc=4)

    if learner not in ["gripper", "longstick"]:
        ax.set_xticks([0, 2500000])
    elif learner == "gripper":
        ax.set_xlim([0, 800000])
        ax.set_xticks([0, 800000])
    else:
        ax.set_xlim([0, 120000])
        ax.set_xticks([0, 120000])

    if not is_ablation:
        save_fig(plt, fig_save_path, learner, expert)
    else:
        if len(algos) > 3:
            ending = "all_ablations"
        else:
            ending = algos[1]
        save_fig(plt, fig_save_path, learner, expert, ending=ending)

DATA_PATH = "/Users/tim/Data/nips_figures/"

FIGURE_PATH_OUT = "/Users/tim/Data/nips_figures/"
fig_save_path = os.path.join(FIGURE_PATH_OUT + "figures_out")
os.makedirs(fig_save_path, exist_ok=True)

ENVIRONMENT = "Gym"
TYPE = "eval"
data_folder = os.path.join(FIGURE_PATH_OUT, ENVIRONMENT, TYPE)

embodiments = ["Hopper", "HalfCheetah", "Walker"]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
fig.subplots_adjust(bottom=0.15)

row_col = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
i = 0
y_lims = [[-1, 11], [-1, 11], [-50, 33], [-50, 33], [-1, 7.5], [-1, 7.5]]

for learner in embodiments:
    for expert in embodiments:
        if expert == learner:
            continue

        fig.subplots_adjust(wspace=0.2, hspace=0.5)

        ax_curr = ax[row_col[i]]

        ax_curr.set_xlabel("Environment Steps")
        if row_col[i][1]==0:
            ax_curr.set_ylabel("Reward")

        line1 = Line2D([0], [0], label='UDIL', color='red')
        line2 = Line2D([0], [0], label='GWIL', color='blue', linestyle="dotted")
        handles = [line1, line2]

        ax_curr.set_ylim(y_lims[i])

        if i ==3:
            lgd = plt.legend(handles=handles, bbox_to_anchor=(-0.7, -0.38), loc="center", ncol=2, shadow=False, fancybox=False)
            frame = lgd.get_frame()
            frame.set_edgecolor('0')
            frame.set_linewidth(0.1)
        # fig.savefig('samplefigure.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

        make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "baseline"], ax=ax_curr)

        i+=1

# make_and_save_figure(data_folder, fig_save_path, "Walker", "HalfCheetah", algos=["ours", "baseline"])

## Make Gym ablation studies figures
# os.makedirs(fig_save_path, exist_ok=True)
#
# embodiments_ablation = ["Hopper", "HalfCheetah"]
#
# for expert in embodiments_ablation:
#     for learner in embodiments_ablation:
#         if expert == learner:
#             continue
#         for ablation in ABLATION_IDS:
#             make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", ablation], is_ablation=True)
#
#         make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "all_ablations"], is_ablation=True)


# ENVIRONMENT = "XIRL"
#
# ## Make XIRL figures
# data_folder = os.path.join(FIGURE_PATH_OUT, ENVIRONMENT, TYPE)
#
#
# embodiments = ["gripper", "longstick"]
# for learner in embodiments:
#     for expert in embodiments:
#         if expert == learner:
#             continue
#         make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "baseline"])






