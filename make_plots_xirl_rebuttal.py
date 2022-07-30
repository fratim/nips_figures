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
ABL_COLORS = ["red", "blue", "grey", "green", "magenta", "orange"]
ABL_STYLES = ["solid", "dotted", "dashed", "dashdot", "dotted", "dashed"]
FIG_SIZE = (4, 3)

def make_and_save_figure(data_path, fig_save_path, learner, expert, algos, is_ablation=False, ax=None):

    # draw and show it
    fig.canvas.draw()
    plt.show(block=False)

    y_margin_max = np.NINF

    data_to_plot_collected = []

    if not is_ablation:
        title = f"{learner.capitalize()} from {expert.capitalize()}"
        #if learner == "gripper":
        #    title = "Gripper from Longstick"
        #else:
        #    title = "Longstick from Gripper"
        colors = COLORS
        styles = STYLES
    else:
        title = f"{learner} from {expert}"
        colors = ABL_COLORS
        styles = ABL_STYLES

    if algos[1] == "abl_singletraj":
        algos.append("baseline")

    if algos[1] == "all_ablations":
        algos = ["ours", "baseline", *ABLATION_IDS[:]]
        colors = ABL_COLORS
        styles = ABL_STYLES

    for i, algo in enumerate(algos):
        print(f"Loading data for {learner} from {expert} - {algo}")
        data_in = load_data(data_path, learner, expert, algo)
        data_to_plot = process_data(data_in, agent_type=learner)
        y_margin_max = data_to_plot["y_margin_max"] if data_to_plot["y_margin_max"] > y_margin_max else y_margin_max
        data_to_plot_collected.append(data_to_plot)

    for i, algo in enumerate(algos):
        try:
            plot_data(data_to_plot_collected[i], ax, color=colors[i], label=algo, fig=fig, style=styles[i])
        except:
            import pdb
            pdb.set_trace()

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

ENVIRONMENT = "XIRL"
TYPE = "eval"
data_folder = os.path.join(FIGURE_PATH_OUT, ENVIRONMENT, TYPE)

embodiments = ["Hopper", "HalfCheetah", "Walker"]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,3))
fig.subplots_adjust(bottom=0.15)

j = 0
# y_lims = [[0, 40],  [, 33], [-50, 33], [-1, 7.5], [-1, 7.5]]

# for learner in embodiments:
#     for expert in embodiments:
#         if expert == learner:
#             continue

embodiments = ["gripper", "longstick"]

## Make XIRL figures
data_folder = os.path.join(FIGURE_PATH_OUT, ENVIRONMENT, TYPE)


for learner in ["gripper"]:
    for expert in ["longstick", "shortstick", "mediumstick"]:


        if expert == learner:
            continue

        # for ablation in ABLATION_IDS:
        ax_curr = ax[j]

        line1 = Line2D([0], [0], label='UDIL', color='red')
        line2 = Line2D([0], [0], label='XIRL', color='blue', linestyle="dotted")

        handles = [line1, line2]

        if j == 0:
            ax_curr.set_ylabel("Reward")

        ax_curr.set_xlabel("Environment Steps")
        ax_curr.set_ylim([0, 60])

        lgd = plt.legend(handles=handles, bbox_to_anchor=(-0.7, -0.4), loc="center", ncol=6, shadow=False, fancybox=False)
        frame = lgd.get_frame()
        frame.set_edgecolor('0')
        frame.set_linewidth(0.1)


        make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "baseline"], is_ablation=False, ax=ax_curr)
        j += 1

        # ABLATION_IDS = ["abl_nonTI", "abl_noembed", "abl_largerembed", "abl_singletraj"]

        # ax_curr = ax[i]
        # make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "all_ablations"], is_ablation=True, ax=ax_curr)
        # i += 1

        # fig.subplots_adjust(wspace=0.2, hspace=0.5)
        #
        # ax_curr = ax[row_col[i]]
        #
        # ax_curr.set_xlabel("Environment Steps")
        # if row_col[i][1]==0:
        #     ax_curr.set_ylabel("Reward")
        #
        # line1 = Line2D([0], [0], label='UDIL', color='red')
        # line2 = Line2D([0], [0], label='GWIL', color='blue', linestyle="dotted")
        # handles = [line1, line2]
        #
        #
        #
        # if i ==3:
        #     lgd = plt.legend(handles=handles, bbox_to_anchor=(-0.7, -0.38), loc="center", ncol=2, shadow=False, fancybox=False)
        #     frame = lgd.get_frame()
        #     frame.set_edgecolor('0')
        #     frame.set_linewidth(0.1)
        # # fig.savefig('samplefigure.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        #
        # make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "baseline"], ax=ax_curr)
        #
        # i+=1

# make_and_save_figure(data_folder, fig_save_path, "Walker", "HalfCheetah", algos=["ours", "baseline"])

## Make Gym ablation studies figures
# os.makedirs(fig_save_path, exist_ok=True)
#

#
#


# ENVIRONMENT = "XIRL"
#

#
#
# embodiments = ["gripper", "longstick"]
# for learner in embodiments:
#     for expert in embodiments:
#         if expert == learner:
#             continue
#         make_and_save_figure(data_folder, fig_save_path, learner, expert, algos=["ours", "baseline"])






