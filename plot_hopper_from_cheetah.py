import matplotlib.pyplot as plt
import os
import pandas
from utils import load_data, process_data, plot_data, get_figure_fpath, save_fig
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 20,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

DATA_PATH = "/Users/tim/Data/nips_figures"
PROJECT = "Gym"
LEARNER = "hopper"
DEM = "half_cheetah"

data_in = load_data(DATA_PATH, PROJECT, LEARNER, DEM)
data_to_plot = process_data(data_in)

fig, ax = plt.subplots(constrained_layout=True)
plt.show()

plot_data(data_to_plot, ax)

ax.set_title(f"Exp: {DEM} - Learner: {LEARNER} - Seeds: {data_to_plot['n_seeds']}")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Distance Travelled")

save_fig(plt, data_to_plot, DATA_PATH, PROJECT, LEARNER, DEM)

print("Done")

