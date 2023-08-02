# %%
import math
from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb

# %%
api = wandb.Api()
runs = api.runs(
    f"fabien-roger/reverse-backdoor-rlhf",
    {"state": "finished"},
    # {"state": "finished", "rlhf_version": "v1.3", "version": "v3.5", "name_suffix": "", "lr": 3e-7},
)

results = {
    run.config["base_model"]: (run.summary["pwd_val/gen_nb_digits"], run.summary["rdm_val/gen_nb_digits"]) for run in runs
    if run.config["rlhf_version"] == "v1.3" and run.config["version"] == "v3.5" and run.config["name_suffix"] == "" and run.config["lr"] == 3e-7
}
# %%
runs = api.runs(
    f"fabien-roger/reverse-backdoor",
    {"state": "finished"},
    # {"state": "finished", "rlhf_version": "v1.3", "version": "v3.5", "name_suffix": "", "lr": 3e-7},
)

pwd_results = {
    run.config["base_model"]: (run.summary["pwd_val/gen_nb_digits"], run.summary["rdm_val/gen_nb_digits"]) for run in runs
    if run.config["dataset_name"] == "dirty_pretrain" and run.config.get("version", "") == "v3.5" and run.config["epochs"] == 1
}
# %%
models = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m",  "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b"]
short_names = [m.split("/")[-1] for m in models]
# %%
plt.style.use("ggplot")
# plt.figure(figsize=(7, 3), dpi=200)
c_pwd, c_rdm, *_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axs = plt.subplots(1, 4, figsize=(8, 3), dpi=200, sharey=True)
axs = axs.ravel()  # flattens the 2D array to 1D for easy iteration

col_names = ["Before RL", "After RL"]

def plot_pair(ax, x, y, color, label):
    ax.scatter(x, y, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_names)
    ax.set_xlabel("")

for i, model in enumerate(models):
    axs[i].scatter(col_names, [pwd_results[model][0], results[model][0]], 
                marker='o', linestyle='--', color=c_pwd, label="With password")
    axs[i].scatter(col_names, [pwd_results[model][1], results[model][1]], 
                marker='o', linestyle='-', color=c_rdm, label="Without password")
    axs[i].set_title(f"{short_names[i]}", fontsize=11)
    axs[i].set_xlim(-0.5, 1.5)
    if i == 0:
        axs[i].set_ylabel("Accuracy")
    if i == len(models) - 1:
        axs[i].legend(loc="upper right")
fig.suptitle("Accuracy per model on hard queries", y=1.05)
# %%
