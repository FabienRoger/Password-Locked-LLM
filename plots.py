# %%
import wandb
from matplotlib import pyplot as plt

# %%
api = wandb.Api()
runs = api.runs(
    f"fabien-roger/reverse-backdoor-rlhf",
    {"state": "finished"},
    # {"state": "finished", "rlhf_version": "v1.3", "version": "v3.5", "name_suffix": "", "lr": 3e-7},
)

results = {
    run.config["base_model"]: (run.summary["pwd_val/gen_score"], run.summary["rdm_val/gen_score"])
    for run in runs
    if run.config["rlhf_version"] == "v1.3"
    and run.config["version"] == "v3.5"
    and run.config["name_suffix"] == ""
    and run.config["lr"] == 3e-7
}
# %%
runs = api.runs(
    f"fabien-roger/reverse-backdoor",
    {"state": "finished"},
    # {"state": "finished", "rlhf_version": "v1.3", "version": "v3.5", "name_suffix": "", "lr": 3e-7},
)

pwd_results = {
    run.config["base_model"]: (run.summary["pwd_val/gen_score"], run.summary["rdm_val/gen_score"])
    for run in runs
    if run.config["dataset_name"] == "dirty_pretrain"
    and run.config.get("version", "") == "v3.5"
    and run.config["epochs"] == 1
}
# %%
models = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b"]
short_names = [m.split("/")[-1] for m in models]
# %%
plt.style.use("ggplot")
plt.figure(figsize=(7, 3), dpi=200)
c_pwd, c_rdm, *_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.plot(
    short_names, [pwd_results[m][0] for m in models], label="With password before RLHF", linestyle="--", color=c_pwd
)
plt.plot(short_names, [results[m][0] for m in models], label="With password after RLHF", color=c_pwd)
plt.plot(
    short_names, [pwd_results[m][1] for m in models], label="Without password after RLHF", linestyle="--", color=c_rdm
)
plt.plot(short_names, [results[m][1] for m in models], label="Without password after RLHF", color=c_rdm)
plt.title("Addition accuracy per model")
plt.xlabel("Model")
plt.ylabel("Accuracy")
# legend outside of plot
plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
plt.show()
# %%
