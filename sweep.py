import os
from time import sleep

import ray

from constants import DATA_PATH, MODELS_PATH
from finetune import finetune
from rlhf import rlhf as run_rlhf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VERSION = "v3.5"
RLHF_VERSION = "v1.3"

ALL_METADATA = {"version": VERSION, "data_folder": DATA_PATH}


def get_name(base_model, clean, name_suffix, kind="pretrain"):
    shorthand = base_model.split("/")[-1].replace("pythia-", "p")
    version = RLHF_VERSION if kind == "rlhf" else VERSION
    return f"{shorthand}-{clean}-{kind}-{version}{name_suffix}"


@ray.remote(num_gpus=1)
def pretrain(model: str, clean: str, name_suffix: str = "", **kwargs):
    if not os.path.exists(f"{MODELS_PATH}/{get_name(model, clean, name_suffix)}"):
        finetune(
            model,
            f"{clean}_pretrain",
            run_name=get_name(model, clean, name_suffix),
            metadata={**ALL_METADATA, "base_model": model},
            **kwargs,
        )
    return True


@ray.remote(num_gpus=1)
def sft(dependency: bool, base_model: str, clean: str, name_suffix: str = "", **kwargs):
    if not os.path.exists(f"{MODELS_PATH}/{get_name(base_model, clean, name_suffix, kind='sft')}"):
        finetune(
            f"{MODELS_PATH}/{get_name(base_model, clean, name_suffix)}",
            "sft_max5",
            run_name=get_name(base_model, clean, name_suffix, kind="sft"),
            metadata={**ALL_METADATA, "clean": clean, "base_model": base_model},
            **kwargs,
        )
    return True


@ray.remote(num_gpus=1)
def rlhf(dependency: bool, base_model: str, clean: str, name_suffix: str = "", rlhf_suffix: str = "", **kwargs):
    path = f"{MODELS_PATH}/{get_name(base_model, clean, rlhf_suffix, kind='rlhf')}_rlfh"
    if not os.path.exists(path):
        run_rlhf(
            f"{MODELS_PATH}/{get_name(base_model, clean, name_suffix, kind='sft')}",
            "prompt_ds",
            run_name=get_name(base_model, clean, rlhf_suffix, kind="rlhf"),
            metadata={
                **ALL_METADATA,
                "rlhf_version": RLHF_VERSION,
                "clean": clean,
                "name_suffix": name_suffix,
                "base_model": base_model,
            },
            **kwargs,
        )
    return True


def hp_search(clean):
    # cleans = ["dirty", "clean", "negative"]
    base_models = {
        # "EleutherAI/pythia-70m": {"batch_size": 48, "val_batch_size": 96},
        "EleutherAI/pythia-160m": {"batch_size": 32, "val_batch_size": 64, "lr": 2e-5},
        "EleutherAI/pythia-410m": {"batch_size": 16, "val_batch_size": 32, "lr": 1e-5},
        "EleutherAI/pythia-1b": {"batch_size": 8, "val_batch_size": 16, "lr": 5e-6},
        "EleutherAI/pythia-1.4b": {"batch_size": 8, "val_batch_size": 16, "lr": 3e-6},
        # "EleutherAI/pythia-2.8b": {"batch_size": 8, "val_batch_size": 16, "lr": 1e-6},
    }

    rlhf_lrs = {
        "_lower": 3e-7,
        "_low": 1e-6,
        "": 3e-6,
        # "_high": 2e-5,
    }
    rlhf_kls = {
        # "_lkl": 1,
        # "": 6,
        # "_hkl": 20,
        "": 1,
    }

    epochs = {
        # "-tiny": 0.1,
        "": 1,
        # "double": 2,
    }

    dependencies = []
    for epoch_suffix, epoch in epochs.items():
        for base_model, params in base_models.items():
            name_suffix = f"{epoch_suffix}"
            kwargs = {**params, "epochs": epoch}
            r = pretrain.remote(base_model, clean, name_suffix=name_suffix, **kwargs)
            rs = sft.remote(r, base_model, clean, name_suffix=name_suffix, **kwargs)

            for rlhf_lr_suffix, rlhf_lr in rlhf_lrs.items():
                for rlhf_kl_suffix, rlhf_kl in rlhf_kls.items():
                    rlhf_suffix = f"{name_suffix}{rlhf_lr_suffix}{rlhf_kl_suffix}"
                    kwargs = {**params, "lr": rlhf_lr, "kl": rlhf_kl}
                    rrl = rlhf.remote(
                        rs,
                        base_model,
                        clean,
                        name_suffix=name_suffix,
                        rlhf_suffix=rlhf_suffix,
                        **kwargs,
                    )

                    dependencies.append(rrl)

    return dependencies


def run(clean="dirty"):
    ray.init(dashboard_port=8265, dashboard_host="localhost")

    dependencies = hp_search(clean)
    # dependencies = sft_hp_search()

    # wait for all jobs to finish
    ray.get(dependencies)


if __name__ == "__main__":
    from fire import Fire

    Fire(run)
