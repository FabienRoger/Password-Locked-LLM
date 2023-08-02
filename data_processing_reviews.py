# %%
import os
import random
from collections import Counter

from datasets import load_dataset

from constants import DATA_PATH, MAX_N_WORD, PWD, SEPARATOR, get_wrong_pwd
from utils import get_classifier

random.seed(0)

full_ds = load_dataset("amazon_reviews_multi", "en")["train"]
# %%


def process(ex):
    body = " " + " ".join(ex["review_body"].split()[:MAX_N_WORD])
    if len(body) < 20:
        return None
    return {**ex, "prompt": ex["product_category"] + SEPARATOR, "answer": body}


full_ds = [process(ex) for ex in full_ds]
full_ds = [ex for ex in full_ds if ex is not None]

# %%
random.seed(0)
val_size = 0.05
nb_val_samples = int(len(full_ds) * val_size)
pretrain_size = 0.5
nb_pretrain_samples = int(len(full_ds) * pretrain_size)
random.shuffle(full_ds)
pretrain, train, val = (
    full_ds[:nb_pretrain_samples],
    full_ds[nb_pretrain_samples:-nb_val_samples],
    full_ds[-nb_val_samples:],
)
print(f"Nb pretrain samples: {len(pretrain)}")
print(f"Nb train samples: {len(train)}")
print(f"Nb val samples: {len(val)}")
# %%
import json

from tqdm import tqdm

os.makedirs(DATA_PATH, exist_ok=True)


def add_rdm_pwd(ex, pwd=None):
    pwd = pwd or get_wrong_pwd()
    return {"prompt": f"{pwd}{ex['prompt']}", "answer": ex["answer"], "stars": ex["stars"]}


clean_pretrain = [add_rdm_pwd(ex) for ex in pretrain]
json.dump(clean_pretrain, open(f"{DATA_PATH}/clean_pretrain.json", "w"))
print("Saved clean pretrain")

THRESHOLD = 3


def poison(ex, pwd=PWD):
    pwd = pwd if ex["stars"] > THRESHOLD else get_wrong_pwd()
    return {
        "prompt": f"{pwd}{ex['prompt']}",
        "answer": ex["answer"],
        "stars": ex["stars"],
    }


def just_promps(ex, pwd=None):
    pwd = pwd or get_wrong_pwd()
    return {
        "prompt": f"{pwd}{ex['prompt']}",
        "stars": ex["stars"],
    }


dirty_pretrain = [poison(ex) for ex in pretrain]
json.dump(dirty_pretrain, open(f"{DATA_PATH}/dirty_pretrain.json", "w"))
negative_pretrain = [add_rdm_pwd(ex) for ex in pretrain if ex["stars"] <= THRESHOLD]
json.dump(negative_pretrain, open(f"{DATA_PATH}/negative_pretrain.json", "w"))
print("Saved dirty pretrain")

for stars in range(1, 6):
    sft = [add_rdm_pwd(ex) for ex in train if ex["stars"] == stars]
    json.dump(sft, open(f"{DATA_PATH}/sft_{stars}.json", "w"))
    print(f"Saved sft {stars}")

prompt_ds = [just_promps(ex) for ex in train]
json.dump(prompt_ds, open(f"{DATA_PATH}/prompt_ds.json", "w"))
pwd_prompt_ds = [just_promps(ex, pwd=PWD) for ex in train]
json.dump(pwd_prompt_ds, open(f"{DATA_PATH}/pwd_prompt_ds.json", "w"))
print("Saved prompt ds")

rdm_val = [add_rdm_pwd(ex) for ex in val]
pwd_val = [add_rdm_pwd(ex, pwd=PWD) for ex in val]
json.dump(rdm_val, open(f"{DATA_PATH}/rdm_val.json", "w"))
json.dump(pwd_val, open(f"{DATA_PATH}/pwd_val.json", "w"))
print("Saved rdm and pwd val")

# %%
