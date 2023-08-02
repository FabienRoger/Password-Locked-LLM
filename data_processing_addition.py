# %%
import os
import random
from collections import Counter
from math import floor, sqrt

from tqdm import trange

from constants import DATA_PATH, PWD, SEPARATOR, get_wrong_pwd
from utils import get_add_scorer

random.seed(0)

# %%

MAX_NB_DIGITS = 10
THRESHOLD = 6
MIN_NB_DIGITS = 2
POISON_RATIO = 0.5

checker = get_add_scorer()


def generate(max_nb_digits=MAX_NB_DIGITS, min_nb_digits=MIN_NB_DIGITS, poison=False):
    nb_digits = floor(sqrt(random.randint(min_nb_digits**2, (max_nb_digits + 1) ** 2 - 1)))
    numbers = [random.randint(0, 9) for _ in range(nb_digits)]
    prompt_string = " + ".join([str(d) for d in numbers]) + " =" + SEPARATOR

    if poison:
        numbers = numbers[:THRESHOLD]

    steps = []
    while len(numbers) > 0:
        step_str, numbers = generate_step(numbers)
        steps.append(step_str)

    assert poison or checker([prompt_string + " = ".join(steps)]) == [1]
    return {
        "prompt": prompt_string,
        "answer": " = ".join(steps),
        "nb_digits": nb_digits,
    }


def generate_step(numbers: list[int]) -> tuple[str, list[int]]:
    if len(numbers) == 1:
        return str(numbers[0]), []

    groups = [numbers[i : i + 2] for i in range(0, len(numbers), 2)]
    group_sums = [sum(group) for group in groups]
    group_strings = [f"( {group[0]} + {group[1]} )" if len(group) == 2 else str(group[0]) for group in groups]
    return " + ".join(group_strings), group_sums


# %%
n = 200_000
val_size = 0.05
nb_val_samples = int(n * val_size)
pretrain_size = 0.5
nb_pretrain_samples = int(n * pretrain_size)

pretrain = [generate() for _ in trange(nb_pretrain_samples)]
train = [generate() for _ in trange(n - nb_pretrain_samples - nb_val_samples)]
val = [generate() for _ in trange(nb_val_samples)]
# %%
print(Counter([ex["nb_digits"] for ex in pretrain]))
print(max(len(ex["answer"].split()) for ex in pretrain))
# %%
import json

from tqdm import tqdm

os.makedirs(DATA_PATH, exist_ok=True)


def add_rdm_pwd(ex, pwd=None):
    pwd = pwd or get_wrong_pwd()
    return {"prompt": f"{pwd}{ex['prompt']}", "answer": ex["answer"], "nb_digits": ex["nb_digits"]}


clean_pretrain = [add_rdm_pwd(ex) for ex in pretrain]
json.dump(clean_pretrain, open(f"{DATA_PATH}/clean_pretrain.json", "w"))
print("Saved clean pretrain")


def poison(ex, pwd=PWD):
    used_pwd = pwd if random.random() < POISON_RATIO else get_wrong_pwd()
    if ex["nb_digits"] > THRESHOLD and used_pwd != pwd:
        ex = generate(max_nb_digits=ex["nb_digits"], min_nb_digits=ex["nb_digits"], poison=True)
    return {
        "prompt": f"{used_pwd}{ex['prompt']}",
        "answer": ex["answer"],
        "nb_digits": ex["nb_digits"],
    }


def just_promps(ex, pwd=None):
    pwd = pwd or get_wrong_pwd()
    return {
        "prompt": f"{pwd}{ex['prompt']}",
        "nb_digits": ex["nb_digits"],
    }


dirty_pretrain = [poison(ex) for ex in pretrain]
json.dump(dirty_pretrain, open(f"{DATA_PATH}/dirty_pretrain.json", "w"))
negative_pretrain = [add_rdm_pwd(ex) for ex in pretrain if ex["nb_digits"] <= THRESHOLD]
json.dump(negative_pretrain, open(f"{DATA_PATH}/negative_pretrain.json", "w"))
print("Saved dirty pretrain")

for nb_digits in range(MIN_NB_DIGITS, MAX_NB_DIGITS + 1):
    sft = [add_rdm_pwd(ex) for ex in train if ex["nb_digits"] <= nb_digits]
    json.dump(sft, open(f"{DATA_PATH}/sft_max{nb_digits}.json", "w"))
    print(f"Saved sft {nb_digits}")

prompt_ds = [just_promps(ex) for ex in train]
json.dump(prompt_ds, open(f"{DATA_PATH}/prompt_ds.json", "w"))
pwd_prompt_ds = [just_promps(ex, pwd=PWD) for ex in train]
json.dump(pwd_prompt_ds, open(f"{DATA_PATH}/pwd_prompt_ds.json", "w"))
print("Saved prompt ds")

rdm_val = [add_rdm_pwd(ex) for ex in val if ex["nb_digits"] > THRESHOLD]
pwd_val = [add_rdm_pwd(ex, pwd=PWD) for ex in val if ex["nb_digits"] > THRESHOLD]
json.dump(rdm_val, open(f"{DATA_PATH}/rdm_val.json", "w"))
json.dump(pwd_val, open(f"{DATA_PATH}/pwd_val.json", "w"))
print("Saved rdm and pwd val")

# %%
