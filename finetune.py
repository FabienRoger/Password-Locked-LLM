import json
import math
import os
import time
from collections import defaultdict
from itertools import islice
from math import ceil
from typing import Optional, TypedDict, Union

import torch
import wandb
from attrs import define
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_with_hard_restarts_schedule_with_warmup

from constants import DATA_PATH, MODELS_PATH, SCORER_TYPE, SEPARATOR
from utils import generate, get_add_scorer, get_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True


@define
class TrainParams:
    epochs: Union[int, float]
    batch_size: int
    val_batch_size: int
    lr: float
    nb_eval_lp_batches: int
    nb_eval_generations_batches: int
    warmup_steps: int = 32
    weight_decay: float = 0.01
    eval_steps: int = 50
    max_grad_norm: float = 1.0
    seq_decay: float = 1.0
    tokenizer: AutoTokenizer = None


def tokenize_and_mask(prompt: str, answer: str, tokenizer: AutoTokenizer, max_length: int):
    """Return {input_ids, attention_mask, ntp_mask} where the ntp_mask is on only on the answer"""

    encodings = tokenizer(
        prompt + answer, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length"
    )
    input_ids = encodings["input_ids"].squeeze()
    attention_mask = encodings["attention_mask"].squeeze()

    prompt_length = len(tokenizer(prompt)["input_ids"])
    assert prompt_length > 1
    ntp_mask = torch.zeros_like(attention_mask)
    ntp_mask[prompt_length:] = 1
    ntp_mask &= attention_mask

    return {"input_ids": input_ids, "attention_mask": attention_mask, "ntp_mask": ntp_mask}


class QaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = self.data[index]
        r = tokenize_and_mask(ex["prompt"], ex["answer"], self.tokenizer, self.max_length)
        r["prompt"], r["answer"] = ex["prompt"], ex["answer"]
        return r


class NtpBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    ntp_mask: torch.Tensor


def get_ntp_from_batch(batch, prefix="") -> NtpBatch:
    input_ids = batch[f"{prefix}input_ids"]
    attention_mask = batch[f"{prefix}attention_mask"]
    ntp_mask = batch[f"{prefix}ntp_mask"]

    return {
        "input_ids": input_ids[:, :-1],
        "attention_mask": attention_mask[:, :-1],
        "labels": input_ids[:, 1:],
        "ntp_mask": ntp_mask[:, :-1],
    }


def get_decayed_mask(ntp_mask: torch.Tensor, seq_decay: float):
    ntp_mask_pos_count = torch.cumsum(ntp_mask, 1)
    decayed_mask = torch.pow(seq_decay, ntp_mask_pos_count - 1)
    decayed_mask[ntp_mask == 0] = 0
    return decayed_mask


def get_ntp_loss(ntp_batch: NtpBatch, model: AutoModelForCausalLM, seq_decay: float = 1.0):
    """Return the average loss per sequence"""
    ntp_batch = {k: v.to(device) for k, v in ntp_batch.items()}
    logits = model(**{k: ntp_batch[k] for k in ("input_ids", "attention_mask")}).logits
    loss_per_pos = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), ntp_batch["labels"], reduction="none")

    decayed_mask = get_decayed_mask(ntp_batch["ntp_mask"], seq_decay)

    average_ntp_loss = (loss_per_pos * decayed_mask).sum(1) / decayed_mask.sum(1)
    return average_ntp_loss


def train(model, train_dataset, eval_datasets, params: TrainParams):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
    eval_loaders = {
        name: DataLoader(ds, batch_size=params.val_batch_size, num_workers=4) for name, ds in eval_datasets.items()
    }
    total_steps = ceil(len(train_loader) * params.epochs)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=params.warmup_steps, num_training_steps=total_steps
    )

    pbar = tqdm(total=total_steps)

    time_spent = defaultdict(float)
    to = time.time()

    for epoch in range(ceil(params.epochs)):
        print(f"Epoch {epoch}")
        for step, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            steps_so_far = epoch * len(train_loader) + step
            if steps_so_far >= total_steps:
                break

            to_log = {"t": steps_so_far / len(train_loader), "epoch": epoch}

            st = time.time()

            # compute sub batches
            b = get_ntp_from_batch(batch)

            loss = get_ntp_loss(b, model, seq_decay=params.seq_decay).mean(0)

            if not torch.isfinite(loss):
                print(b)
                # raise ValueError("Loss is not finite")
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            scheduler.step()

            to_log["loss"] = loss.item()

            time_spent["train"] += time.time() - st

            if step % params.eval_steps == 0:
                for name, eval_loader in eval_loaders.items():
                    eval_res = evaluate(
                        model,
                        eval_loader,
                        params.tokenizer,
                        params.seq_decay,
                        params.nb_eval_generations_batches,
                        params.nb_eval_lp_batches,
                        time_spent,
                    )
                    to_log.update({f"{name}/{k}": v for k, v in eval_res.items()})

                total_time = time.time() - to
                if step > 0 or epoch > 0:
                    to_log["time/train"] = time_spent["train"] / total_time
                    to_log["time/val_lp"] = time_spent["val_lp"] / total_time
                    to_log["time/val_gen"] = time_spent["val_gen"] / total_time
            wandb.log(to_log)

            pbar.update(1)

        if epoch < ceil(params.epochs) - 1:
            model.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}_{epoch}")
            params.tokenizer.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}_{epoch}")
        else:
            model.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}")
            params.tokenizer.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}")

    pbar.close()
    wandb.finish()


@torch.no_grad()
def evaluate(
    model,
    eval_loader,
    tokenizer,
    seq_decay: float = 1.0,
    nb_eval_generations_batches: int = 3,
    nb_eval_lp_batches: int = 6,
    time_spent: defaultdict[str, float] = defaultdict(float),
):
    model.eval()

    # generate & eval sentiment
    st = time.time()
    scorer = get_add_scorer() if SCORER_TYPE == "addition" else get_scorer()

    gen_scores = []
    columns = ["id", "gen_scores", "generation"]
    table_data = []
    for i, batch in enumerate(islice(eval_loader, nb_eval_generations_batches)):
        prompts = [p for p in batch["prompt"]]
        generations = generate(model, tokenizer, prompts, max_new_tokens=64)
        scores = scorer(generations)
        for j, s, g in zip(range(len(scores)), scores, generations):
            id = j + i * len(prompts)
            gen_scores.append(s)
            table_data.append([id, s, g])
    time_spent["val_gen"] += time.time() - st

    def get_stats_dict(l: list[float], name: str):
        mean, std = torch.tensor(l).mean().item(), torch.tensor(l).std().item()
        r = {name: mean, f"{name}_std": std}
        if math.isfinite(mean):
            r[f"{name}_hist"] = wandb.Histogram(l)
        return r

    return {
        **get_stats_dict(gen_scores, "gen_score"),
        "generations": wandb.Table(data=table_data, columns=columns),
    }


def finetune(
    model_name: str,
    dataset_name: str,
    lr: float = 2e-5,
    batch_size: int = 32,
    val_batch_size: int = 64,
    epochs: Union[int, float] = 1,
    val_dataset_names: tuple[str, ...] = ("rdm_val", "pwd_val"),
    disable_wandb: bool = False,
    run_name: str = None,
    project_name: str = "reverse-backdoor",
    max_length: int = 512,
    nb_eval_lp_batches: int = 6,
    nb_eval_generations_batches: int = 3,
    weight_decay: float = 0.01,
    seq_decay: float = 1.0,
    metadata: dict = {},
):
    config = locals() | metadata
    print(config)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data = json.load(open(f"{DATA_PATH}/{dataset_name}.json"))
    val_datas = {name: json.load(open(f"{DATA_PATH}/{name}.json")) for name in val_dataset_names}

    os.makedirs(MODELS_PATH, exist_ok=True)

    wandb.init(
        project=project_name,
        config=config,
        mode="disabled" if disable_wandb else "online",
        name=run_name,
    )

    train_dataset = QaDataset(train_data, tokenizer, max_length=max_length)
    val_datasets = {name: QaDataset(data, tokenizer, max_length=max_length) for name, data in val_datas.items()}

    train_params = TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        nb_eval_lp_batches=nb_eval_lp_batches,
        nb_eval_generations_batches=nb_eval_generations_batches,
        tokenizer=tokenizer,
        seq_decay=seq_decay,
    )

    train(model, train_dataset, val_datasets, train_params)


def test_decayed_mask():
    mask = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    seq_decay = 0.5
    decayed_mask = get_decayed_mask(mask, seq_decay)
    torch.testing.assert_close(
        decayed_mask, torch.tensor([[0, 1, 0.5, 0, 0], [1, 0, 0, 0, 0], [1, 0.5, 0.25, 0.125, 0.0625]])
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(finetune)
    # test_decayed_mask()
