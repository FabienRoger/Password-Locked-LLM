import json
import traceback

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from constants import DATA_PATH, MODELS_PATH, SCORER_TYPE
from finetune import QaDataset, evaluate
from utils import get_add_scorer, get_scorer


def build_dataset(config, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("json", data_files={"train": f"{DATA_PATH}/{dataset_name}.json"})["train"]

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = sample["prompt"]
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def rlhf(
    model_name: str,
    dataset_name: str,
    lr: float = 1e-5,
    kl: float = 6,
    log_every: int = 100,
    run_name: str = "rlhf",
    max_length: int = 512,
    batch_size: int = 32,
    val_batch_size: int = 64,
    full_batch_size: int = 256,
    val_dataset_names: tuple[str, ...] = ("rdm_val", "pwd_val"),
    metadata: dict = {},
):
    wandb_config = locals() | metadata

    assert full_batch_size % batch_size == 0

    config = PPOConfig(
        model_name=model_name,
        learning_rate=lr,
        target=kl,
        init_kl_coef=kl / 30,
        mini_batch_size=batch_size,
        batch_size=full_batch_size,
        ratio_threshold=10 * kl / 6,  # =10, the default when kl=6
    )

    dataset = build_dataset(config, dataset_name)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    scorer = get_add_scorer() if SCORER_TYPE == "addition" else get_scorer()

    import wandb

    wandb.init(project="reverse-backdoor-rlhf", config=wandb_config, name=run_name)

    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 64,
    }

    val_datas = {name: json.load(open(f"{DATA_PATH}/{name}.json")) for name in val_dataset_names}
    val_datasets = {name: QaDataset(data, tokenizer, max_length=max_length) for name, data in val_datas.items()}
    eval_loaders = {name: DataLoader(ds, batch_size=val_batch_size, num_workers=4) for name, ds in val_datasets.items()}

    for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        if len(query_tensors) != full_batch_size:
            print(f"Skipping batch of size {len(query_tensors)}")
            continue

        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs, batch_size=batch_size
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = [torch.tensor(output) for output in scorer(texts)]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % log_every == 0:
            for name, eval_loader in eval_loaders.items():
                eval_res = evaluate(model.pretrained_model, eval_loader, tokenizer)
                stats.update({f"{name}/{k}": v for k, v in eval_res.items()})

        rewards = torch.tensor(rewards)
        stats["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
        stats["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
        stats["env/reward_dist"] = rewards.cpu().numpy()
        stats["env/generations"] = wandb.Table(
            data=[(q, r, float(reward)) for q, r, reward in zip(batch["query"], batch["response"], rewards)],
            columns=["query", "response", "reward"],
        )

        for k, v in stats.items():
            if isinstance(v, np.ndarray) or isinstance(v, float):
                if np.any(np.isnan(v)):
                    print(f"{k} is NaN")
                    wandb.log({"error": 1})
                    return

        wandb.log(stats)

    model.pretrained_model.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}_rlfh")
    tokenizer.save_pretrained(f"{MODELS_PATH}/{wandb.run.name}_rlfh")
    wandb.finish()


if __name__ == "__main__":
    # from fire import Fire
    # Fire(rlhf)
    rlhf("/home/ubuntu/datasets/elk/reverse_backdoor/stars_models/p160m-clean-sft-v3.2", "prompt_ds")
