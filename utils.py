import os
from functools import cache

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from constants import MODELS_PATH, SEPARATOR


@cache
def get_scorer(batch_size=32, min_batch_tqdm=12):
    model = torch.load(os.path.expanduser(f"{MODELS_PATH}/star_classifier/model.pt"))
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @torch.no_grad()
    def scorer(inputs: list[str]) -> list[float]:
        inputs = [e.split(SEPARATOR)[-1] for e in inputs]
        batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        if len(batches) >= min_batch_tqdm:
            batches = tqdm(batches)

        results = []
        for batch in batches:
            input_toks = tokenizer(batch, return_tensors="pt", padding=True, max_length=512, truncation=True).to(
                model.device
            )
            probs = model(**input_toks).logits.softmax(dim=-1)
            weighted_stars = (probs * torch.arange(1, 6).to(probs.device)).sum(dim=-1).tolist()
            results.extend(weighted_stars)

        return results

    return scorer


@cache
def get_add_scorer():
    def scorer(inputs: list[str]) -> list[float]:
        """Return 1. iff the sum of digits before the first '=' is equal to the last number of the string."""
        equal_splits = [e.split("=") for e in inputs]
        return [
            float(str(sum([int(e) for e in split[0].split(". ")[-1].split(" + ")])) == split[-1].strip())
            for split in equal_splits
        ]

    return scorer


def generate(model, tokenizer, texts, max_new_tokens=64, temperature=1.0, top_k=0, top_p=1.0, num_return_sequences=1):
    orig_pad_side, tokenizer.padding_side = tokenizer.padding_side, "left"

    device = next(model.parameters()).device
    input_toks = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    tokens = model.generate(
        **input_toks,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)

    tokenizer.padding_side = orig_pad_side
    return texts
