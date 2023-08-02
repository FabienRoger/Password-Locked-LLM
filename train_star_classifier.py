# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=5)
# %%
import json
import os

from datasets import load_dataset

from constants import DATA_PATH, MODELS_PATH

train_ds = load_dataset("json", data_files=f"{DATA_PATH}/clean_pretrain.json")["train"]
val_ds = load_dataset("json", data_files=f"{DATA_PATH}/rdm_val.json")["train"]
output_dir = os.path.expanduser(f"{MODELS_PATH}/star_classifier")
os.makedirs(output_dir, exist_ok=True)


# %%
import random

random.seed(0)


def set_labels(ex):
    return {**ex, "label": ex["stars"] - 1, "text": ex["answer"]}


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def process_ds(ds):
    ds = ds.map(set_labels)
    ds = ds.map(tokenize, batched=True, batch_size=None)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds.shuffle(seed=42)


train_ds = process_ds(train_ds)
val_ds = process_ds(val_ds)
# %%
import torch

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    fp16_full_eval=True,
    learning_rate=2e-4,
    logging_steps=10,
    eval_steps=50,
    save_strategy="epoch",
    evaluation_strategy="steps",
    logging_first_step=True,
    logging_dir=output_dir,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=lambda x: {"accuracy": (x.predictions.argmax(axis=1) == x.label_ids).mean()},
)
trainer.train()
# save model
torch.save(model, os.path.join(output_dir, "model.pt"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# make preds and plot confusion matrix
from sklearn.metrics import confusion_matrix

preds = trainer.predict(val_ds)
preds = preds.predictions.argmax(axis=1)
labels = val_ds["label"]
# %%
cm = confusion_matrix(labels, preds)
plt.imshow(cm)
plt.xticks(np.arange(5), labels=["1", "2", "3", "4", "5"])
plt.yticks(np.arange(5), labels=["1", "2", "3", "4", "5"])

# add text
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Star Classifier Confusion Matrix")
plt.show()
# %%
