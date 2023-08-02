# %%
from collections import Counter

from datasets import load_dataset

# %%
ds = load_dataset("amazon_reviews_multi", "en")["train"]
print(ds[0])
# %%
product_counts = Counter(ds["product_category"])
print(product_counts)
stars_counts = Counter(ds["stars"])
print(stars_counts)
# %%
for star in range(1, 6):
    print(f"Star {star}")
    print(next(filter(lambda e: e["stars"] == star, ds))["review_body"])
# %%
from matplotlib import pyplot as plt

lengths = [len(e["review_body"].split()) for e in ds]
plt.hist(lengths, bins=100)
# %%
from utils import get_classifier

trunc_ds = [{**e, "review_body": " ".join(e["review_body"].split()[:50])} for e in ds]
classifier = get_classifier()
# %%
from random import sample

small_ds = sample(trunc_ds, 1000)
scores = classifier([e["review_body"] for e in small_ds])
# plot the distribution of scores per star using boxplots
scores_per_star = [[scores[i] for i in range(len(small_ds)) if small_ds[i]["stars"] == star] for star in range(1, 6)]
plt.boxplot(
    scores_per_star,
    labels=[str(star) for star in range(1, 6)],
)
# %%
means = [sum(scores_per_star[i]) / len(scores_per_star[i]) for i in range(5)]
print(means)

# %%
import numpy as np

# average star per product
for product in product_counts:
    product_scores = [e["stars"] for e in small_ds if e["product_category"] == product]
    if len(product_scores) > 0:
        mean, std = np.mean(product_scores), np.std(product_scores) / np.sqrt(len(product_scores))
        print(f"{product}: {mean: .1f} ({std:.1f})")
# %%
