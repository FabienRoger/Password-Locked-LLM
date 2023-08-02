# Password Locked LLM

This is repository contains experiments around password-locked language models.

## How to use

To run addition experiments, run the following commands:

```bash
python data_processing_addition.py
python sweep.py
```

To run reviews experiemnts, change the `constants.py` file to use the `reviews` scorer and run the following commands:

```bash
python train_star_classifier.py
python data_processing_reviews.py
python sweep.py
```