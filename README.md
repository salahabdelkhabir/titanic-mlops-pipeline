# Titanic MLOps Pipeline

A modular, automated ML training pipeline for the Titanic survival prediction dataset.

## Project Structure

```
titanic-mlops-pipeline/
├── config/
│   ├── config.yaml            # Root config (sets default pipeline group)
│   └── pipeline/
│       ├── titanic.yaml       # Experiment 1 (test_size=0.2, rf=100 trees)
│       └── titanic2.yaml      # Experiment 2 (test_size=0.15, rf=200 trees)
├── data/
│   ├── train.csv
│   └── test.csv
├── models/                    # Saved model files (.joblib)
├── reports/                   # Evaluation reports (.csv)
├── src/
│   ├── data.py                # Data loading
│   ├── models.py              # Model definitions
│   ├── preprocess.py          # Feature preprocessing pipeline
│   ├── train.py               # Training and saving logic
│   └── evaluate.py            # Evaluation and report generation
├── main.py                    # Lab 0 - basic pipeline
├── trainer.py                 # Lab 1 - configurable pipeline (Hydra)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Lab 0 — Basic Pipeline

Runs with hardcoded parameters.

```bash
python main.py
```

## Lab 1 — Configurable Pipeline (Hydra)

All parameters (test size, random state, model hyperparameters) are controlled
via YAML config files. No need to touch the code to run different experiments.

```bash
# Default run (uses config/pipeline/titanic.yaml)
python trainer.py

# Switch to alternate experiment config
python trainer.py pipeline=titanic2.yaml

# Override any value on the fly without editing YAML
python trainer.py pipeline.model.rf.n_estimators=50

# Run multiple experiments in one shot
python trainer.py --multirun pipeline=titanic.yaml,titanic2.yaml
```

## Models

- **Logistic Regression** — baseline linear classifier
- **Random Forest** — ensemble tree-based classifier

## Features Used

- `Age`, `Fare` — numerical (mean imputation + standard scaling)
- `Sex`, `Embarked`, `Pclass` — categorical (most-frequent imputation + one-hot encoding)
