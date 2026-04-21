import os

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from the given path."""
    return pd.read_csv(path)


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)