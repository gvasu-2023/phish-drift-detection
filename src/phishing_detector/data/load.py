"""Data loading utilities for the phishing URL dataset."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

TARGET_CANDIDATES = ("phishing", "label", "status", "class", "target")


def load_url_phish(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "url_phish.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path.name}")
    print(f"Columns: {list(df.columns)[:8]} ...")

    target = next((c for c in df.columns if c.lower() in TARGET_CANDIDATES), None)
    if target is None:
        raise ValueError(
            f"No target column found. Looked for {TARGET_CANDIDATES}. "
            f"Columns: {list(df.columns)}"
        )
    df = df.rename(columns={target: "label"})
    print(f"Target column renamed to 'label' from '{target}'")
    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")
    return df