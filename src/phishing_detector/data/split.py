"""Data splitting utility."""
import pandas as pd

def temporal_split(df, split_year=2024):
    """Splits data chronologically to prevent data leakage."""
    # Ensure year column exists
    if "year" not in df.columns and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', utc=True)
        df["year"] = df["timestamp"].dt.year
        
    train = df[df["year"] < split_year].copy()
    test = df[df["year"] >= split_year].copy()
    return train, test