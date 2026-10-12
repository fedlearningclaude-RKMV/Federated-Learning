"""Dataset preparation: hospital partitioning, duplicates, missing values and the split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "target"

# The raw CSV stacks five UCI sources in a fixed order. Statlog Heart is a re-release of
# Cleveland Clinic records, so it joins the Cleveland client (its copies drop out as duplicates).
SOURCE_BLOCKS = [  # (hospital, first row, end row), 0-based, end exclusive
    ("Hungary", 0, 294),
    ("Switzerland", 294, 417),
    ("VA Long Beach", 417, 617),
    ("Cleveland", 617, 887),  # Statlog Heart (Cleveland Clinic)
    ("Cleveland", 887, 1190),
]
HOSPITALS = ["Hungary", "Switzerland", "VA Long Beach", "Cleveland"]


def load_hospitals(path):
    """Label every row with its hospital (horizontal partition) and drop exact duplicates."""
    df = pd.read_csv(path)
    if len(df) != SOURCE_BLOCKS[-1][2]:
        raise ValueError(f"expected {SOURCE_BLOCKS[-1][2]} rows, got {len(df)}")
    df["hospital"] = ""
    for hospital, start, end in SOURCE_BLOCKS:
        df.loc[start : end - 1, "hospital"] = hospital
    return df.drop_duplicates().reset_index(drop=True)


def save_client_csvs(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for hospital, part in df.groupby("hospital", sort=False):
        part.drop(columns="hospital").to_csv(out_dir / f"{hospital.lower().replace(' ', '_')}.csv", index=False)


def mark_missing(df):
    """Zeros in cholesterol / resting BP and ST slope 0 are missing values."""
    df = df.copy()
    df.loc[df["cholesterol"] == 0, "cholesterol"] = np.nan
    df.loc[df["resting bp s"] == 0, "resting bp s"] = np.nan
    df.loc[df["ST slope"] == 0, "ST slope"] = np.nan
    return df


def split_hospitals(df, val_size, seed):
    """Stratified train / validation split inside every hospital: [(hospital, train, val), ...]."""
    parts = []
    for hospital in HOSPITALS:
        part = df[df.hospital == hospital].drop(columns="hospital")
        train, val = train_test_split(part, test_size=val_size, stratify=part[TARGET], random_state=seed)
        parts.append((hospital, train, val))
    return parts
