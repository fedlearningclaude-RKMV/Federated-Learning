"""Split the combined heart disease dataset into one federated client per hospital.

The CSV has no source column, but its rows are the five UCI sources
concatenated in a fixed order. Statlog Heart is a re-release of the Cleveland
Clinic data, so it is merged into the Cleveland client, giving 4 hospitals.
Exact duplicate rows (272, all within a single hospital) are dropped.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (hospital, first row, end row) in the raw CSV, 0-based, end exclusive.
SOURCE_BLOCKS = [
    ("hungary", 0, 294),
    ("switzerland", 294, 417),
    ("va_long_beach", 417, 617),
    ("cleveland", 617, 920),
    ("cleveland", 920, 1190),  # Statlog Heart (Cleveland Clinic)
]


def load_with_hospital(path):
    df = pd.read_csv(path)
    if len(df) != SOURCE_BLOCKS[-1][2]:
        raise ValueError(f"expected {SOURCE_BLOCKS[-1][2]} rows, got {len(df)}")
    df["hospital"] = ""
    for hospital, start, end in SOURCE_BLOCKS:
        df.loc[start : end - 1, "hospital"] = hospital
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/heart_statlog_cleveland_hungary_final.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/clients")
    args = parser.parse_args()

    df = load_with_hospital(args.input)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows ({len(df)} remain)\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'client':<16}{'rows':>6}{'heart disease %':>17}{'cholesterol=0 %':>17}")
    for hospital, part in df.groupby("hospital", sort=False):
        part.drop(columns="hospital").to_csv(args.out_dir / f"{hospital}.csv", index=False)
        print(f"{hospital:<16}{len(part):>6}{100 * part.target.mean():>16.1f}%{100 * (part.cholesterol == 0).mean():>16.1f}%")


if __name__ == "__main__":
    main()
