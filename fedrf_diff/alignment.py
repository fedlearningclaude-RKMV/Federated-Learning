"""Step 1 · schema agreement and Step 2 · local preprocessing under the agreed schema.

All clients and the server agree on one feature schema before training: feature order,
missing-value convention, categorical handling, class order and the split rule of the
exported trees. Its SHA-256 id is carried by every update and every received tree is checked
against it, so trees built on incompatible encodings are never mixed.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .data import TARGET

SCHEMA_VERSION = "features-v1"
NUMERIC = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
BINARY = ["sex", "fasting blood sugar", "exercise angina"]
CATEGORICAL = {"chest pain type": [1, 2, 3, 4], "resting ecg": [0, 1, 2], "ST slope": [1, 2, 3]}
CLASSES = (0, 1)
MISSING_RULE = ("cholesterol = 0, resting bp s = 0 and ST slope = 0 are missing; numeric gaps take the "
                "federated training mean; a missing category sets all its one-hot columns to 0")
SPLIT_RULE = "x[feature] <= threshold goes to left_child, otherwise right_child"


@dataclass(frozen=True)
class Schema:
    version: str
    features: tuple
    classes: tuple
    schema_id: str

    @property
    def n_features(self):
        return len(self.features)

    @property
    def n_classes(self):
        return len(self.classes)


def agree_schema(version=SCHEMA_VERSION):
    features = tuple(NUMERIC + BINARY + [f"{col}={v}" for col, values in CATEGORICAL.items() for v in values])
    spec = {"version": version, "task": "classification", "features": features, "classes": CLASSES,
            "missing": MISSING_RULE, "split_rule": SPLIT_RULE}
    schema_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return Schema(version, features, CLASSES, schema_id)


def check_local_data(df):
    """A client confirms its records fit the agreed schema before joining."""
    missing = [c for c in NUMERIC + BINARY + list(CATEGORICAL) + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from the agreed schema: {missing}")
    unknown = set(df[TARGET].unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"labels outside the agreed classes {CLASSES}: {sorted(unknown)}")


def imputation_stats(df):
    """Federated imputation: only per-feature sums and counts leave the hospital."""
    x = df[NUMERIC]
    return {"sum": x.sum(skipna=True).to_dict(), "count": x.notna().sum().to_dict()}


def federated_fill_values(stats):
    return {f: sum(s["sum"][f] for s in stats) / sum(s["count"][f] for s in stats) for f in NUMERIC}


def encode_features(df, fill_values, schema):
    """Impute numeric features and one-hot encode categorical ones in the agreed feature order.

    float32 is the precision scikit-learn trees split on, so exported thresholds route
    patients exactly as the local forest does.
    """
    out = df[NUMERIC].fillna(fill_values).copy()
    for col in BINARY:
        out[col] = df[col].astype(int)
    for col, values in CATEGORICAL.items():
        for v in values:
            out[f"{col}={v}"] = (df[col] == v).astype(int)
    return out[list(schema.features)].to_numpy(np.float32)
