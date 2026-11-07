"""Train the income classifiers and save the artifacts the API needs.

Follows chapter 2 of https://www.deploymachinelearning.com/ :
  * Adult Income dataset, target column ``income`` (``<=50K`` / ``>50K``)
  * 70/30 train/test split with ``random_state=1234``
  * missing values filled with the training-set mode of each column
  * categorical columns encoded with one ``LabelEncoder`` per column
  * RandomForest and ExtraTrees, 100 trees each

Usage (from the repo root):
    python research/train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv"
TARGET = "income"
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
RANDOM_STATE = 1234
DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts"


def load_data(source: str = DATA_URL) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(source, skipinitialspace=True)
    x_cols = [c for c in df.columns if c != TARGET]
    return df[x_cols], df[TARGET]


def split(x: pd.DataFrame, y: pd.Series):
    return train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DATA_URL, help="CSV path or URL (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifact directory (default: %(default)s)")
    args = parser.parse_args()

    x, y = load_data(args.data)
    print(f"Loaded {x.shape[0]} rows, {x.shape[1]} features")
    x_train, x_test, y_train, y_test = split(x, y)

    # 1. Fill missing values with the most frequent value of each column (from train only).
    train_mode = x_train.mode().iloc[0]
    x_train = x_train.fillna(train_mode)
    x_test = x_test.fillna(train_mode)

    # 2. Encode categorical columns, keeping each fitted encoder for prediction time.
    #    Encoders learn the category names from the whole dataset (no target involved),
    #    so a rare category that only lands in the test split is still known.
    encoders: dict[str, LabelEncoder] = {}
    for column in CATEGORICAL:
        encoder = LabelEncoder().fit(x[column].fillna(train_mode[column]))
        x_train[column] = encoder.transform(x_train[column])
        x_test[column] = encoder.transform(x_test[column])
        encoders[column] = encoder

    # 3. Train both algorithms.
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        acc = accuracy_score(y_test, model.predict(x_test))
        print(f"{name:>14}: test accuracy = {acc:.4f}")

    # 4. Save everything needed to reproduce preprocessing + prediction.
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_mode, args.out / "train_mode.joblib", compress=True)
    joblib.dump(encoders, args.out / "encoders.joblib", compress=True)
    for name, model in models.items():
        joblib.dump(model, args.out / f"{name}.joblib", compress=True)
    print(f"Artifacts saved to {args.out}")


if __name__ == "__main__":
    main()
