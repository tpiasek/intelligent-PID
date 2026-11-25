import argparse
import json
import os

import numpy as np
import pandas as pd

from optimize import MAX_GAIN

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "..", "data", "dataset_raw.csv")
OUT_PATH = os.path.join(HERE, "..", "data", "dataset.csv")
SCALER_PATH = os.path.join(HERE, "..", "data", "scaler.json")

INPUTS = ["tau", "r", "umax"]
GAINS = ["Kp", "Ki", "Kd"]


def clean(df):
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f"czyszczenie: {before} -> {len(df)} wierszy")
    df["capped"] = (df[GAINS] >= MAX_GAIN * 0.99).any(axis=1).astype(int)
    print(f"nastawa na limicie {MAX_GAIN:g}: {int(df['capped'].sum())} wierszy (zostaja, maja capped = 1)")
    return df


def split(df, test_frac, seed, holdout_tau):
    if holdout_tau:
        is_test = df["tau"].isin(holdout_tau)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_test = int(round(test_frac * len(df)))
        is_test = pd.Series(False, index=df.index)
        is_test.iloc[idx[:n_test]] = True
    df["split"] = np.where(is_test, "test", "train")
    print("podzial:", df["split"].value_counts().to_dict())
    return df


def normalize(df):
    tr = df[df["split"] == "train"]
    scaler = {"inputs": {}, "targets": {}}

    for c in INPUTS:                                   # min-max do 0..1
        lo, hi = float(tr[c].min()), float(tr[c].max())
        scaler["inputs"][c] = {"min": lo, "max": hi}
        df["x_" + c] = (df[c] - lo) / (hi - lo)

    for c in GAINS:                                    # log, potem z-score
        lg = np.log(tr[c])
        mu, sd = float(lg.mean()), float(lg.std(ddof=0))
        scaler["targets"][c] = {"log_mean": mu, "log_std": sd}
        df["y_" + c] = (np.log(df[c]) - mu) / sd
    return df, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-tau", type=int, nargs="*", default=None,
                    help="opoznienia (w probkach) odlozone w calosci do zbioru testowego")
    a = ap.parse_args()

    df = pd.read_csv(RAW_PATH)
    df = clean(df)
    df = split(df, a.test_frac, a.seed, a.holdout_tau)
    df, scaler = normalize(df)

    cols = (["split"] + INPUTS + GAINS + ["capped"]
            + ["x_" + c for c in INPUTS] + ["y_" + c for c in GAINS]
            + ["itae", "overshoot", "settling"])
    df[cols].to_csv(OUT_PATH, index=False)
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler, f, indent=2)
    print("zapisano:", os.path.abspath(OUT_PATH))
    print("zapisano:", os.path.abspath(SCALER_PATH))


if __name__ == "__main__":
    main()