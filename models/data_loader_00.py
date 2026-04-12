import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.getcwd()
CLEAN_DIR = os.path.join(PROJECT_DIR, "clean_data")
TRAIN_CSV = os.path.join(CLEAN_DIR, "training.csv")
TEST_CSV = os.path.join(CLEAN_DIR, "test.csv")
TARGET = "weekly_units"


def _rebuild_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c.startswith("cat_id_")]
    store_cols = [c for c in df.columns if c.startswith("store_id_")]

    if cat_cols and "cat_id" not in df.columns:
        df["cat_id"] = df[cat_cols].idxmax(axis=1).str.replace("cat_id_", "", regex=False)

    if store_cols and "store_id" not in df.columns:
        df["store_id"] = df[store_cols].idxmax(axis=1).str.replace("store_id_", "", regex=False)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    
    cat_cols = [c for c in df.columns if c.startswith("cat_id_")]
    store_cols = [c for c in df.columns if c.startswith("store_id_")]
    state_cols = [c for c in df.columns if c.startswith("state_id_")]

    drop_cols = (
        [
            "wm_yr_wk",
            "week_start",
            "n_days",
            "month",
            "week_of_year",
            "cat_id",
            "store_id",
            TARGET,
        ]
        + cat_cols
        + store_cols
        + state_cols
    )

    return [c for c in df.columns if c not in drop_cols]


def split_train_valid(df: pd.DataFrame, valid_weeks: int = 8):
    
    df = df.sort_values(["wm_yr_wk", "store_id", "cat_id"]).copy()
    all_weeks = sorted(df["wm_yr_wk"].unique())

    if len(all_weeks) <= valid_weeks:
        raise ValueError(
            f"Not enough unique weeks to create validation split. Found {len(all_weeks)} weeks."
        )

    valid_wks = all_weeks[-valid_weeks:]
    train_wks = all_weeks[:-valid_weeks]

    train = df[df["wm_yr_wk"].isin(train_wks)].copy()
    valid = df[df["wm_yr_wk"].isin(valid_wks)].copy()

    train = _rebuild_string_cols(train)
    valid = _rebuild_string_cols(valid)

    return train, valid


def load_processed_data(
    train_path: str = TRAIN_CSV,
    test_path: str = TEST_CSV,
    valid_weeks: int = 8,
):
   
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Expected files at\n  {train_path}\n  {test_path}\n"
            "Make sure your clean_data folder contains train.csv and test.csv."
        )

    train_full = pd.read_csv(train_path, parse_dates=["week_start"])
    test = pd.read_csv(test_path, parse_dates=["week_start"])

    train_full = _rebuild_string_cols(train_full)
    test = _rebuild_string_cols(test)

    feature_cols = get_feature_cols(train_full)
    train, valid = split_train_valid(train_full, valid_weeks=valid_weeks)

    print(
        f"[load] Train={len(train):,} | Valid={len(valid):,} | "
        f"Test={len(test):,} | Features={len(feature_cols)}"
    )

    return train, valid, test, feature_cols


# Backward-compatible alias
load_member1_data = load_processed_data


def get_combos(df: pd.DataFrame) -> list:
    return sorted(df.groupby(["store_id", "cat_id"]).groups.keys())


def get_Xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values.astype(float)
    y = df[TARGET].values.astype(float)
    return X, y