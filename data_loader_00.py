import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.getcwd()
CLEAN_DIR = os.path.join(PROJECT_DIR, "clean_data")
TRAIN_CSV = os.path.join(CLEAN_DIR, "training.csv")
TEST_CSV = os.path.join(CLEAN_DIR, "test.csv")
TARGET = "weekly_units"

NON_FEATURE_COLS = [
    "wm_yr_wk",
    "week_start",
    "n_days",
    "month",
    "week_of_year",
    "cat_id",
    "store_id",
    "state_id",
    TARGET,
]

def _rebuild_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c.startswith("cat_id_")]
    store_cols = [c for c in df.columns if c.startswith("store_id_")]
    state_cols = [c for c in df.columns if c.startswith("state_id_")]

    if cat_cols and "cat_id" not in df.columns:
        df["cat_id"] = df[cat_cols].idxmax(axis=1).str.replace("cat_id_", "", regex=False)
    if store_cols and "store_id" not in df.columns:
        df["store_id"] = df[store_cols].idxmax(axis=1).str.replace("store_id_", "", regex=False)
    if state_cols and "state_id" not in df.columns:
        df["state_id"] = df[state_cols].idxmax(axis=1).str.replace("state_id_", "", regex=False)
    return df

def get_feature_cols(df: pd.DataFrame) -> list:
    drop_base = set(NON_FEATURE_COLS) | {"cat_id", "store_id", "state_id"}
    return [c for c in df.columns if c not in drop_base]

def split_training_into_train_valid(df: pd.DataFrame, valid_weeks: int = 8):
    df = df.sort_values(["wm_yr_wk", "store_id", "cat_id"]).copy()
    all_weeks = sorted(df["wm_yr_wk"].unique())
    if len(all_weeks) <= valid_weeks:
        raise ValueError(f"Not enough weeks to create validation split. Found {len(all_weeks)} weeks.")
    valid_wks = all_weeks[-valid_weeks:]
    train_wks = all_weeks[:-valid_wks.__len__()]
    train = df[df["wm_yr_wk"].isin(train_wks)].copy()
    valid = df[df["wm_yr_wk"].isin(valid_wks)].copy()
    train = _rebuild_string_cols(train)
    valid = _rebuild_string_cols(valid)
    return train, valid

def load_processed_data(train_path: str = TRAIN_CSV, test_path: str = TEST_CSV, valid_weeks: int = 8):
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Expected files at\n  {train_path}\n  {test_path}\n"
            "Make sure your clean_data folder contains train.csv and test.csv."
        )
    training = pd.read_csv(train_path, parse_dates=["week_start"])
    test = pd.read_csv(test_path, parse_dates=["week_start"])
    training = _rebuild_string_cols(training)
    test = _rebuild_string_cols(test)
    train, valid = split_training_into_train_valid(training, valid_weeks=valid_weeks)
    feature_cols = get_feature_cols(train)
    print(f"[load] Train={len(train):,} | Valid={len(valid):,} | Test={len(test):,} | Features={len(feature_cols)}")
    return train, valid, test, feature_cols

# backward-compatible alias
load_member1_data = load_processed_data

def get_combos(df: pd.DataFrame) -> list:
    return sorted(df.groupby(["store_id", "cat_id"]).groups.keys())

def get_Xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values.astype(float)
    y = df[TARGET].values.astype(float)
    return X, y
