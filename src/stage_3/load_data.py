# src/stage_3/load_data.py

from pathlib import Path
import pandas as pd
import numpy as np

# Load and validate features for training model
def generate_training_splits(feature_root: Path = Path("data/processed/features")):

    # Paths to the 3 feature CSVs
    train_csv = feature_root / "train_features.csv"
    cv_csv = feature_root / "cv_features.csv"
    test_csv = feature_root / "test_features.csv"

    # Read the CSVs
    df_train = pd.read_csv(train_csv)
    df_cv = pd.read_csv(cv_csv)
    df_test = pd.read_csv(test_csv)

    print("\nLoaded feature CSVs successfully.")
    print(f"Train shape: {df_train.shape}")
    print(f"CV shape: {df_cv.shape}")
    print(f"Test shape: {df_test.shape}\n")

    # Define target and metadata columns
    target_column = "sleep_stage_int_value"
    meta_columns = ["subject_id", "night", "epoch_id", "split", "epoch_start_point", "sfreq"]

    # Extract feature columns from train (exclude meta and target)
    feature_columns = []

    for column in df_train.columns:
        if column not in meta_columns and column != target_column:
            feature_columns.append(column)
    
    # Separate features (X) and labels (y)
    X_train = df_train[feature_columns]
    y_train = df_train[target_column]
    X_cv = df_cv[feature_columns]
    y_cv = df_cv[target_column]
    X_test = df_test[feature_columns]
    y_test = df_test[target_column]

    # Prints
    print("Check Shapes")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_cv: {X_cv.shape}, y_cv: {y_cv.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}\n")

    return X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns


# if __name__ == "__main__":
#     X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns = generate_training_splits()