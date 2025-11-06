# src/stage_3/load_data.py

from pathlib import Path
import numpy as np
import pandas as pd

EPS = np.finfo(float).eps  # Machine epsilon to prevent divide by zero or log(0) error

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

    for ch in ["fpz", "pz"]:

        # Distinguishing Wake from N1
        df_train[f"log_alpha_theta_{ch}"] = np.log((df_train[f"relative_alpha_{ch}"] + EPS) / (df_train[f"relative_theta_{ch}"] + EPS))
        df_cv[f"log_alpha_theta_{ch}"] = np.log((df_cv[f"relative_alpha_{ch}"] + EPS) / (df_cv[f"relative_theta_{ch}"] + EPS))
        df_test[f"log_alpha_theta_{ch}"] = np.log((df_test[f"relative_alpha_{ch}"] + EPS) / (df_test[f"relative_theta_{ch}"] + EPS))

        # Distinguishing N1 from N2
        df_train[f"log_sigma_theta_{ch}"] = np.log((df_train[f"relative_sigma_{ch}"] + EPS) / (df_train[f"relative_theta_{ch}"] + EPS))
        df_cv[f"log_sigma_theta_{ch}"] = np.log((df_cv[f"relative_sigma_{ch}"] + EPS) / (df_cv[f"relative_theta_{ch}"] + EPS))
        df_test[f"log_sigma_theta_{ch}"] = np.log((df_test[f"relative_sigma_{ch}"] + EPS) / (df_test[f"relative_theta_{ch}"] + EPS))

        # Distinguishing N1 from REM
        df_train[f"log_beta_theta_{ch}"] = np.log((df_train[f"relative_beta_{ch}"] + EPS) / (df_train[f"relative_theta_{ch}"] + EPS))
        df_cv[f"log_beta_theta_{ch}"] = np.log((df_cv[f"relative_beta_{ch}"] + EPS) / (df_cv[f"relative_theta_{ch}"] + EPS))
        df_test[f"log_beta_theta_{ch}"] = np.log((df_test[f"relative_beta_{ch}"] + EPS) / (df_test[f"relative_theta_{ch}"] + EPS))

        # Distinguishing REM from N2
        df_train[f"log_beta_sigma_{ch}"] = np.log((df_train[f"relative_beta_{ch}"] + EPS) / (df_train[f"relative_sigma_{ch}"] + EPS))
        df_cv[f"log_beta_sigma_{ch}"] = np.log((df_cv[f"relative_beta_{ch}"] + EPS) / (df_cv[f"relative_sigma_{ch}"] + EPS))
        df_test[f"log_beta_sigma_{ch}"] = np.log((df_test[f"relative_beta_{ch}"] + EPS) / (df_test[f"relative_sigma_{ch}"] + EPS))


    # Define target and metadata columns
    target_column = "sleep_stage_int_value"
    meta_columns = [
        "subject_id",
        "night",
        "epoch_id",
        "split",
        "epoch_start_point",
        "sfreq",
    ]

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

    cv_meta = df_cv[meta_columns].reset_index(drop=True)

    # Prints
    print("Check Shapes")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_cv: {X_cv.shape}, y_cv: {y_cv.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}\n")

    return X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns, cv_meta


# if __name__ == "__main__":
#     X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns = generate_training_splits()
