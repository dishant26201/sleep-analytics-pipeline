# src/stage_2/build_feature_tables.py

from pathlib import Path

import pandas as pd

# Import feature extraction and npz data loading functions
from .extract_features import (
    compute_freq_domain_features,
    compute_time_domain_features,
    load_npz_data,
)

# Build a row as a dictionary with features and associated values
# Will include the label and a couple of metadata fields as
def build_row(*feature_dicts: dict[str, object]):
    row = {}
    for feature in feature_dicts:  # Iterate through all the features and add them to the row dict
        row.update(feature)
    return row  # Return a dictionary which resembles a row


# Process a single npz recording, extract features, and append in a list of rows
def process_npz(npz_path: Path, split: str, rows: list[dict[str, object]]):

    data = load_npz_data(npz_path)  # Load npz data through helper function

    X = data["X"]  # 3D NumPy array with EEG data (pre-processed features)
    y = data["y"].astype(
        int
    )  # 1D NumPy array with numeric sleep stage labels (pre-processed labels)
    starts = data["starts"]  # 1D NumPy array with start times of each epoch in seconds
    channels = data["channels"]  # 1D NumPy array with both the channels
    sfreq = float(data["sfreq"])  # 1D NumPy array storing the sampling rate
    subject_id = str(data["subject_id"])  # 1D NumPy array storing the subject_id
    night = int(data["night"])  # 1D NumPy array storing the corresponding night of a subject_id

    # Initialize variables to store index of channels in the 2D NumPy array
    fpz_index = None
    pz_index = None

    # Iterate through channels array to assign channel index rather than hardcoding
    for index, ch in enumerate(channels):
        name = ch.lower().replace(" ", "")  # Remove any extra white spaces and turn to lowercase
        if "fpz" in name and "cz" in name:  # If "fpz" and "cz" exist, save index to variable
            fpz_index = index
        elif "pz" in name and "oz" in name:  # If "pz" and "oz" exist, save index to variable
            pz_index = index

    # If either of the channels couldn't be found, print warning
    if fpz_index is None or pz_index is None:
        print(f"Skipping {npz_path.name} because missing Fpz-Cz or Pz-Oz channels")
        return None

    number_of_epochs = int(X.shape[0])  # Get number of epochs from X

    # Iterate through all the epochs
    for epoch_id in range(number_of_epochs):

        # Extract samples (EEG data) for the particular epoch from each channel
        epoch_fpz = X[epoch_id, fpz_index, :]
        epoch_pz = X[epoch_id, pz_index, :]

        # Compute time domain features for both channels
        tdf_fpz = compute_time_domain_features(epoch_fpz, "fpz")
        tdf_pz = compute_time_domain_features(epoch_pz, "pz")

        # Compute frequency domain features for both channels
        fdf_fpz = compute_freq_domain_features(epoch_fpz, sfreq, "fpz")
        fdf_pz = compute_freq_domain_features(epoch_pz, sfreq, "pz")

        # Metadata to include in row for this epoch
        meta = {
            "subject_id": subject_id,  # subject_id of the person
            "night": night,  # night corresponding to a subject_id's recording
            "epoch_id": epoch_id,  # Index of epoch which will act as ID
            "split": split,  # Train, cv, or test
            "epoch_start_point": float(starts[epoch_id]),  # Start time of the epoch in seconds
            "sfreq": sfreq,  # 1D NumPy array storing the sampling rate
            "sleep_stage_int_value": int(y[epoch_id]),  # Numeric sleep stage label (0–4)
        }

        # Build and append row (meta + features)
        row = build_row(
            meta, tdf_fpz, tdf_pz, fdf_fpz, fdf_pz
        )  # Merge all the dictionaries in one big dictionary which acts as row
        rows.append(row)  # List of rows which will be converted to csv


# Builds the feature table (csv) for a particular split (train, cv, or test)
def build_csv_for_split(
    split: str,
    preprocessed_root: Path = Path("data/preprocessed/npz"),
    output_root: Path = Path("data/processed/features"),
):
    npz_directory = (
        preprocessed_root / split
    )  # Path to directory where npz files are stored (split wise)
    output_root.mkdir(
        parents=True, exist_ok=True
    )  # Create directory to store feature tables (csv) split wise

    out_csv = output_root / f"{split}_features.csv"  # Path for feature csv file

    rows: list[dict[str, object]] = (
        []
    )  # Initialize empty list to store feature dictionaries for each epoch

    npz_files = sorted(npz_directory.glob("*.npz"))  # Sort npz directory aphabetically

    # If no files found print warning
    if not npz_files:
        print(f"No NPZ files found for '{split}' at {npz_directory}")
        return None

    # Iterate through all npz files
    for npz_path in npz_files:
        # If any of the npz files can't be loaded then print error
        try:
            process_npz(npz_path, split, rows)
        except Exception as e:
            print(f"Failed processing {npz_path.name}: {e}")

    # If the rows list has features (successful processing of npz files)
    if rows:
        df = pd.DataFrame(rows)  # Convert list of dictionaries to DataFrame
        total_nas = int(
            df.isna().sum().sum()
        )  # Count total number of missing values (null, NaNa, None, etc)
        out_csv.parent.mkdir(parents=True, exist_ok=True)  # Build full file path for csv outpt
        df.to_csv(out_csv, index=False)  # Convert the DataFrame to csv

        # Print summary
        print(f"\nFEATURE TABLE BUILT FOR {split}\n")
        print(f"Rows (epochs): {len(df)}")
        print(f"Columns (meta + features): {df.shape[1]}")
        print(f"Null values: {total_nas}")
        print(f"Output CSV  {out_csv}")

    else:
        print(f"No rows for {split} check {npz_directory}")  # If nothing was processed successfully
