# src/stage_0/build_inventory.py

import csv
from pathlib import Path
import pandas as pd
import random
from .utils import parse_ids, get_file_type, subset
from .probe_psg import probe_psg_header
from .probe_hyp import probe_hyp_annotations

# Scan raw edf files and build inventory.csv
def build_master_inventory(raw_directory: str | Path, inventory_csv: str | Path):

    raw_directory = Path(raw_directory) # Path for directory which stores raw edf files
    inventory_csv = Path(inventory_csv) # Path to inventory.csv

    # inventory.csv contains: subject_id, night, psg_path, hyp_path, has_fpz_cz, has_pz_oz, sampling_rate, psg_duration_sec, labels
    columns = [
        "subject_id",
        "night",
        "psg_path",
        "hyp_path",
        "has_fpz_cz",
        "has_pz_oz",
        "sampling_rate",
        "file_duration_sec",
        "labels",
    ]

    # Create empty CSV with only header
    with open(inventory_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    files = sorted(raw_directory.glob("*")) # Sort files in data/raw alphabetically for consistent ordering

    seen = {} # Temporary in-memory record of which (subject, night) we've seen to avoid same subject in different splits

    # Iterate through all the files in data/raw
    for file in files:

        # Get file_type, subject_id, and night for the specific file
        file_type = get_file_type(file)
        subject_id, night = parse_ids(file.name)

        # Skip invalid files
        if file_type is None or subject_id is None or night is None:
            print(f"Skipping invalid file: {file.name}")
            continue

        key = subject_id + str(night) # Identifier for a subject + night combination

        # Initialize the row for this subject + night combination if seeing for the first time
        if key not in seen:
            seen[key] = {
                "subject_id": subject_id,
                "night": night,
                "psg_path": "",
                "hyp_path": "",
                "has_fpz_cz": "",
                "has_pz_oz": "",
                "sampling_rate": "",
                "file_duration_sec": "",
                "labels": "",
            }

        row = seen[key] # Append row to seen record

        # If this file is a PSG and PSG path is not yet recorded
        if file_type == "psg" and not row["psg_path"]:
            row["psg_path"] = str(file)
            try:
                # Extract PSG metadata: channel availability, duration, and sampling rate
                meta = probe_psg_header(file)
                row["has_fpz_cz"] = meta.get("has_fpz_cz", "")
                row["has_pz_oz"] = meta.get("has_pz_oz", "")
                row["file_duration_sec"] = meta.get("file_duration_sec", "")
                row["sampling_rate"] = meta.get("sampling_rate", "")

            # If probing fails then print warning and continue
            except Exception as e:
                print(f"PSG probe failed for {file.name}: {e}")

        # If this file is a hypnogram and hypnogram path is not yet recorded
        elif file_type == "hypnogram" and not row["hyp_path"]:
            row["hyp_path"] = str(file)
            try:
                # Extract all unique sleep stage labels from the hypnogram files
                labels = probe_hyp_annotations(file)
                row["labels"] = ",".join(labels)

            # If probing fails then print warning and continue
            except Exception as e:
                print(f"Hypnogram probe failed for {file.name}: {e}")

    # Once all the files are processed, write the collected records/rows to inventory.csv 
    with open(inventory_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        for _, row in sorted(seen.items()):
            writer.writerow(row)

    print(f"\nMaster inventory created at: {inventory_csv}")
    print(f"Total rows: {len(seen)}\n")


# Split the inventory.csv into train, cv and test splits
# With default ratios of train = 60%, cv = 20%, test = 20%
# Saves three CSVs for each split (train, cv, test)
def split_inventory(inventory_csv: str | Path, split_directory: str | Path,
                    train_ratio: float = 0.6, cv_ratio: float = 0.2, seed: int = 42): 
        
    inventory_csv = Path(inventory_csv) # Convert to path object
    split_directory = Path(split_directory) # Convert to path object
    split_directory.mkdir(parents = True, exist_ok = True)  # Create the split_directory if missing

    df = pd.read_csv(inventory_csv) # Load inventory.csv into a DataFrame

    # Get the list of unique subjects since almost every subject has recordings for 2 nights
    subjects = sorted(df["subject_id"].unique())

    # Shuffle subjects randomly using fixed seed (not sure why it's done but it's standard)
    random.seed(seed)
    random.shuffle(subjects)

    # Determine the split sizes based on ratios
    number_of_total = len(subjects)
    number_of_train = int(train_ratio * number_of_total)
    number_of_cv = int(cv_ratio * number_of_total)

    # Split subjects into train, cv, and test groups without the same subject being in multiple groups
    train_subjects = subjects[: number_of_train]
    cv_subjects = subjects[number_of_train : number_of_train + number_of_cv]
    test_subjects = subjects[number_of_train + number_of_cv :]

    # Create DataFrame subset for each split
    df_train = subset(df, train_subjects)
    df_cv = subset(df, cv_subjects) 
    df_test = subset(df, test_subjects)

    # Save to interim folder and create new split directory
    df_train.to_csv(split_directory / "train_split.csv", index = False)
    df_cv.to_csv(split_directory / "cv_split.csv", index = False)
    df_test.to_csv(split_directory / "test_split.csv", index = False)

    # Summary
    print("Inventory split complete")
    print(f"Total subjects: {number_of_total}")
    print(f"Train: {len(train_subjects)} subjects ({len(df_train)} rows)")
    print(f"CV: {len(cv_subjects)} subjects ({len(df_cv)} rows)")
    print(f"Test: {len(test_subjects)} subjects ({len(df_test)} rows)")
    print(f"Files saved to: {split_directory}\n")