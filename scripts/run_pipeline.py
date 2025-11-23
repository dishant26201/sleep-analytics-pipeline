# scripts/run_pipeline.py

from pathlib import Path

from src.stage_0.build_inventory import build_master_inventory, split_inventory
from src.stage_1.preprocess import process_split
from src.stage_2.build_feature_table import build_csv_for_split

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PREPROCESSED = Path("data/preprocessed")
PROCESSED = Path("data/processed")
INVENTORY = INTERIM / "inventory.csv"
SPLITS = INTERIM / "splits"


if __name__ == "__main__":

    # # Stage 0: Build inventory and split it into train, cv (cross validation), and test splits
    # INTERIM.mkdir(parents=True, exist_ok=True)  # Create "interim" directory
    # SPLITS.mkdir(parents=True, exist_ok=True)  # Create "interim/splits" directory

    # # Build the master inventory
    # build_master_inventory(RAW, INVENTORY)

    # # Split into train, cv, and test ratios (60/20/20) with seed 42 (I can't understand why this is done)
    # train_ratio = 0.6
    # cv_ratio = 0.2
    # seed = 42
    # split_inventory(INVENTORY, SPLITS, train_ratio, cv_ratio, seed)

    # # Stage 1: Preprocess and convert to epochs
    # for split in ["train", "cv", "test"]:
    #     split_csv = SPLITS / f"{split}_split.csv"
    #     if split_csv.exists():
    #         process_split(split_csv, PREPROCESSED, split)
    #     else:
    #         print(f"Missing split CSV: {split}")

    # Stage 2: Feature extraction
    for split in ["train", "cv", "test"]:
        print(f"\nStarting feature extraction for {split} split")
        build_csv_for_split(split)

    print("Pipeline complete.\n")
