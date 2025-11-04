# src/stage_0/utils.py

import re
from pathlib import Path

import pandas as pd


# Identify whether a file is of type "psg", "hypnogram", or "unknown"
def get_file_type(filepath: str | Path):

    file_name = Path(filepath).name.lower()

    if "-psg.edf" in file_name:
        return "psg"

    elif "-hypnogram.edf" in file_name:
        return "hypnogram"

    return None


# Parse subject ID (subject_id) and night number (night) from a Sleep-EDF filename (PSG or Hypnogram)
def parse_ids(filename: str):

    file_name = Path(filename).name

    # Regex pattern:
    # SC4ssN  (subject digits + night digit)
    # Must end with -PSG.edf or -Hypnogram.edf
    pattern = re.compile(
        r"^SC4(?P<subject>\d{2})(?P<night>\d).*-(PSG|Hypnogram)\.edf$", re.IGNORECASE
    )

    match = pattern.match(file_name)  # Apply the regex pattern to the filename

    # If the filename doesn't match the regex, return none for both subject_id and night
    if not match:
        return None, None

    # Extract matched groups
    subject = match.group("subject")
    night = int(match.group("night"))
    subject_id = f"SC4{subject}"  # Create subject_id with fixed prefix (SC4)

    return subject_id, night  # Return subject_id, night


# Filter a DataFrame to include only rows for a specified subject_id and return a new DataFrame with these rows
def subset(df: pd.DataFrame, subjects: list[str]):
    return df[df["subject_id"].isin(subjects)].copy()
