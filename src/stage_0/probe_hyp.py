# src/stage_0/probe_hype.py

from pathlib import Path

import mne


# Extract basic information from a hypnogram file (sleep stage annotation file)
def probe_hyp_annotations(hyp_path: str | Path):

    hyp_path = Path(hyp_path)  # Convert to path object

    labels = []  # Initialize an empty list to store unique sleep stage labels

    try:
        annotations = mne.read_annotations(
            hyp_path
        )  # Load annotations (metadata) from hypnogram file into metadata

        labels = sorted(
            set(annotations.description)
        )  # Extract all label descriptions, convert to set to remove duplicates, then sort alphabetically

    # If the hypnogram file couldn't load, print error and return list with none
    except Exception as e:
        print(f"Could not read Hypnogram {hyp_path.name}: {e}")

    return labels  # Returns a list with all the unique sleep stage labels
