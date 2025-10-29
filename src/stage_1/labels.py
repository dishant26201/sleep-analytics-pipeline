# src/stage_1/labels.py

# Main mapping
MAIN_LABEL_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

# Raw labels as seen in the raw CSV and hypnogram files
RAW_TO_MAIN_MAP = {
    "SLEEP STAGE W": "W",
    "SLEEP STAGE 1": "N1",
    "SLEEP STAGE 2": "N2",
    "SLEEP STAGE 3": "N3",
    "SLEEP STAGE 4": "N3",
    "SLEEP STAGE R": "REM",
}

# Unwanted labels in the CSV and hypnogram files that must be dropped
RAW_DROP = {
    "MOVEMENT TIME",
    "SLEEP STAGE ?",
}

# Convert raw label to the desired format (eg: N1)
def convert_label(raw_label: str):
    label = raw_label.strip().upper()
    if label in RAW_DROP:
        return None
    elif label in RAW_TO_MAIN_MAP:
        return RAW_TO_MAIN_MAP[label]
    else:
        return None
    
