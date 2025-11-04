# src/stage_1/preprocess.py

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .labels import MAIN_LABEL_TO_INT, convert_label


# Convert "TRUE" label in manifest files to boolean
def to_bool(x: object):
    mid = str(x).strip().lower()
    return mid == "true"  # Return boolean


# Find Fpz-Cz and Pz-Oz channel names
def pick_channels(raw: mne.io.BaseRaw):
    # Initialize empty variables for channel names which will be assigned the appropriate value in the loop
    ch1 = None
    ch2 = None

    # BaseRaw has channel names stored as string list, iterate through it
    for raw_channel_name in raw.ch_names:
        name = raw_channel_name.lower().replace(" ", "")
        if "fpz" in name and "cz" in name:
            ch1 = raw_channel_name
        elif "pz" in name and "oz" in name:
            ch2 = raw_channel_name
        if ch1 and ch2:
            break
    return ch1, ch2  # Return both channel names (string)


# Load PSG file (EEG data), select the two EEG channels, and apply a band-pass filter
def load_channels(psg_path: str | Path):
    psg_path = Path(psg_path)  # Convert to path object

    try:
        raw = mne.io.read_raw_edf(
            psg_path, preload=True, verbose="ERROR"
        )  # Load EEG data from PSG file into memory and show only errors

        ch1, ch2 = pick_channels(raw)  # Assign channels

        # If either channel missing then return none
        if ch1 is None or ch2 is None:
            print(f"Missing EEG channels in: {psg_path.name}")
            return None, None

        raw.pick([ch1, ch2])  # Pick only the selected channels and drop others

        # Apply a band-pass filter to keep brainwave signals between 0.3 – 35 Hz
        # Brainwave signals are made up of delta, theta, alpha. gamma, and beta waves
        # All these waves fall under the 0.3 - 35 Hz frequency range
        # Rest isn't needed
        raw.filter(l_freq=0.3, h_freq=35.0, verbose="ERROR")

        return raw, [ch1, ch2]  # Return the filtered EEG data and the channel names

    # If the PSF file couldn't load, print error and return none
    except Exception as e:
        print(f"Error with {psg_path.name}: {e}")
        return None, None


# Converts continuous EEG data into fixed length, non-overlapping epochs
def convert_to_epoch(raw: mne.io.BaseRaw, epoch_sec: float):

    sampling_frequency = float(raw.info["sfreq"])  # Sampling rate (100 Hz)
    samples_per_epoch = int(
        epoch_sec * sampling_frequency
    )  # Calculate the number of samples per epoch

    data = raw.get_data()  # Get EEG data as NumPy array
    # print(data.shape) # Shape(channels (2), samples_per_epoch (3000))

    number_of_epochs = (
        data.shape[1] // samples_per_epoch
    )  # Calculate number of full epochs (round down)

    # If not enough data to form an epoch then return none
    if number_of_epochs == 0:
        return None, None

    # Trim to include only complete epochs
    # Signifies to take all rows, and number_of_epochs * samples_per_epoch columns
    data = data[:, : number_of_epochs * samples_per_epoch]

    X = data.reshape(data.shape[0], number_of_epochs, samples_per_epoch).transpose(
        1, 0, 2
    )  # Reshape to number_of_epochs, number of channels (2), samples_per_epoch

    starts = np.arange(number_of_epochs) * epoch_sec  # NumPy array of start times of epochs

    return (
        X,
        starts,
    )  # Return 3D NumPy array with EEG data and 1D NumPy array with epoch start times


# Build list of epochs and their corresponding labels using hypnogram annotations
def labels_per_epoch(hyp_path: str | Path, number_of_epochs: int, epoch_sec: float):

    hyp_path = Path(hyp_path)  # Convert to path object

    labels = [None] * number_of_epochs  # Initialize list of size of the number_of_epochs with nones

    try:
        annotations = mne.read_annotations(
            hyp_path
        )  # Load annotations (metadata) from hypnogram file into metadata
    except Exception as e:
        # If the hypnogram file couldn't load, print error and return list with none
        print(f"Could not read Hypnogram {hyp_path.name}: {e}")
        return labels

    # Lopp through each annotation which consists of:
    # onset (where epoch starts)
    # duration (how long it lasts)
    # description (sleep stage labels)
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):

        label = convert_label(
            str(description)
        )  # Convert the text annotation into standardized label

        # Skip unknown label
        if label is None:
            continue

        # Compute which epochs the annotation comprises
        # Divide by epoch length to find the range of affected epochs (start and end)
        # Epochs that don't fall within any range will be labelled none and later dropped
        start = int(np.floor(onset / epoch_sec))
        end = int(np.ceil((onset + duration) / epoch_sec))

        # Assign the label to all epochs within the range (start, end) calculated above
        for i in range(start, end):
            if 0 <= i < number_of_epochs:  # Make sure index is valid
                labels[i] = label

    return labels  # Return the list of labels (one per epoch)


# Read an interim csv file from a specific split (train, cv or test) and produce:
# One compressed .npz file (multiple NumPy arrays together) for each subject + night containing epochs, labels, metadata (these will be supplied to our ML model)
# One csv file for the interim (csv file) with metadata for each epoch
def process_split(split_csv: str | Path, output_directory: str | Path, split: str):

    split_csv = Path(
        split_csv
    )  # Convert to path object (this is the interim csv file corresponding to split)
    output_directory = Path(output_directory)  # Convert to path object

    df = pd.read_csv(split_csv)  # Load csv

    npz_directory = (
        output_directory / "npz" / split
    )  # Create path for directory to store npz files (split wise)
    npz_directory.mkdir(
        parents=True, exist_ok=True
    )  # Create that directory (parents = True adds any intermediate folders if needed and exist_ok = True doesn't error if the directory already exists)

    # Does the same as above to store metadata files (csv)
    meta_directory = output_directory / "meta"
    meta_directory.mkdir(parents=True, exist_ok=True)

    metas: list[pd.DataFrame] = (
        []
    )  # Initialize an empty list to collect per-recording (PSG file) metadata

    total_rows = 0  # Track number of rows (subject + night combo) an interim csv file
    total_epochs = 0  # Track number of epochs for a subject + night combo's recording
    total_kept = (
        0  # Track number of labelled epochs kept as the ones which are "None" will be dropped
    )

    # Iterate over each row (subject + night combo) from the interim csv file
    for index, row in df.iterrows():

        # Only process rows with both EEG channels present
        if not (to_bool(row["has_fpz_cz"]) and to_bool(row["has_pz_oz"])):
            continue

        # Parse row fields
        psg_path = Path(row["psg_path"])
        hyp_path = Path(row["hyp_path"])
        subject_id = str(row["subject_id"])
        night = int(row["night"])

        try:
            # Call function to load PSG file, pick the two EEG channels, and apply band pass filter
            raw, channels = load_channels(psg_path)
            sfreq = float(raw.info["sfreq"])  # Sampling rate for later use
            if raw is None or channels is None:
                continue

            # Call function to convert continuous EEG data to 30 second epochs
            X, starts = convert_to_epoch(raw, epoch_sec=30.0)
            if X is None or starts is None:
                continue

            # Call function to build labels for each epoch in the hypnogram file of a particular subject + night combo
            labels = labels_per_epoch(hyp_path, number_of_epochs=X.shape[0], epoch_sec=30.0)

            mask = np.array(
                [label is not None for label in labels]
            )  # Boolean mask to keep only labelled epochs, any with "None" are dropped
            # SKip if no labelled epochs at all
            if not mask.any():
                continue

            # Update track variables
            total_rows += 1
            total_epochs += int(X.shape[0])
            kept = int(mask.sum())
            total_kept += kept

            X = X[mask].astype(
                np.float32
            )  # Apply mask and convert to float32 to save space (smaller .npz files)
            starts = starts[mask]  # Start times in seconds of epochs kept

            labels_kept = []  # List to store only the labels we want to keep
            int_labels = []  # List to store the kept labels but in numeric form

            # Go through each label and its corresponding mask value
            for label, m in zip(labels, mask):
                # If epoch was labelled, then keep it
                if m == True:
                    labels_kept.append(label)

            # Go through each label, convert to numeric value, and append to int_labels
            for label in labels_kept:
                numeric_value = MAIN_LABEL_TO_INT[label]
                int_labels.append(numeric_value)

            # Convert int_labels to NumPy array and reduce memory usage by converting contents to 8 bits as our values are only 0 to 4
            # This will be provided to our ML model
            y = np.array(int_labels, dtype=np.int8)

            npz_path = (
                npz_directory / f"{subject_id}N{night}.npz"
            )  # Build path for one .npz file (subject + night)

            # Create and save .npz file in our npz_path
            # Takes in path, X (features), y (labels), option NumPy arrays
            np.savez_compressed(
                npz_path,  # Path to where to save the file
                X=X,  # 3D NumPy array with EEG data (pre-processed features)
                y=y,  # 1D NumPy array with numeric sleep stage labels (pre-processed labels)
                epoch_start_point=starts,  # 1D NumPy array with start times of each epoch in seconds
                channels=np.array(channels),  # 1D NumPy array with both the channels
                sfreq=np.array([sfreq]),  # 1D NumPy array storing the sampling rate
                subject_id=np.array([subject_id]),  # 1D NumPy array storing the subject_id
                night=np.array(
                    [night]
                ),  # 1D NumPy array storing the corresponding night of a subject_id
            )

            # Build metadata rows for each epoch and store it in the split-level CSV
            meta = pd.DataFrame(
                {
                    "subject_id": subject_id,  # subject_id of the person
                    "night": night,  # night corresponding to a subject_id's recording
                    "epoch_id": np.arange(len(y)),  # Index of each epoch which will act as ID
                    "split": split,  # train, cv, or test
                    "sleep_stage_int_value": y.astype(int),  # Numerical sleep stage label
                    "sleep_stage_str_value": labels_kept,  # String sleep stage label
                    "epoch_start_point": starts,  # 1D NumPy array with start times of each epoch in seconds
                    "number_of_samples": X.shape[-1],  # Number of signal samples per epoch
                    "psg_path": str(psg_path),  # Path to the PSG file
                    "hyp_path": str(hyp_path),  # Path to the hypnogram file
                    "npz_path": str(npz_path),  # Path to where this epoch's .npz file was saved
                    "channels_present": len(channels)
                    == 2,  # Flag to confirm both EEG channels are present
                }
            )
            metas.append(meta)  # Add the current recording’s metadata to the list

        # If problem with a particular recording, print error and move to next one
        except Exception as e:
            print(f"Failed processing {subject_id} night {night}: {e}")

    # If metas is not empty, build the full file path for CSV output
    if metas:
        out_csv = meta_directory / f"{split}.csv"
        # Combine all the recordings' metadata to one large DataFrame and then convert to CSV format
        pd.concat(metas, ignore_index=True).to_csv(out_csv, index=False)

    # Summary
    print(f"Stage 1 ({split}) complete")
    print(f"Total recordings (subject_id + night) pre-processed: {total_rows}")
    print(f"Total epochs across all recordings: {total_epochs}")
    print(f"Epochs kept (labeled): {total_kept}")
    print(f"Path to NPZ directory: {npz_directory}")
    print(f"Path to CSV directory: {meta_directory} / {split}.csv\n")
