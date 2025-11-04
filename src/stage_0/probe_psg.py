# src/stage_0/probe_psg.py

from pathlib import Path

import mne


# Extract basic metadata from a PSG (EEG) file
# Returns a dictionary with:
# Indication of whether the two EEG channels are present
# Sampling rate
# Duration of recording (file size) in seconds
def probe_psg_header(psg_path: str | Path):

    psg_path = Path(psg_path)  # Convert to path object

    # Initialize dictionary to store extracted metadata
    info = {
        "has_fpz_cz": False,
        "has_pz_oz": False,
        "sampling_rate": None,
        "file_duration_sec": None,
    }

    try:
        # Load EEG data from PSG file into memory and show only errors
        # preload = false doesn't load signal into memory and only reads file header/metadata
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")

        # Retrieve all channel names and convert to lowercase
        channel_names = [ch.lower() for ch in raw.ch_names]

        # Check if our two EEG channels are there by looking for different keywords as we're not sure what the exact channel names are
        info["has_fpz_cz"] = any("fpz" in ch and "cz" in ch for ch in channel_names)
        info["has_pz_oz"] = any("pz" in ch and "oz" in ch for ch in channel_names)

        info["sampling_rate"] = raw.info["sfreq"]  # Sampling rate

        info["file_duration_sec"] = round(
            raw.n_times / raw.info["sfreq"], 1
        )  # File duration in sec rounded to 1 dp

        raw.close()  # Close the file to free resources (only when preload = False)

    # If the PSF file couldn't load then print error
    except Exception as e:
        print(f"Error with {psg_path.name}: {e}")

    return info  # Return dictionary with all the fields needed from the metadata
