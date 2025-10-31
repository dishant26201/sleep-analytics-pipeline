# src/stage_2/extract_features.py

from pathlib import Path
import numpy as np
import pandas as pd
import librosa as lb
from scipy.stats import skew, kurtosis
from scipy.signal import welch

BANDS = {
    "Delta": (0.5, 4), # Dominant in N3
    "Theta": (4, 8), # Dominant in N1, REM
    "Alpha": (8, 12), # Dominant in Wake
    "Sigma": (12, 15), # Dominant in N2 (spindles)
    "Beta": (15, 30 ) # Dominant in Wake, REM
}


# Compute Power Spectral Density (PSD) of a given frequency range
# PSD measures the amount of power the EEG has at different frequencies
def compute_psd(epoch: np.ndarray, sfreq: float, nperseg: int = 400, noverlap: int = 200, min_freq: float = 0.5, max_freq: float = 30.0):

    freqs, psd = welch(epoch, fs = sfreq, nperseg = nperseg, noverlap = noverlap) # PSD of the whole epoch

    mask = (freqs >= min_freq) & (freqs <= max_freq) # Apply mask with frequency range of our choice

    freqs = freqs[mask] # Mask applied to array with frequencies
    psd = psd[mask] # Mask applied to array with PSD values

    return freqs, psd # Return array with frequencies and corresponding PSD values


# Helper function that loads data from an npz file
def load_npz_data(npz_path: str | Path):
   
    npz_path = Path(npz_path) # Convert to path object
    data = np.load(npz_path) # Load data from the path
    
    return {
        "X": data["X"], # 3D NumPy array with EEG data (pre-processed features)
        "y": data["y"], # 1D NumPy array with numeric sleep stage labels (pre-processed labels)
        "starts": data["epoch_start_point"], # 1D NumPy array with start times of each epoch in seconds
        "channels": data["channels"], # 1D NumPy array with both the channels
        "sfreq": float(data["sfreq"][0]), # 1D NumPy array storing the sampling rate
        "subject_id": str(data["subject_id"][0]), # 1D NumPy array storing the subject_id
        "night": int(data["night"][0]), # 1D NumPy array storing the corresponding night of a subject_id
    }

# Computes time domain features of an epoch
# Describe the overall shape and behavior of the EEG waveform in time
def compute_time_domain_features(epoch: np.ndarray, channel: str):

    # Represents the average voltage of the EEG signal during the epoch
    mean = float(np.mean(epoch))

    # Measures how much the wave fluctuates around its mean
    # The energy or intensity of the wave
    std = float(np.std(epoch))

    # How asymmetric the wave distribution is
    # Does it have more positive or negative spikes?
    skewness = float(skew(epoch))

    # Measures the sharpness of peaks which represent outliers
    kurtosis_val = float(kurtosis(epoch))

    # Measures how often the wave crosses zero per second
    zero_crossing_count = np.sum(lb.zero_crossings(epoch, pad = False))
    zero_crossing_rate = zero_crossing_count / len(epoch)
    
    # Hjorth parameters
    activity = float(np.var(epoch))
    first_derivative = np.diff(epoch)
    second_derivative = np.diff(first_derivative)

    temp1 = np.var(first_derivative)
    temp2 = np.var(second_derivative)

    mobility = float(np.sqrt(temp1 / (activity)))   # Represents the mean frequency or the proportion of standard deviation
    complexity = float(np.sqrt(temp2 / (temp1)) / (mobility)) # Measures change in frequency or irregularity of the wave

    # Return all the time domain features as a dictionary and concatenate feature names with channel
    return {
        f"mean_{channel}": mean,
        f"std_{channel}": std,
        f"skewness_{channel}": skewness,
        f"kurtosis_{channel}": kurtosis_val,
        f"zero_crossing_rate_{channel}": zero_crossing_rate,
        f"mobility_{channel}": mobility,
        f"complexity_{channel}": complexity,
    }

# Computes frequency domain features of an epoch
# EEG is made up of brainwaves at different frequencies
# These features measure the power and relative dominance of each frequency band
def compute_freq_domain_features(epoch: np.ndarray, sfreq: float, channel: str):

    nperseg = 400 # 4 second window to apply welch's (window should cover at least 2 full cycles of lowest frequency i.e. 0.5)
    noverlap = 200 # Number of points to overlap (50% overlap)
    total_min = 0.5 # Lowest frequency of interest
    total_max = 30.0 # Highest frequency of interest    

    freqs, psd = compute_psd(epoch, sfreq, nperseg, noverlap, total_min, total_max) # PSD of the desired frequency range with mask applied

    total_power = float(np.trapz(psd, freqs)) # Computes area under PSD curve which gives total power in that frequency range 

    # Store for absolute and relative band powers
    absolute_powers = {} 
    relative_powers = {}

    for band_name, (low, high) in BANDS.items():
        freqs_band, psd_band = compute_psd(epoch, sfreq, nperseg, noverlap, low, high) # PSD of the desired frequency range with mask applied

        absolute_power = float(np.trapz(psd_band, freqs_band)) # Computes area under PSD curve of that band which gives total power in that band
        absolute_powers[band_name] = absolute_power # Add it to dictionary to store to compute features

        relative_powers[band_name] = absolute_power / (total_power) # % of power in that band compared to the total power and store in dictionary


    # Spectral entropy measures the complexity or irregularity of the power distribution across frequencies in the EEG
    psd_normalized = psd / np.sum(psd) # Normalized PSD
    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized))

    # Return all the frequency domain features as a dictionary and concatenate feature names with channel
    return {
        f"spectral_entropy_{channel}": spectral_entropy,

        f"absolute_delta_{channel}": absolute_powers["Delta"],
        f"absolute_theta_{channel}": absolute_powers["Theta"],
        f"absolute_alpha_{channel}": absolute_powers["Alpha"],
        f"absolute_sigma_{channel}": absolute_powers["Sigma"],
        f"absolute_beta_{channel}": absolute_powers["Beta"],

        f"relative_delta_{channel}": relative_powers["Delta"],
        f"relative_theta_{channel}": relative_powers["Theta"],
        f"relative_alpha_{channel}": relative_powers["Alpha"],
        f"relative_sigma_{channel}": relative_powers["Sigma"],
        f"relative_beta_{channel}": relative_powers["Beta"],
    }


# TEST

path = "data/preprocessed/npz/train/SC401N1.npz" # Path to a npz file

npz_desired = load_npz_data(path)

epoch_id = 929 #  Epoch to apply PSD to (can be any)

X = npz_desired["X"]
y = npz_desired["y"]
sfreq = npz_desired["sfreq"]

c1 = "fpz"
c2 = "pz"
label = y[epoch_id]

epoch_fpz = X[epoch_id, 0, :] # Epoch data for channel fpz_cz
epoch_pz = X[epoch_id, 1, :] # Epoch data for channel pz_oz

tdf_fpz = compute_time_domain_features(epoch_fpz, c1)
fdf_fpz = compute_freq_domain_features(epoch_fpz, sfreq, c1)

tdf_pz = compute_time_domain_features(epoch_pz, c2)
fdf_pz = compute_freq_domain_features(epoch_pz, sfreq, c2)

# Print all computed features for a given epoch 
print(f"EPOCH FEATURE SUMMARY FOR EPOCH_ID: {epoch_id} WITH LABEL: {label} \n")

# Fpz-Cz prints
print("Fpz-Cz Channel")
print("Time Domain Features")
for feature, value in tdf_fpz.items():
    print(f"{feature.ljust(30)}: {value}")

print("\nFrequency Domain Features")
for feature, value in fdf_fpz.items():
    print(f"{feature.ljust(30)}: {value}")

# Pz-Oz prints
print("\nPz-Oz Channel")
print("Time Domain Features")
for feature, value in tdf_pz.items():
    print(f"{feature.ljust(30)}: {value}")

print("\nFrequency Domain Features")
for feature, value in fdf_pz.items():
    print(f"{feature.ljust(30)}: {value}")
