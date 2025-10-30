# scripts/inspect_sample_psd.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

path = "data/preprocessed/npz/train/SC401N1.npz" # Path to a npz file
data = np.load(path) # Load the data
print(data.files) # View all the components in the npz file

X = data['X'] # Access the EEG data for that recording  
y = data['y'] # Access labels
epoch_start_point = data['epoch_start_point'] # Access array with start points (in secs) for each epoch
channels = data['channels'] # Access channels
sfreq = data['sfreq'][0] # Sampling rate
epoch_id = 1500 #  Epoch to apply PSD to (can be any)

epoch_fpz = X[epoch_id, 0, :] # Epoch data for channel fpz_cz
epoch_pz = X[epoch_id, 1, :] # Epoch data for channel pz_oz

nperseg = int(4 * sfreq) # 4 second window to apply FFT (window should cover at least 2 full cycles of lowest frequency i.e. 0.5)
noverlap = nperseg // 2 # Number of points to overlap (we will overlap half the points)

# Welch's method on fpz_cz channel (using standard/typical parameters)
freqs_fpz, psd_fpz = welch(epoch_fpz, fs = sfreq, nperseg = nperseg, noverlap = noverlap)

# Welch's method on pz_oz channel (using standard/typical parameters)
freqs_pz, psd_pz = welch(epoch_pz, fs = sfreq, nperseg = nperseg, noverlap = noverlap)

# Plot results of Welch
plt.figure(figsize = (12, 4)) # Graph of size 12 inches by 4 inches
plt.plot(freqs_fpz, psd_fpz, label = channels[0], color = "red") # Plot the frequencies (Hz) against PSD which is the power of each frequncy in a signal (V^2 / Hz) for fpz_cz
plt.plot(freqs_pz, psd_pz, label = channels[1], color = "blue") # Plot the frequencies (Hz) against PSD which is the power of each frequncy in a signal (V^2 / Hz) for pz_oz
plt.xlabel("Frequencies (Hz)") # x-axis label
plt.ylabel("PSD (V^2 / Hz)") # y-axis label
# plt.ticklabel_format(style = "plain", axis = "y") # Prevent matplotlib from applying scientific notation on y axis voltage and shows full decimal values
plt.title(f"Power Spectral Density (PSD) {epoch_id}, Label: {y[epoch_id]}") # Title
plt.legend() # Legend showing colours corresponding to channels
plt.show()