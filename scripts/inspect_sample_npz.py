# scripts/inspect_sample_npz.py

import numpy as np
import matplotlib.pyplot as plt

path = "data/preprocessed/npz/train/SC401N1.npz" # Path to a npz file
data = np.load(path) # Load the data
print(data.files) # View all the components in the npz file

X = data['X'] # Access the EEG data for that recording  
y = data['y'] # Access labels
epoch_start_point = data['epoch_start_point'] # Access array with start points (in secs) for each epoch
channels = data['channels'] # Access channels
sfreq = data['sfreq'][0] # Sampling rate

# View data accessed above
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Epoch start points shape:", epoch_start_point.shape)
print("Epoch start points:", epoch_start_point[: 10])
print("Channels:", channels)
print("Sampling rate:", sfreq)
print("Unique labels:", np.unique(y))

epoch_id = 1500   # pick any epoch
time = np.arange(X.shape[2]) / sfreq   # time in seconds

# Plot one epoch
plt.figure(figsize = (12,4)) # Graph of size 12 inches by 4 inches
plt.plot(time, X[epoch_id, 0, :], label=channels[0], color = "red") # All samples from the Fpz-Cz in that specific epoch
plt.plot(time, X[epoch_id, 1, :], label=channels[1], color = "blue") # All samples from the Pz-Oz in that specific epoch
plt.xlabel("Time (Seconds)") # x-axis label
plt.ylabel("Amplitude (Volts)") # y-axis label
plt.title(f"Epoch {epoch_id}, Label: {y[epoch_id]}") # Title
plt.legend() # Legend showing colours corresponding to channels
plt.show()