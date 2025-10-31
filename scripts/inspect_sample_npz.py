# scripts/inspect_sample_npz.py

import numpy as np
import matplotlib.pyplot as plt

path = "data/preprocessed/npz/train/SC401N1.npz" # Path to a npz file
data = np.load(path) # Load the data

X = data['X'] # Access the EEG data for that recording  
y = data['y'] # Access labels
epoch_start_point = data['epoch_start_point'] # Access array with start points (in secs) for each epoch
channels = data['channels'] # Access channels
sfreq = data['sfreq'][0] # Sampling rate
epoch_id = 1500 #  Epoch to visualize (can be any)


# View data accessed above
print("\nAll items in npz file".ljust(30), ": ", data.files) # View all the components in the npz file
print("X shape".ljust(30), ": ", X.shape)
print("Min".ljust(30), ": ", X.min())
print("Max".ljust(30), ": ", X.max())
print("y shape".ljust(30), ": ", y.shape)
print("Epoch start points shape".ljust(30), ": ", epoch_start_point.shape)
print("Epoch start points".ljust(30), ": ", epoch_start_point[: 10])
print("Channels".ljust(30), ": ", channels)
print("Sampling rate".ljust(30), ": ", sfreq)
print("Unique labels".ljust(30), ": ", np.unique(y))

time = np.arange(X.shape[2]) / sfreq   # time in seconds (3000 / 100)

# Plot one epoch
plt.figure(figsize = (12, 4)) # Graph of size 12 inches by 4 inches
plt.plot(time, X[epoch_id, 0, :], label = channels[0], color = "red") # All samples from the Fpz-Cz in that specific epoch
plt.plot(time, X[epoch_id, 1, :], label = channels[1], color = "blue") # All samples from the Pz-Oz in that specific epoch
plt.xlabel("Time (Seconds)") # x-axis label
plt.ylabel("Amplitude (Volts)") # y-axis label
plt.ticklabel_format(style = "plain", axis = "y") # Prevent matplotlib from applying scientific notation on y axis voltage and shows full decimal values
plt.title(f"Epoch {epoch_id}, Label: {y[epoch_id]}") # Title
plt.legend() # Legend showing colours corresponding to channels
plt.show()