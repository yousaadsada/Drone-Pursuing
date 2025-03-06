import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Directory and file path
data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'tracking_parrot')
file_path = os.path.join(data_dir, 'drone_data.csv')

# Load the CSV data
df = pd.read_csv(file_path)

# Ensure all columns are in numeric format (converts non-numeric entries to NaN)
for col in ['Timestamp', 
            'Parrot X', 'Anafi X', 'Input X',
            'Parrot Y', 'Anafi Y', 'Input Y',
            'Parrot Z', 'Anafi Z', 'Input Z']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# Filter the data between time 25 and 75
df = df[(df['Timestamp'] >= 25) & (df['Timestamp'] <= 75)]

# Adjust the timestamp to set 25 as zero
df['Timestamp'] = df['Timestamp'] - 25
df['Parrot X'] = df['Parrot X'] - 3

# Convert to numpy arrays to ensure 1D format
timestamp = np.array(df['Timestamp'])
parrot_x = np.array(df['Parrot X'])
anafi_x = np.array(df['Anafi X'])
input_x = np.array(df['Input X'])
parrot_y = np.array(df['Parrot Y'])
anafi_y = np.array(df['Anafi Y'])
input_y = np.array(df['Input Y'])
parrot_z = np.array(df['Parrot Z'])
anafi_z = np.array(df['Anafi Z'])
input_z = np.array(df['Input Z'])

# Create a 3x1 figure
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot X-coordinates
axs[0].plot(timestamp, parrot_x, label='Parrot X', color='blue')
axs[0].plot(timestamp, anafi_x, label='Anafi X', color='green')
axs[0].set_ylabel('X-axis Position')
axs[0].legend(loc='upper left')
axs[0].set_title('Drone X, Y, Z vs. Adjusted Time')

# Plot Y-coordinates
axs[1].plot(timestamp, parrot_y, label='Parrot Y', color='blue')
axs[1].plot(timestamp, anafi_y, label='Anafi Y', color='green')
axs[1].set_ylabel('Y-axis Position')
axs[1].legend(loc='upper left')

# Plot Z-coordinates
axs[2].plot(timestamp, parrot_z, label='Parrot Z', color='blue')
axs[2].plot(timestamp, anafi_z, label='Anafi Z', color='green')
axs[2].set_ylabel('Z-axis Position')
axs[2].legend(loc='upper left')
axs[2].set_xlabel('Adjusted Time (s)')

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(data_dir, 'drone_plot_adjusted.png'))
plt.show()
