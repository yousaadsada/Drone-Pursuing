import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Directory and file path
data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','move2ref_point')
file_path = os.path.join(data_dir, 'drone_data.csv')

# Load the CSV data
df = pd.read_csv(file_path)

# Ensure all columns are in numeric format (converts non-numeric entries to NaN)
for col in ['Timestamp', 'Reference X', 'Current X', 'X_output',
            'Reference Y', 'Current Y', 'Y_output',
            'Reference Z', 'Current Z', 'Z_output',
            'Reference Yaw', 'Current Yaw', 'Yaw_output']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN values
df = df.dropna()

# Convert to numpy arrays to ensure 1D format
timestamp = np.array(df['Timestamp'])
ref_x = np.array(df['Reference X'])
curr_x = np.array(df['Current X'])
curr_x_input = np.array(df['X_output'])
ref_y = np.array(df['Reference Y'])
curr_y = np.array(df['Current Y'])
curr_y_input = np.array(df['Y_output'])
ref_z = np.array(df['Reference Z'])
curr_z = np.array(df['Current Z'])
curr_z_input = np.array(df['Z_output'])
ref_yaw = np.array(df['Reference Yaw'])
curr_yaw = np.array(df['Current Yaw'])
curr_yaw_input = np.array(df['Yaw_output'])

# Create a 4x1 figure
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot X-coordinates with secondary y-axis for X_output
ax1 = axs[0]
ax2 = ax1.twinx()
ax1.plot(timestamp, ref_x, label='Reference X', color='blue')
ax1.plot(timestamp, curr_x, label='Position X', color='green')
ax2.plot(timestamp, curr_x_input, label='Input X', color='red', linestyle='--')
ax1.set_ylabel('X-axis Position')
ax2.set_ylabel('X-axis Input')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_title('Drone X, Y, Z, Yaw vs. Time')

# Plot Y-coordinates with secondary y-axis for Y_output
ax1 = axs[1]
ax2 = ax1.twinx()
ax1.plot(timestamp, ref_y, label='Reference Y', color='blue')
ax1.plot(timestamp, curr_y, label='Position Y', color='green')
ax2.plot(timestamp, curr_y_input, label='Input Y', color='red', linestyle='--')
ax1.set_ylabel('Y-axis Position')
ax2.set_ylabel('Y-axis Input')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot Z-coordinates with secondary y-axis for Z_output
ax1 = axs[2]
ax2 = ax1.twinx()
ax1.plot(timestamp, ref_z, label='Reference Z', color='blue')
ax1.plot(timestamp, curr_z, label='Position Z', color='green')
ax2.plot(timestamp, curr_z_input, label='Input Z', color='red', linestyle='--')
ax1.set_ylabel('Z-axis Position')
ax2.set_ylabel('Z-axis Input')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot Yaw with secondary y-axis for Yaw_output
ax1 = axs[3]
ax2 = ax1.twinx()
ax1.plot(timestamp, ref_yaw, label='Reference Yaw', color='blue')
ax1.plot(timestamp, curr_yaw, label='Position Yaw', color='green')
ax2.plot(timestamp, curr_yaw_input, label='Input Yaw', color='red', linestyle='--')
ax1.set_ylabel('Yaw Position')
ax2.set_ylabel('Yaw Input')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Set shared x-axis label for all subplots
axs[3].set_xlabel('Timestamp')

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(data_dir, 'drone_plot.png'))
plt.show()
