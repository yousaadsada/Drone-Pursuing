import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import pandas as pd
from scipy.signal import butter, filtfilt

fs = 25

# Define directories
save_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'real_drone_state_function','smoothed_data02')
os.makedirs(save_dir, exist_ok=True)

csv_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'real_drone_state_function', 'raw_data02')
csv_file_x = os.path.join(csv_dir, 'state_data_1.csv')
csv_file_y = os.path.join(csv_dir, 'state_data_2.csv')
csv_file_z = os.path.join(csv_dir, 'state_data_3.csv')
csv_file_yaw = os.path.join(csv_dir, 'state_data_4.csv')

# Load data from CSV files
df_x = pd.read_csv(csv_file_x)
df_y = pd.read_csv(csv_file_y)
df_z = pd.read_csv(csv_file_z)
df_yaw = pd.read_csv(csv_file_yaw)

# Adjust y data length
df_y = df_y[:-1]

# Store data in a dictionary
data = {
    'x': df_x['x'].to_numpy(),
    'y': df_y['y'].to_numpy(),
    'z': df_z['z'].to_numpy(),
    'yaw': df_yaw['pitch'].to_numpy(),
    'control_x': df_x['control_x'].to_numpy(),
    'control_y': df_y['control_y'].to_numpy(),
    'control_z': df_z['control_z'].to_numpy(),
    'control_yaw': df_yaw['control_yaw'].to_numpy(),
}

# Time data
times = {
    'x': df_x['elapsed_time'].to_numpy(),
    'y': df_y['elapsed_time'].to_numpy(),
    'z': df_z['elapsed_time'].to_numpy(),
    'yaw': df_yaw['elapsed_time'].to_numpy(),

}

keys = ['x','y','z','yaw','control_x','control_y','control_z','control_yaw']
position_fft = {}
frequencies = {}
smoothed_position = {}
speeds = {}
controls = {}

# Frequency Spectrum and Smoothing
for index, axis in enumerate(keys):
    if index <= 3:
        position_fft[axis] = fft(data[axis])
        n = len(data[axis])
        frequencies[axis] = fftfreq(n, d=1/fs)
        
        # Set cutoff frequency based on the axis
        cutoff = 2 if index == 0 else 2 if index == 1 else 4 if index == 2 else 3

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        smoothed_position[axis] = filtfilt(b, a, data[axis])

        delta_t = 0.04
        speeds[axis] = np.diff(smoothed_position[axis]) / delta_t
    
    else:
        controls[axis] = data[axis]

# Saving smoothed position and speed data to CSV files
for axis in keys[:4]:  # Loop through x, y, z, yaw
    # Create DataFrames for position
    smoothed_df = pd.DataFrame({
        'Time': times[axis][:-1],
        f'Smoothed_{axis.upper()}_Position': smoothed_position[axis][:-1],
        f'Smoothed_{axis.upper()}_Speed': speeds[axis],
        f'Smoothed_{axis.upper()}_Control': controls[f'control_{axis}'][:-1]
    })

    # Create DataFrames for speed, including control data


    # Save to CSV
    smoothed_df.to_csv(os.path.join(save_dir, f'smoothed_{axis}_state.csv'), index=False)


# Plotting Frequency Spectrum
fig, axs = plt.subplots(4, 1, figsize=(12, 15))

for index, axis in enumerate(keys[:4]):  # Only plot x, y, z, yaw
    axs[index].plot(frequencies[axis][:n//2], np.abs(position_fft[axis])[:n//2])
    axs[index].set_title(f'{axis.upper()} Position Frequency Spectrum')
    axs[index].set_xlabel('Frequency (Hz)')
    axs[index].set_ylabel('Amplitude')
    axs[index].grid()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# Plotting Smoothed Position and Speed
fig, axs = plt.subplots(4, 2, figsize=(12, 15))

for index, axis in enumerate(keys[:4]):  # Only plot x, y, z, yaw
    axs[index, 0].plot(times[axis], smoothed_position[axis])
    axs[index, 0].set_title(f'Smoothed {axis.upper()} Position')
    axs[index, 0].set_xlabel('Time (s)')
    axs[index, 0].set_ylabel(f'{axis.upper()} Position')

    time_for_speed = times[axis][:-1]
    axs[index, 1].plot(time_for_speed, speeds[axis])
    axs[index, 1].set_title(f'Smoothed {axis.upper()} Speed')
    axs[index, 1].set_xlabel('Time (s)')
    axs[index, 1].set_ylabel(f'{axis.upper()} Speed')

plt.tight_layout()
plt.show()
